import torch
from torch import nn
from torch_geometric.nn import TransformerConv, GCNConv
from data import *
import torch.nn.functional as F

def ensure_pos_emb_capacity(model, need):
   cur = model.pos_emb.num_embeddings
   if need <= cur:
       return
   new_emb = nn.Embedding(need, model.pos_emb.embedding_dim).to(model.pos_emb.weight.device)
   with torch.no_grad():
       new_emb.weight[:cur] = model.pos_emb.weight
       nn.init.normal_(new_emb.weight[cur:], std=0.02)
   model.pos_emb = new_emb

class GraphTransformer(nn.Module):
    def __init__(
        self,
        expr_path: str, label_path: str, tf_path: str, priors_path: str,
        tf_bind_path: str, corr_threshold: float = 0.1,
        hidden_dim: int = 128, heads: int = 7, num_layers: int = 7,
        dropout: float = 0.2, lr: float = 5e-4, seed: int = 42
    ):
        super().__init__()
        self.seed = int(seed)
        self.rng = torch.Generator(device="cuda").manual_seed(self.seed)
        self.bilin = nn.Bilinear(hidden_dim, hidden_dim, 1, bias=True)
        self.cls_mix = nn.Parameter(torch.tensor(0.5))
        self.edge_gate = nn.Parameter(torch.tensor(0.3))  # start conservative
        self.cos_mix = nn.Parameter(torch.tensor(0.2))

        # ── 1) Load data ────────────────────────────────────────────────────────
        self.data = load_data(
            expr_path=expr_path,
            label_path=label_path,
            tf_path=tf_path,
            priors_path=priors_path,
            tf_bind_path=tf_bind_path,
            tf_bind2_path="/home/tmnthc/miniconda3/GRN/fimo_with_genenames_noduplicates.tsv",   
            corr_threshold=corr_threshold,
            anno_csv_path="/home/tmnthc/miniconda3/GRN/real_test.csv",
            llama_model="sentence-transformers/all-MiniLM-L6-v2" ,  
        )

        # ── 2) Positional / structural embedding ───────────────────────────────
        num_nodes     = self.data.x.size(0)
        orig_feat_dim = self.data.x.size(1)
        self.pos_emb  = nn.Embedding(num_nodes, hidden_dim)

        # ── 3) Input & edge dims ───────────────────────────────────────────────
        in_dim   = orig_feat_dim + hidden_dim
        edge_dim = self.data.edge_attr.size(1)

        # ── 4) Edge loss (classifier) criterion ────────────────────────────────
        y = self.data.y  # [E]
        n_pos = (y == 1).sum().item()
        n_neg = (y == 0).sum().item()
        alpha = n_neg / (n_pos + n_neg + 1e-8) if n_pos > 0 else 0.5

        # ── 5) Node feature projection ────────────────────────────────────────
        self.node_lin = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, hidden_dim * 2),
            nn.GLU(dim=-1),
            nn.Dropout(dropout),
        )

        # ── 6) TransformerConv stack ──────────────────────────────────────────
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                TransformerConv(
                    hidden_dim, hidden_dim, heads=heads, concat=False,
                    edge_dim=edge_dim, dropout=dropout
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        # ── 7) Edge attribute updater ─────────────────────────────────────────
        self.edge_updater = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim + edge_dim),
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, edge_dim),
        )

        # ── 8) Gene expression head ───────────────────────────────────────────
        self.expr_dim = self.data.expr_dim
        self.expr_pred_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.expr_dim),
        )

        # ── 9) Edge classifier head ───────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(4 * hidden_dim),             
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.dropout = dropout
        self.lr = lr

 
    def _encode(self, x, edge_index, edge_attr, training: bool):
        device = x.device
        idx = torch.arange(x.size(0), device=device)
        p   = self.pos_emb(idx)
        x   = torch.cat([x, p], dim=1)          
        h   = self.node_lin(x)               

        e = edge_attr
        for conv, norm in zip(self.convs, self.norms):
            e_drop = F.dropout(e, p=self.dropout, training=training)
            h_in = h
            h = conv(h, edge_index, e_drop)
            h = norm(h + h_in)
            h = F.relu(h)

            row, col = edge_index
            h_i, h_j = h[row], h[col]
            e_new = self.edge_updater(torch.cat([h_i, h_j, e_drop], dim=-1))
            e     = e_drop + torch.sigmoid(self.edge_gate) * e_new   

        return h, e

    def reinit_edge_head(self):
        import math, torch.nn as nn
        def _reset_linear_or_norm(m):
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
            else:
                for p in m.parameters(recurse=False):
                    if p.dim() >= 2:
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    else:
                        nn.init.zeros_(p)

        if hasattr(self, "classifier") and isinstance(self.classifier, nn.Sequential):
            for m in self.classifier:
                if isinstance(m, (nn.Linear, nn.LayerNorm)):
                    _reset_linear_or_norm(m)


        if hasattr(self, "bilin") and self.bilin is not None:
            if hasattr(self.bilin, "reset_parameters"):
                self.bilin.reset_parameters()
            else:
                for p in self.bilin.parameters():
                    if p.dim() >= 2:
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    else:
                        nn.init.zeros_(p)

    
        if hasattr(self, "cls_mix") and self.cls_mix is not None:
            with torch.no_grad():
                self.cls_mix.fill_(0.5)  


    def encoder_modules(self):
    
        return nn.ModuleList([self.pos_emb, self.node_lin, self.convs, self.norms, self.edge_updater])

    def freeze_encoder(self, frozen: bool = True):
        for p in self.pos_emb.parameters():    p.requires_grad = (not frozen)
        for p in self.node_lin.parameters():   p.requires_grad = (not frozen)
        for m in self.convs:                   [setattr(p, 'requires_grad', (not frozen)) for p in m.parameters()]
        for m in self.norms:                   [setattr(p, 'requires_grad', (not frozen)) for p in m.parameters()]
        for p in self.edge_updater.parameters(): p.requires_grad = (not frozen)

    def forward(self, mask_ratio: float = 0.0, task: str = "edge", split: str | None = None):
       
        device = next(self.parameters()).device
        x  = self.data.x.to(device)
        ei = self.data.edge_index.to(device)
        ea = self.data.edge_attr.to(device)

        if self.training:

            x = x + torch.randn_like(x) * 0.1

        def _nodes_touched_by(mask_e: torch.Tensor | None) -> torch.Tensor:
         
            N = x.size(0)
            touched = torch.zeros(N, dtype=torch.bool, device=device)
            if mask_e is None:
                row, col = ei
            else:
                sel = torch.nonzero(mask_e, as_tuple=False).view(-1)
                if sel.numel() == 0:
                    return touched
                row, col = ei[:, sel]
            touched[row] = True
            touched[col] = True
            return touched

        def _nodes_in_split(split_name: str | None) -> torch.Tensor:
            
            if split_name is None:
                return _nodes_touched_by(None)
            if split_name == "train" and hasattr(self.data, "train_mask"):
                return _nodes_touched_by(self.data.train_mask)
            if split_name == "val"   and hasattr(self.data, "val_mask"):
                return _nodes_touched_by(self.data.val_mask)
            if split_name == "test"  and hasattr(self.data, "test_mask"):
                return _nodes_touched_by(self.data.test_mask)
            return _nodes_touched_by(None)
        
        def make_node_mask(N: int) -> torch.Tensor:

            if mask_ratio <= 0.0:
                return torch.zeros(N, dtype=torch.bool, device=device)

           
            split_nodes = _nodes_in_split(split)   
            excluded = set()
            if hasattr(self, "_fixed_mask_sets"):
                for s in ("val", "test"):
                    fs = self._fixed_mask_sets.get(s)
                    if isinstance(fs, set):
                        excluded |= fs

            if excluded:
                subset = getattr(self.data, "subset", None)  
                if subset is not None:
                    excl_local = torch.tensor(
                        [int(g.item()) in excluded for g in subset],
                        dtype=torch.bool, device=device
                    )
                else:
                    excl_local = torch.zeros(N, dtype=torch.bool, device=device)
                    keep = [i for i in excluded if 0 <= i < N]
                    if keep:
                        excl_local[torch.as_tensor(keep, device=device, dtype=torch.long)] = True

                split_nodes = split_nodes & (~excl_local)


            total = int(split_nodes.sum().item())
            if total == 0:
                return torch.zeros(N, dtype=torch.bool, device=device)

            k = int(round(mask_ratio * total))
            if k <= 0 and mask_ratio > 0.0:
                k = 1
            k = min(k, total)

            elig_idx = torch.nonzero(split_nodes, as_tuple=False).view(-1)
            perm = torch.randperm(elig_idx.numel(), device=device, generator=self.rng)
            chosen = elig_idx[perm[:k]]
            mask = torch.zeros(N, dtype=torch.bool, device=device)
            mask[chosen] = True
            return mask
        
        if not hasattr(self, "_fixed_mask_sets"):
           
            self._fixed_mask_sets = {"train": None, "val": None, "test": None}

        def _fixed_mask_for(split_name: str, ratio: float) -> torch.Tensor:

            subset = getattr(self.data, "subset", None) 

            local_elig = _nodes_in_split(split_name)   
            if subset is not None:
                elig_global = subset[local_elig]       
            else:
                elig_global = torch.nonzero(local_elig, as_tuple=False).view(-1)

            if self._fixed_mask_sets.get(split_name) is None:
                exclude = set()
                if split_name == "test":
                    for s in ("train", "val"):
                        if isinstance(self._fixed_mask_sets.get(s), set):
                            exclude |= self._fixed_mask_sets[s]

                elig_list = [int(i) for i in elig_global.tolist() if int(i) not in exclude]
                k = max(1, int(round(ratio * len(elig_list)))) if (ratio > 0.0 and len(elig_list) > 0) else 0

                chosen = set()
                if k > 0:
                    perm = torch.randperm(len(elig_list), device=device)[:k].tolist()
                    chosen = { elig_list[j] for j in perm }
                self._fixed_mask_sets[split_name] = chosen  


            fixed_set = self._fixed_mask_sets[split_name]
            if subset is not None:
                mask_local = torch.tensor([int(gid.item()) in fixed_set for gid in subset],
                                          dtype=torch.bool, device=device)
            else:
                N_here = x.size(0)
                mask_local = torch.zeros(N_here, dtype=torch.bool, device=device)
                if isinstance(fixed_set, set):
                    keep = [i for i in fixed_set if 0 <= i < N_here]
                    if keep:
                        mask_local[torch.as_tensor(keep, device=device, dtype=torch.long)] = True

            touched = torch.zeros_like(mask_local, dtype=torch.bool, device=device)
            r, c = ei
            touched[r] = True; touched[c] = True
            return mask_local & touched

        if task == "edge":

            h_e, _ = self._encode(x, ei, ea, self.training)
            row, col = ei
            hi, hj = h_e[row], h_e[col]
            had   = hi * hj
            diff1 = (hi - hj).abs()
            edge_repr = torch.cat([hi, hj, had, diff1], dim=-1) 


            mlp_score = self.classifier(edge_repr).squeeze(-1)

            bil_score = None
            if hasattr(self, "bilin") and self.bilin is not None:
                bil_score = self.bilin(hi, hj).squeeze(-1)

            if bil_score is not None and hasattr(self, "cls_mix") and self.cls_mix is not None:
                mix = torch.sigmoid(self.cls_mix)
                
                edge_logits = mix * mlp_score + (1.0 - mix) * bil_score
                cos = F.cosine_similarity(hi, hj, dim=-1)                 
                edge_logits = edge_logits + torch.tanh(self.cos_mix) * cos

            else:
                edge_logits = mlp_score
                cos = F.cosine_similarity(hi, hj, dim=-1)                 
                edge_logits = edge_logits + torch.tanh(self.cos_mix) * cos
            return edge_logits

        if task == "gene_expression":
            N = x.size(0)
            if mask_ratio > 0.0:
                mask_nodes = _fixed_mask_for(split or "train", mask_ratio) if split is not None else make_node_mask(N)
                x_masked = x.clone()
                x_masked[mask_nodes, :self.expr_dim] = 0.0     # zero only expression channels
            else:
                mask_nodes = torch.zeros(N, dtype=torch.bool, device=device)
                x_masked = x

            h_g, _ = self._encode(x_masked, ei, ea, self.training)
            expr_pred = self.expr_pred_head(h_g)
            return expr_pred, mask_nodes

        if task == "both":

            h_e, _ = self._encode(x, ei, ea, self.training)
            row, col = ei
            hi, hj = h_e[row], h_e[col]
            edge_repr = torch.cat([hi, hj], dim=-1)

            mlp_score = self.classifier(edge_repr).squeeze(-1)

            bil_score = None
            if hasattr(self, "bilin") and self.bilin is not None:
                bil_score = self.bilin(hi, hj).squeeze(-1)

            if bil_score is not None and hasattr(self, "cls_mix") and self.cls_mix is not None:
                mix = torch.sigmoid(self.cls_mix)
                edge_logits = mix * mlp_score + (1.0 - mix) * bil_score
                cos = F.cosine_similarity(hi, hj, dim=-1)                
                edge_logits = edge_logits + torch.tanh(self.cos_mix) * cos

            else:
                edge_logits = mlp_score
                cos = F.cosine_similarity(hi, hj, dim=-1)                
                edge_logits = edge_logits + torch.tanh(self.cos_mix) * cos


            N = x.size(0)
            if mask_ratio > 0.0:
                mask_nodes = _fixed_mask_for(split or "train", mask_ratio) if split is not None else make_node_mask(N)

                x_masked = x.clone()
                x_masked[mask_nodes, :self.expr_dim] = 0.0
            else:
                mask_nodes = torch.zeros(N, dtype=torch.bool, device=device)
                x_masked = x

            h_g, _ = self._encode(x_masked, ei, ea, self.training)
            expr_pred = self.expr_pred_head(h_g)
            return edge_logits, expr_pred, mask_nodes

        raise ValueError("task must be 'edge', 'gene_expression', or 'both'")



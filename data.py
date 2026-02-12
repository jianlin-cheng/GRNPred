import os
import numpy as np
import pandas as pd
from torch_geometric.data import Data
import torch
import random
from sklearn.preprocessing import MultiLabelBinarizer
from torch_geometric.data import Data
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
from torch_geometric.utils import k_hop_subgraph

def extract_tf_subgraphs(data, num_hops=5):
        subgraphs = {}
        N = data.x.size(0)
        for tf_idx in data.tf_indices:
            subset, ei, mapping, mask = k_hop_subgraph(
                node_idx=tf_idx,
                num_hops=num_hops,
                edge_index=data.edge_index,
                relabel_nodes=True,
                num_nodes=N,
                flow='source_to_target'
            )
            subgraphs[tf_idx] = {
                "x": data.x[subset],
                "edge_index": ei,
                "edge_attr": data.edge_attr[mask],
                "y": data.y[mask],
                "train_mask": data.train_mask[mask],
                "val_mask":   data.val_mask[mask],
                "test_mask":  data.test_mask[mask],
                "subset":     subset,
            }
        return subgraphs

def _sanitize_graph(g):
    g.x         = torch.nan_to_num(g.x, nan=0.0, posinf=1e6, neginf=-1e6)
    if hasattr(g, "edge_attr") and g.edge_attr is not None:
        g.edge_attr = torch.nan_to_num(g.edge_attr, nan=0.0, posinf=1e6, neginf=-1e6)
    if hasattr(g, "y") and g.y is not None:
        g.y = torch.nan_to_num(g.y, nan=0.0, posinf=1.0, neginf=0.0)
    return g

def match_feature_dim(g: Data, target_dim: int) -> Data:
    cur = g.x.size(1)
    if cur == target_dim:
        return g
    if cur < target_dim:
        pad = torch.zeros(g.x.size(0), target_dim - cur, dtype=g.x.dtype, device=g.x.device)
        g.x = torch.cat([g.x, pad], dim=1)
    else:
        g.x = g.x[:, :target_dim]
    return g

def make_paths(base_name):
    return (
        f"{base_name}_ExpressionData.csv",
        f"{base_name}_refNetwork.csv",
        f"{base_name}_TFs.csv",
        "go_term.csv",
        "tf_binding_logodds.csv"
    )


def load_data(
    expr_path:      str,
    label_path:     str,
    tf_path:        str,
    priors_path:    str,
    tf_bind_path:   str,
    tf_bind2_path:  str | None = None,      
    corr_threshold: float = 0.1,
    anno_csv_path:  str | None = None,
    llama_model:    str | None = None,
    llama_batch_size: int = 16,
    llama_max_tokens: int = 256):

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    def _build_llama_text_embeddings(
        anno_csv_path: str,
        genes: list[str],
        model_name: str = "meta-llama/Llama-3.2-1B",
        batch_size: int = 16,
        max_tokens: int = 256,
        cache_key: str | None = None,
    ) -> np.ndarray:


        device = "cuda" if torch.cuda.is_available() else "cpu"
        cache_dir = os.path.join(os.path.dirname(anno_csv_path), ".cache_llama_embeds")
        os.makedirs(cache_dir, exist_ok=True)
        model_tag = model_name.split("/")[-1].replace(".", "_")
        key = cache_key or os.path.splitext(os.path.basename(anno_csv_path))[0]

        df = pd.read_csv(anno_csv_path)
        if "gene" not in df.columns:
            raise ValueError("Annotation CSV must contain a 'gene' column.")
        df["gene"] = df["gene"].astype(str)
        non_gene_cols = [c for c in df.columns if c != "gene"]
        for c in non_gene_cols:
            df[c] = df[c].fillna("").astype(str).str.strip()
        merged_text = df[non_gene_cols].apply(
            lambda r: ". ".join([t for t in r.tolist() if t]), axis=1
        )

        gene_to_text = dict(zip(df["gene"], merged_text))
        gene_to_text_ci = {str(k).strip().upper(): v for k, v in gene_to_text.items()}
        if "entry_name" in df.columns:
            for k, v in zip(df["entry_name"].astype(str), merged_text):
                if k and v:
                    gene_to_text.setdefault(k, v)
                    gene_to_text_ci.setdefault(k.strip().upper(), v)
        if "accession" in df.columns:
            for k, v in zip(df["accession"].astype(str), merged_text):
                if k and v:
                    gene_to_text.setdefault(k, v)
                    gene_to_text_ci.setdefault(k.strip().upper(), v)

        texts, hits = [], 0
        for g in genes:
            t = gene_to_text.get(g) or gene_to_text_ci.get(str(g).strip().upper()) or ""
            hits += int(bool(t))
            texts.append(t)
        print(f"[TEXT-EMB] matched annotations: {hits} / {len(genes)}")

        def _encode_with(model_name_try: str) -> np.ndarray:
            mt = model_name_try.split("/")[-1].replace(".", "_")
            cp = os.path.join(cache_dir, f"{mt}_{key}_embeds.npy")
            if os.path.exists(cp):
                arr = np.load(cp)
                if arr.shape[0] == len(genes):
                    return arr
            tok = AutoTokenizer.from_pretrained(model_name_try, use_fast=True)
            mdl = AutoModel.from_pretrained(model_name_try).to(device)
            mdl.eval()
            outs = []
            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    enc = tok(batch, padding=True, truncation=True, max_length=max_tokens, return_tensors="pt").to(device)
                    out = mdl(**enc)
                    hidden = out.last_hidden_state
                    mask = enc.attention_mask.unsqueeze(-1).float()
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                    outs.append(pooled.detach().cpu())
            emb = torch.cat(outs, dim=0).numpy()
            empties = np.array([t == "" for t in texts], dtype=bool)
            if empties.any():
                emb[empties] = 0.0
            np.save(cp, emb)
            return emb

        fallbacks = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "intfloat/e5-base-v2",
            "BAAI/bge-small-en-v1.5",
        ]
        tried = []
        for candidate in [model_name] + fallbacks:
            try:
                return _encode_with(candidate)
            except (GatedRepoError, HfHubHTTPError, OSError) as e:
                msg = str(e).lower()
                if any(k in msg for k in ["gated", "401", "access to model"]):
                    tried.append((candidate, "gated"))
                    continue
                raise
        raise RuntimeError(f"Could not load any embedding model (tried {tried}).")

    expr_df     = pd.read_csv(expr_path, index_col=0)
    genes       = expr_df.index.tolist()
    expr_np     = expr_df.values

    gene_to_idx = {g: i for i, g in enumerate(genes)}
    N = len(genes)

    tf_df      = pd.read_csv(tf_path)
    tf_list    = tf_df.iloc[:, 0].astype(str).tolist()
    tf_indices = [gene_to_idx[g] for g in tf_list if g in gene_to_idx]
    tf_set     = set(tf_indices)

    # Load new file with columns: gene, go_term
    priors_df = pd.read_csv(priors_path)  # priors_path -> "gene_go_pairs.csv"
    
    # Filter to keep only genes present in your dataset
    priors_df = priors_df[priors_df["gene"].isin(genes)]
    
    # Build mapping: gene -> set of GO terms
    gene2go = defaultdict(set)
    for _, r in priors_df.iterrows():
        if pd.notna(r.get("go_term")):
            gene2go[r["gene"]].add(str(r["go_term"]))
    
    # Create binary GO-term feature matrix aligned with the 'genes' list
    go_mat = MultiLabelBinarizer(sparse_output=False).fit_transform(
        [sorted(gene2go.get(g, set())) for g in genes]
    ).astype(np.float32, copy=False)

    corr_csv = os.path.splitext(expr_path)[0] + "_pearson_correlation.csv"
    if os.path.exists(corr_csv):
        corr_mat = (
            pd.read_csv(corr_csv, index_col=0)
            .reindex(index=genes, columns=genes, fill_value=0)
            .values
        )
    else:
        corr_mat = np.corrcoef(expr_np)
        pd.DataFrame(corr_mat, index=genes, columns=genes).to_csv(corr_csv)

    # (old file) motif,gene,best_score
    bind_df = pd.read_csv(tf_bind_path)
    bind_df = bind_df[bind_df["motif"].isin(genes) & bind_df["gene"].isin(genes)]
    bind_dict = {
        (gene_to_idx[r.motif], gene_to_idx[r.gene]): float(r.best_score)
        for r in bind_df.itertuples()
    }

    bind2_dict = None
    if tf_bind2_path is not None:
        b2 = pd.read_csv(tf_bind2_path, sep=None, engine="python")
        cols = set(b2.columns)

        def _idx(x):
            if pd.isna(x): return None
            s = str(x)
            return gene_to_idx.get(s) or gene_to_idx.get(s.split(".", 1)[0])

        if {"motif_alt_id","sequence_name"}.issubset(cols):    
            if "score" in cols:
                b2 = b2[["motif_alt_id","sequence_name","score"]].rename(
                    columns={"motif_alt_id":"motif","sequence_name":"gene","score":"best_score"})
            elif "p-value" in cols:
                b2 = b2[["motif_alt_id","sequence_name","p-value"]].rename(
                    columns={"motif_alt_id":"motif","sequence_name":"gene"})
                b2["best_score"] = -np.log10(b2["p-value"].astype(float).clip(lower=1e-300))
            else:
                raise ValueError("tf_bind2_path: need 'score' or 'p-value' in FIMO file")
        elif {"gene","tf_name","score_max"}.issubset(cols):        # old format
            b2 = b2.rename(columns={"tf_name":"motif","score_max":"best_score"})
        else:
            raise ValueError("tf_bind2_path: unrecognized columns")

        b2["s"] = b2["motif"].map(_idx)
        b2["d"] = b2["gene"].map(_idx)
        b2 = b2.dropna(subset=["s","d","best_score"]).groupby(["s","d"], as_index=False)["best_score"].max()
        bind2_dict = {(int(r.s), int(r.d)): float(r.best_score) for r in b2.itertuples(index=False)}

    
  
    # ---------------------------------------------------------
    # Build graph edges STRICTLY by correlation threshold
    # Drop ALL edges with corr < corr_threshold (no exceptions)
    # ---------------------------------------------------------
    iu, ju = np.triu_indices(N, k=1)
    mask = corr_mat[iu, ju] >= corr_threshold

    src = np.concatenate([iu[mask], ju[mask]])
    dst = np.concatenate([ju[mask], iu[mask]])
    corr_vals = np.concatenate([corr_mat[iu[mask], ju[mask]]] * 2).astype(np.float32, copy=False)

    bind_vals = np.array(
        [bind_dict.get((s, d), 0.0) for s, d in zip(src, dst)],
        dtype=np.float32
    )

    if bind2_dict is not None:
        bind2_vals = np.array(
            [bind2_dict.get((s, d), 0.0) for s, d in zip(src, dst)],
            dtype=np.float32
        )

    # ---------------------------------------------------------
    # Positives are defined ONLY on the kept edges (corr >= threshold)
    # ---------------------------------------------------------
    labels_df = pd.read_csv(label_path)

    pos_set = {
        (gene_to_idx[r.Gene1], gene_to_idx[r.Gene2])
        for r in labels_df.itertuples()
        if (r.Gene1 in gene_to_idx)
        and (r.Gene2 in gene_to_idx)
        and (gene_to_idx[r.Gene1] in tf_set)
    }

    # Build y on current edge list only (no adding missing positives)
    E0 = len(src)
    y_np = np.fromiter(
        ((s, d) in pos_set for s, d in zip(src, dst)),
        dtype=np.float32,
        count=E0
    )

    # Sanity check (should be zero)
    n_bad = int((corr_vals < corr_threshold).sum())
    if n_bad != 0:
        raise RuntimeError(f"Found {n_bad} edges with corr < corr_threshold after filtering (bug).")

 

    E0 = len(src)
    y_np = np.fromiter(((s, d) in pos_set for s, d in zip(src, dst)),
                    dtype=np.float32, count=E0)

    tf_to_edges = defaultdict(list)
    for e, (s, d) in enumerate(zip(src, dst)):
        if s in tf_set: tf_to_edges[s].append(e)
        if d in tf_set and d != s: tf_to_edges[d].append(e)

    total_edges = E0
    total_pos   = int(y_np.sum())
    targets = {
        "train": {"edges": int(round(0.70 * total_edges)), "pos": int(round(0.70 * total_pos))},
        "val":   {"edges": int(round(0.15 * total_edges)), "pos": int(round(0.15 * total_pos))},
        "test":  {"edges": total_edges,                   "pos": total_pos},
    }
    targets["test"]["edges"] -= (targets["train"]["edges"] + targets["val"]["edges"])
    targets["test"]["pos"]   -= (targets["train"]["pos"]   + targets["val"]["pos"])

    splits = {"train": set(), "val": set(), "test": set()}
    stats  = {"train": {"edges": 0, "pos": 0}, "val": {"edges": 0, "pos": 0}, "test": {"edges": 0, "pos": 0}}
    tf_order = sorted(tf_to_edges, key=lambda tf: (-len(tf_to_edges[tf]), -int(sum(y_np[e] for e in tf_to_edges[tf]))))
    for tf in tf_order:
        n_e = len(tf_to_edges[tf]); n_p = int(sum(y_np[e] for e in tf_to_edges[tf]))
        best_split, best_score = None, float("inf")
        for split in ("train", "val", "test"):
            frac_e = stats[split]["edges"] / (targets[split]["edges"] or 1)
            frac_p = stats[split]["pos"]   / (targets[split]["pos"]   or 1)
            score = frac_e + frac_p
            if score < best_score:
                best_score, best_split = score, split
        splits[best_split].add(tf)
        stats[best_split]["edges"] += n_e
        stats[best_split]["pos"]   += n_p

    tf_to_split = {tf: split for split in splits for tf in splits[split]}
    undirected_gene_gene = defaultdict(list)
    for e, (s, d) in enumerate(zip(src, dst)):
        if (s not in tf_set) and (d not in tf_set):
            undirected_gene_gene[frozenset((s, d))].append(e)

    all_gg = list(undirected_gene_gene.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(all_gg)
    n_train = int(0.7  * len(all_gg))
    n_val   = int(0.15 * len(all_gg))
    gg_split = {}
    for i, key in enumerate(all_gg):
        bucket = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
        for e in undirected_gene_gene[key]:
            gg_split[e] = bucket

    keep_mask = np.zeros(E0, dtype=bool)
    owned_split = np.empty(E0, dtype=object)

    # -------------------------
    # Assign each edge to a split (same logic as your code)
    # -------------------------
    for e, (s, d) in enumerate(zip(src, dst)):
        if (s in tf_set) and (d in tf_set):
            # TF-TF edge must belong to same TF split, otherwise drop
            if tf_to_split.get(s) != tf_to_split.get(d):
                owned_split[e] = None
                continue
            assigned = tf_to_split[s]
        elif s in tf_set:
            assigned = tf_to_split[s]
        elif d in tf_set:
            assigned = tf_to_split[d]
        else:
            assigned = gg_split.get(e, "train")

        owned_split[e] = assigned
        keep_mask[e] = True

    # ---------------------------------------------------------
    # Enforce: NO low-correlation positives in TEST
    # ---------------------------------------------------------
    # NOTE: y_np is already built on E0 edges (including any added missing positives)
    lowcorr_pos = (y_np == 1) & (corr_vals < corr_threshold)
    is_test     = (owned_split == "test")
    drop_test_lowcorr_pos = lowcorr_pos & is_test

    if drop_test_lowcorr_pos.any():
        print(f"[SPLIT-FIX] Dropping {int(drop_test_lowcorr_pos.sum())} "
              f"test positives with corr < {corr_threshold}.")
        keep_mask[drop_test_lowcorr_pos] = False
        owned_split[drop_test_lowcorr_pos] = None  # optional, for clarity/debugging

    # -------------------------
    # Apply keep_mask
    # -------------------------
    src         = src[keep_mask]
    dst         = dst[keep_mask]
    corr_vals   = corr_vals[keep_mask]
    bind_vals   = bind_vals[keep_mask]
    y_np        = y_np[keep_mask]
    owned_split = owned_split[keep_mask]
    if bind2_dict is not None:
        bind2_vals = bind2_vals[keep_mask]

    E = len(src)

    # -------------------------
    # Split masks (torch + bool)
    # -------------------------
    train_mask = torch.tensor([s == "train" for s in owned_split], dtype=torch.bool)
    val_mask   = torch.tensor([s == "val"   for s in owned_split], dtype=torch.bool)
    test_mask  = torch.tensor([s == "test"  for s in owned_split], dtype=torch.bool)

    train_mask_bool = (owned_split == "train")
    val_mask_bool   = (owned_split == "val")
    test_mask_bool  = (owned_split == "test")

    # -------------------------
    # Sanity check: must be zero
    # -------------------------
    bad_test = int(((y_np == 1) & (corr_vals < corr_threshold) & test_mask_bool).sum())
    print("[CHECK] low-corr positives in TEST:", bad_test)


    edge_cols = [corr_vals.astype(np.float32, copy=False),
                bind_vals.astype(np.float32, copy=False)]
    if tf_bind2_path is not None and 'bind2_vals' in locals():
        edge_cols.append(bind2_vals.astype(np.float32, copy=False))
    EF = np.stack(edge_cols, axis=1).astype(np.float32, copy=False)   # [E, F]


    EF[:, 0] = np.arctanh(np.clip(EF[:, 0], -0.999999, 0.999999))

    def _log1p_signed(x):
        pos = x >= 0
        out = np.empty_like(x, dtype=np.float32)
        out[pos]  = np.log1p(x[pos])
        out[~pos] = -np.log1p(np.abs(x[~pos]))
        return out

    EF[:, 1] = _log1p_signed(EF[:, 1])
    if EF.shape[1] >= 3:
        EF[:, 2] = _log1p_signed(EF[:, 2])


    eps = 1e-6
    mu  = EF[train_mask_bool].mean(axis=0)
    sd  = EF[train_mask_bool].std(axis=0)
    sd  = np.where(sd < eps, 1.0, sd)

    EF = (EF - mu) / sd
    EF = np.clip(EF, -5.0, 5.0)

    edge_attr = torch.tensor(EF, dtype=torch.float32)

    edge_norm_stats = {"mu": mu.astype(np.float32), "sd": sd.astype(np.float32),
                    "transform": "corr->atanh; bind->signed log1p; zscore(train); clip±5"}

    degrees = np.zeros(N, dtype=np.float32)
    for s, d in zip(src, dst):
        degrees[s] += 1; degrees[d] += 1

    tf_indicator = np.zeros((N, 1), dtype=np.float32)
    if tf_indices:
        tf_indicator[np.array(tf_indices, dtype=int)] = 1.0

    feats = [expr_np, go_mat.astype(np.float32, copy=False), degrees[:, None], tf_indicator]
    if anno_csv_path is not None:
        model_name = llama_model or "meta-llama/Llama-3.2-1B"
        text_emb = _build_llama_text_embeddings(
            anno_csv_path=anno_csv_path,
            genes=genes,
            model_name=model_name,
            batch_size=llama_batch_size,
            max_tokens=llama_max_tokens,
            cache_key=os.path.basename(expr_path),
        )
        feats.append(text_emb.astype(np.float32, copy=False))

    X = np.hstack(feats)
    x_tensor = torch.tensor(X, dtype=torch.float32)

    edge_feats = [corr_vals, bind_vals]
    if tf_bind2_path is not None and 'bind2_vals' in locals():
        edge_feats.append(bind2_vals)
    edge_attr  = torch.tensor(np.stack(edge_feats, axis=1), dtype=torch.float32)
    y_tensor   = torch.tensor(y_np, dtype=torch.float32)

    def _nodes_in(mask_t: torch.Tensor):
        idx = np.nonzero(mask_t.numpy())[0]
        nodes = set()
        for e in idx:
            nodes.add(int(src[e])); nodes.add(int(dst[e]))
        return nodes

    train_nodes = _nodes_in(train_mask)
    val_nodes   = _nodes_in(val_mask)
    test_nodes  = _nodes_in(test_mask)
    train_tfs, val_tfs, test_tfs = train_nodes & tf_set, val_nodes & tf_set, test_nodes & tf_set

    print("Undirected edge overlap (kept edges only):")
    def _undir(mask_bool: np.ndarray):
        return set(frozenset((int(src[i]), int(dst[i]))) for i in np.nonzero(mask_bool)[0])
    print("  Train/Test:", len(_undir(train_mask.numpy()) & _undir(test_mask.numpy())))
    print("  Train/Val: ", len(_undir(train_mask.numpy()) & _undir(val_mask.numpy())))
    print("  Val/Test:  ", len(_undir(val_mask.numpy()) & _undir(test_mask.numpy())))
    for name, m in (("train", train_mask), ("val", val_mask), ("test", test_mask)):
        print(f"Split {name}: {int(m.sum())} edges, {int(y_np[m.numpy()].sum())} positives")
    print("\nTF membership by split (unique TF nodes touched by edges):")
    print(f"  Train TFs: {len(train_tfs)}  Val TFs: {len(val_tfs)}  Test TFs: {len(test_tfs)}")

    data = Data(
        x          = x_tensor,
        edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long),
        edge_attr  = edge_attr,
        y          = y_tensor,
        train_mask = train_mask,
        val_mask   = val_mask,
        test_mask  = test_mask,
    )
    data.gene_names   = genes
    data.tf_indices   = tf_indices
    data.expr_dim     = expr_np.shape[1]
    return data

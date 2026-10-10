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
_SPLIT_SEED = {"train": 0, "val": 1, "test": 2}


def _bfs_hop_layers(center, edge_index, num_nodes, max_hops):
    """Return [layer0=TF, layer1=1-hop, ...] using only edge_index."""
    device = edge_index.device
    src, dst = edge_index[0], edge_index[1]
    seen = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    center = int(center)
    seen[center] = True
    layers = [torch.tensor([center], dtype=torch.long, device=device)]
    frontier = layers[0]
    for _ in range(max_hops):
        if frontier.numel() == 0 or src.numel() == 0:
            break
        is_front = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        is_front[frontier] = True
        neigh = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        neigh[dst[is_front[src]]] = True
        new_idx = (neigh & ~seen).nonzero(as_tuple=False).view(-1)
        if new_idx.numel() == 0:
            break
        layers.append(new_idx)
        seen[new_idx] = True
        frontier = new_idx
    return layers


def _hop_fill_groups(layers, max_nodes, generator):
    """Fill hops in order until max_nodes. Overflow hops are packed without replacement.

    Stays on the device of ``layers`` (CPU or CUDA).
    """
    device = layers[0].device
    tf = layers[0].reshape(-1)
    leftovers = [layer.reshape(-1).clone() for layer in layers[1:]]
    if all(layer.numel() == 0 for layer in leftovers):
        return [tf]

    groups = []
    while True:
        parts = [tf]
        n_keep = 1
        progressed = False
        for h, layer in enumerate(leftovers):
            if layer.numel() == 0:
                continue
            room = max_nodes - n_keep
            if room <= 0:
                break
            if layer.numel() <= room:
                parts.append(layer)
                n_keep += int(layer.numel())
                leftovers[h] = layer[:0]
                progressed = True
            else:
                # Generator may be CPU-only on some builds; fall back to seeded CPU perm.
                try:
                    perm = torch.randperm(layer.numel(), generator=generator, device=device)
                except (RuntimeError, TypeError):
                    perm = torch.randperm(layer.numel(), generator=generator).to(device)
                parts.append(layer[perm[:room]])
                leftovers[h] = layer[perm[room:]]
                n_keep += room
                progressed = True
                break
        if n_keep <= 1 and not progressed:
            break
        if n_keep <= 1:
            break
        groups.append(torch.cat(parts, dim=0))
        if all(layer.numel() == 0 for layer in leftovers):
            break
    return groups if groups else [tf]


def _hop_lookup(layers, num_nodes, device):
    """hop_of[node] = BFS layer index (0 = center TF); -1 for nodes outside the layers."""
    hop_of = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    for h, layer in enumerate(layers):
        if layer.numel() > 0:
            hop_of[layer.to(device)] = h
    return hop_of


def _tfs_touched_by_mask(data, split_mask):
    if not bool(split_mask.any()):
        return []
    # Keep unique node discovery on the edge_index device (GPU-friendly).
    src = data.edge_index[0, split_mask]
    dst = data.edge_index[1, split_mask]
    touched = torch.zeros(data.x.size(0), dtype=torch.bool, device=src.device)
    touched[src] = True
    touched[dst] = True
    out = []
    for tf in data.tf_indices:
        ti = int(tf)
        if ti < touched.numel() and bool(touched[ti]):
            out.append(ti)
    return out


def _record_from_global_nodes(data, keep_global, split_name, hop_dist):
    """Induce all coexpression edges among keep_global; copy global ChIP-split masks.

    Message passing sees the full |corr| graph among the subset. Train/val/test
    masks stay on the original TF-wise label split and are used only for scoring.
    All tensor ops follow ``data.edge_index.device`` (CPU or CUDA).
    """
    device = data.edge_index.device
    keep_global = keep_global.long().reshape(-1).to(device)
    N = data.x.size(0)
    new_id = torch.full((N,), -1, dtype=torch.long, device=device)
    new_id[keep_global] = torch.arange(keep_global.numel(), device=device)
    src = data.edge_index[0]
    dst = data.edge_index[1]
    edge_keep = (new_id[src] >= 0) & (new_id[dst] >= 0)
    ei = torch.stack([new_id[src[edge_keep]], new_id[dst[edge_keep]]], dim=0)
    train_m = data.train_mask.to(device)[edge_keep]
    val_m = data.val_mask.to(device)[edge_keep]
    test_m = data.test_mask.to(device)[edge_keep]
    y = data.y.to(device)[edge_keep]
    ea = data.edge_attr.to(device)[edge_keep]
    x = data.x.to(device)[keep_global]
    hops = hop_dist.to(device)
    return {
        "x": x,
        "edge_index": ei,
        "edge_attr": ea,
        "y": y,
        "train_mask": train_m,
        "val_mask": val_m,
        "test_mask": test_m,
        "subset": keep_global,
        "n_id": keep_global,
        "hop_dist": hops,
        "split": split_name,
    }


def extract_split_tf_subgraphs(data, split, num_hops=2, max_nodes=100, seed=42):
    if max_nodes < 1:
        raise ValueError(f"max_nodes must be >= 1, got {max_nodes}")
    split_mask = getattr(data, f"{split}_mask")
    subgraphs = {}
    if not bool(split_mask.any()):
        return subgraphs
    N = data.x.size(0)
    full_ei = data.edge_index
    device = full_ei.device
    tfs = _tfs_touched_by_mask(data, split_mask)
    n_multi = 0
    sizes = []
    # Prefer a device-local Generator; fall back to CPU if CUDA generators unavailable.
    try:
        rng = torch.Generator(device=device)
    except (RuntimeError, TypeError):
        rng = torch.Generator(device="cpu")
    for tf_idx in tfs:
        layers = _bfs_hop_layers(tf_idx, full_ei, N, num_hops)
        hop_of = _hop_lookup(layers, N, device)
        rng.manual_seed(int(seed) + int(tf_idx) + 1_000_003 * _SPLIT_SEED.get(split, 0))
        groups = _hop_fill_groups(layers, max_nodes, rng)
        if len(groups) > 1:
            n_multi += 1
        sizes.append(int(sum(int(layer.numel()) for layer in layers)))
        for part_i, keep_global in enumerate(groups):
            hops = hop_of[keep_global.to(device)].clamp_min(0)
            rec = _record_from_global_nodes(data, keep_global, split, hops)
            rec["center_tf"] = int(tf_idx)
            subgraphs[(int(tf_idx), part_i)] = rec
    if tfs:
        filled = [int(sub["subset"].numel()) for sub in subgraphs.values()] or [0]
        print(
            f"[INFO] TF-subgraphs split={split}: device={device} max_hops={num_hops} "
            f"max_nodes={max_nodes} seed={seed}; n_tf={len(tfs)}; "
            f"n_subgraphs={len(subgraphs)}; multi_tf={n_multi}; "
            f"filled node range [{min(filled)}, {max(filled)}]; "
            f"raw hop-union range [{min(sizes)}, {max(sizes)}]; "
            f"mp=full_coexp_among_subset score_split={split}",
            flush=True,
        )
    return subgraphs


def extract_tf_subgraphs(data, num_hops=2, max_nodes=100, seed=42):
    """TF-centered subgraphs on the full coexpression graph; labels stay TF-split."""
    return {
        "train": extract_split_tf_subgraphs(
            data, "train", num_hops=num_hops, max_nodes=max_nodes, seed=seed
        ),
        "val": extract_split_tf_subgraphs(
            data, "val", num_hops=num_hops, max_nodes=max_nodes, seed=seed
        ),
        "test": extract_split_tf_subgraphs(
            data, "test", num_hops=num_hops, max_nodes=max_nodes, seed=seed
        ),
    }

def freeze_split_from_data(data, ref_corr_threshold: float, seed: int) -> dict:
    """Serialize train/val/test assignment as gene-name pair keys."""
    genes = list(data.gene_names)
    src, dst = data.edge_index.cpu().numpy()
    pair_to_split = {}
    tf_to_split = {}
    tf_set = {int(i) for i in data.tf_indices}
    for split in ("train", "val", "test"):
        mask = getattr(data, f"{split}_mask").cpu().numpy()
        for e in np.nonzero(mask)[0]:
            a, b = genes[int(src[e])], genes[int(dst[e])]
            pair_to_split["|".join(sorted([a, b]))] = split
            if int(src[e]) in tf_set:
                tf_to_split.setdefault(genes[int(src[e])], split)
            if int(dst[e]) in tf_set:
                tf_to_split.setdefault(genes[int(dst[e])], split)
    return {
        "ref_corr_threshold": float(ref_corr_threshold),
        "seed": int(seed),
        "pair_to_split": pair_to_split,
        "tf_to_split": tf_to_split,
    }

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
        # CHANGED [A11-filename]: was "go_term.csv"; matches main.py / README (gene_go_pairs.csv).
        # ORIGINAL: "go_term.csv",
        "gene_go_pairs.csv",
        "tf_binding_logodds.csv"
    )

def load_data(
    expr_path: str,
    tf_path: str,
    priors_path: str,
    tf_bind_path: str,
    label_path: str | None = None,
    tf_bind2_path: str | None = None,
    corr_threshold: float = 0.1,
    anno_csv_path: str | None = None,
    llama_model: str | None = None,
    llama_batch_size: int = 16,
    llama_max_tokens: int = 256,
    test_pair_path: str | None = None,
    # CHANGED [A-seed]: expose seed for multi-seed ablations.
    # ORIGINAL: (no parameter; seed = 42 hard-coded below)
    seed: int = 42,
    # CHANGED [A7-tf-indicator]: gate tf_indicator for ablation of TF-identity shortcut.
    # ORIGINAL: (always included; no parameter)
    use_tf_indicator: bool = True,
    freeze_split: dict | None = None,
    # Feature ablations (zero the block; keep dims so the architecture matches full).
    ablate_go: bool = False,
    ablate_text: bool = False,
    ablate_tf_bind: bool = False,
    ablate_motif: bool = False,
):
    import os
    import random
    from collections import defaultdict

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from sklearn.preprocessing import MultiLabelBinarizer
    from transformers import AutoTokenizer, AutoModel
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    # CHANGED [A-seed]: was hard-coded to 42; now uses the seed parameter.
    # ORIGINAL: seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    def _build_llama_text_embeddings(
        anno_csv_path: str,
        genes: list[str],
        # CHANGED [A5-log-encoder]: default matches what actually runs (MiniLM), not LLaMA.
        # ORIGINAL: model_name: str = "meta-llama/Llama-3.2-1B",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 16,
        max_tokens: int = 256,
        cache_key: str | None = None,
    ) -> np.ndarray:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cache_dir = os.path.join(os.path.dirname(anno_csv_path), ".cache_llama_embeds")
        os.makedirs(cache_dir, exist_ok=True)
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
            # CHANGED [A5-log-encoder]: log which encoder actually loads.
            # ORIGINAL: (no logging)
            print(f"[TEXT-EMB] Using encoder: {model_name_try}")

            outs = []
            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    enc = tok(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=max_tokens,
                        return_tensors="pt"
                    ).to(device)
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

        # CHANGED [A5-log-encoder]: MiniLM is the explicit default above; fallbacks are others.
        # ORIGINAL: fallbacks started with all-MiniLM-L6-v2 (first fallback after gated LLaMA).
        fallbacks = [
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

    def _log1p_signed(x: np.ndarray) -> np.ndarray:
        pos = x >= 0
        out = np.empty_like(x, dtype=np.float32)
        out[pos] = np.log1p(x[pos])
        out[~pos] = -np.log1p(np.abs(x[~pos]))
        return out

    # ─────────────────────────────────────────────────────────────
    # Load base data
    # ─────────────────────────────────────────────────────────────
    expr_df = pd.read_csv(expr_path, index_col=0)
    genes = expr_df.index.astype(str).tolist()
    expr_np = expr_df.values.astype(np.float32, copy=False)

    gene_to_idx = {g: i for i, g in enumerate(genes)}
    N = len(genes)

    tf_df = pd.read_csv(tf_path)
    tf_list = tf_df.iloc[:, 0].astype(str).tolist()
    tf_indices = [gene_to_idx[g] for g in tf_list if g in gene_to_idx]
    tf_set = set(tf_indices)

    # GO priors
    priors_df = pd.read_csv(priors_path)
    priors_df = priors_df[priors_df["gene"].isin(genes)]

    gene2go = defaultdict(set)
    for _, r in priors_df.iterrows():
        if pd.notna(r.get("go_term")):
            gene2go[str(r["gene"])].add(str(r["go_term"]))

    go_mat = MultiLabelBinarizer(sparse_output=False).fit_transform(
        [sorted(gene2go.get(g, set())) for g in genes]
    ).astype(np.float32, copy=False)

    # Pearson correlation
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

    corr_mat = np.asarray(corr_mat, dtype=np.float32)

    # TF binding 1
    bind_df = pd.read_csv(tf_bind_path)
    bind_df = bind_df[bind_df["motif"].isin(genes) & bind_df["gene"].isin(genes)]
    bind_dict = {
        (gene_to_idx[r.motif], gene_to_idx[r.gene]): float(r.best_score)
        for r in bind_df.itertuples()
    }

    # TF binding 2
    bind2_dict = None
    if tf_bind2_path is not None:
        b2 = pd.read_csv(tf_bind2_path, sep=None, engine="python")
        cols = set(b2.columns)

        def _idx(x):
            if pd.isna(x):
                return None
            s = str(x)
            return gene_to_idx.get(s, gene_to_idx.get(s.split(".", 1)[0]))

        def _gene_from_fimo_seq(x):
            """FIMO/BED sequence_name often looks like GENE(+) or GENE(-)."""
            if pd.isna(x):
                return None
            g = str(x).split("(")[0].strip().split("::")[0]
            return gene_to_idx.get(g, gene_to_idx.get(g.split(".", 1)[0]))

        if {"motif_alt_id", "sequence_name"}.issubset(cols):
            if "score" in cols:
                b2 = b2[["motif_alt_id", "sequence_name", "score"]].rename(
                    columns={"motif_alt_id": "motif", "sequence_name": "gene", "score": "best_score"}
                )
            elif "p-value" in cols:
                b2 = b2[["motif_alt_id", "sequence_name", "p-value"]].rename(
                    columns={"motif_alt_id": "motif", "sequence_name": "gene"}
                )
                b2["best_score"] = -np.log10(b2["p-value"].astype(float).clip(lower=1e-300))
            else:
                raise ValueError("tf_bind2_path: need 'score' or 'p-value' in FIMO file")
            map_gene = _gene_from_fimo_seq
        elif {"gene", "tf_name", "score_max"}.issubset(cols):
            b2 = b2.rename(columns={"tf_name": "motif", "score_max": "best_score"})
            map_gene = _idx
        else:
            raise ValueError("tf_bind2_path: unrecognized columns")

        b2["s"] = b2["motif"].map(_idx)
        b2["d"] = b2["gene"].map(map_gene)
        b2 = b2.dropna(subset=["s", "d", "best_score"]).groupby(
            ["s", "d"], as_index=False
        )["best_score"].max()

        bind2_dict = {
            (int(r.s), int(r.d)): float(r.best_score)
            for r in b2.itertuples(index=False)
        }

    # ─────────────────────────────────────────────────────────────
    # Inference-only mode: build graph ONLY from test_pair_path
    # ─────────────────────────────────────────────────────────────
    if test_pair_path is not None:
        test_df = pd.read_csv(test_pair_path)
        if not {"Gene1", "Gene2"}.issubset(test_df.columns):
            raise ValueError("test_pair_path must contain columns: Gene1, Gene2")

        pair_rows = []
        missing_pairs = 0

        for r in test_df.itertuples(index=False):
            g1 = str(r.Gene1).strip()
            g2 = str(r.Gene2).strip()

            if g1 not in gene_to_idx or g2 not in gene_to_idx:
                missing_pairs += 1
                continue

            s = gene_to_idx[g1]
            d = gene_to_idx[g2]

            # keep both directions internally
            pair_rows.append((s, d))
            pair_rows.append((d, s))

        if len(pair_rows) == 0:
            raise ValueError("No valid pairs from test_pair_path matched genes in expr_path")

        src = np.array([s for s, d in pair_rows], dtype=np.int64)
        dst = np.array([d for s, d in pair_rows], dtype=np.int64)

        corr_vals = np.array([corr_mat[s, d] for s, d in pair_rows], dtype=np.float32)
        bind_vals = np.array([bind_dict.get((s, d), 0.0) for s, d in pair_rows], dtype=np.float32)
        if ablate_tf_bind:
            bind_vals = np.zeros_like(bind_vals)
        edge_cols = [corr_vals, bind_vals]
        if bind2_dict is not None:
            bind2_vals = np.array([bind2_dict.get((s, d), 0.0) for s, d in pair_rows], dtype=np.float32)
            if ablate_motif:
                bind2_vals = np.zeros_like(bind2_vals)
            edge_cols.append(bind2_vals)

        edge_attr = torch.tensor(np.stack(edge_cols, axis=1), dtype=torch.float32)
        y_tensor = torch.zeros(len(src), dtype=torch.float32)

        degrees = np.zeros(N, dtype=np.float32)
        for s, d in zip(src, dst):
            degrees[s] += 1
            degrees[d] += 1

        # CHANGED [A7-tf-indicator]: conditional on use_tf_indicator
        # ORIGINAL: always included tf_indicator
        go_use = go_mat.astype(np.float32, copy=False)
        if ablate_go:
            go_use = np.zeros_like(go_use)

        feats = [
            expr_np,
            go_use,
            degrees[:, None],
        ]
        if use_tf_indicator:
            tf_indicator = np.zeros((N, 1), dtype=np.float32)
            if tf_indices:
                tf_indicator[np.array(tf_indices, dtype=int)] = 1.0
            feats.append(tf_indicator)

        if anno_csv_path is not None:
            # CHANGED [A5-log-encoder]: default MiniLM
            # ORIGINAL: model_name = llama_model or "meta-llama/Llama-3.2-1B"
            model_name = llama_model or "sentence-transformers/all-MiniLM-L6-v2"
            text_emb = _build_llama_text_embeddings(
                anno_csv_path=anno_csv_path,
                genes=genes,
                model_name=model_name,
                batch_size=llama_batch_size,
                max_tokens=llama_max_tokens,
                cache_key=os.path.basename(expr_path),
            )
            if ablate_text:
                text_emb = np.zeros_like(text_emb)
            feats.append(text_emb.astype(np.float32, copy=False))

        X = np.hstack(feats)
        x_tensor = torch.tensor(X, dtype=torch.float32)

        E = len(src)
        train_mask = torch.zeros(E, dtype=torch.bool)
        val_mask = torch.zeros(E, dtype=torch.bool)
        test_mask = torch.ones(E, dtype=torch.bool)

        data = Data(
            x=x_tensor,
            edge_index=torch.tensor(np.vstack([src, dst]), dtype=torch.long),
            edge_attr=edge_attr,
            y=y_tensor,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )
        data.gene_names = genes
        data.tf_indices = tf_indices
        data.expr_dim = expr_np.shape[1]

        print("[INFO] Inference mode enabled from test_pair_path")
        print(f"[INFO] Requested pairs: {len(test_df)}")
        print(f"[INFO] Missing pairs skipped: {missing_pairs}")
        print(f"[INFO] Directed test edges built: {E}")
        print(f"[INFO] Undirected valid pairs kept: {E // 2}")

        return data

    # ─────────────────────────────────────────────────────────────
    # Normal training/eval mode
    # ─────────────────────────────────────────────────────────────
    iu, ju = np.triu_indices(N, k=1)
    # CHANGED [A1-abs-corr]: was signed >= corr_threshold, which dropped all repressive
    # (negatively correlated) edges. Now uses magnitude; signed corr kept as edge feature.
    # ORIGINAL: mask = corr_mat[iu, ju] >= corr_threshold
    mask = np.abs(corr_mat[iu, ju]) >= corr_threshold

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

    # Labels
    if label_path is None:
        print("[INFO] label_path is None → using all-zero labels")
        pos_set = set()
    else:
        labels_df = pd.read_csv(label_path)
        if not {"Gene1", "Gene2"}.issubset(labels_df.columns):
            raise ValueError("label_path must contain columns: Gene1, Gene2")

        pos_set = {
            (gene_to_idx[r.Gene1], gene_to_idx[r.Gene2])
            for r in labels_df.itertuples()
            if (r.Gene1 in gene_to_idx)
            and (r.Gene2 in gene_to_idx)
            and (gene_to_idx[r.Gene1] in tf_set)
        }

    E0 = len(src)
    y_np = np.fromiter(
        ((s, d) in pos_set for s, d in zip(src, dst)),
        dtype=np.float32,
        count=E0
    )

    # CHANGED [A1-abs-corr]: invariant uses abs so repressive edges do not false-alarm.
    # ORIGINAL: n_bad = int((corr_vals < corr_threshold).sum())
    #           raise ... "corr < corr_threshold"
    n_bad = int((np.abs(corr_vals) < corr_threshold).sum())
    if n_bad != 0:
        raise RuntimeError(f"Found {n_bad} edges with |corr| < corr_threshold after filtering (bug).")

    tf_to_edges = defaultdict(list)
    for e, (s, d) in enumerate(zip(src, dst)):
        if s in tf_set:
            tf_to_edges[s].append(e)
        if d in tf_set and d != s:
            tf_to_edges[d].append(e)

    freeze_pairs = (freeze_split or {}).get("pair_to_split")
    freeze_tf = (freeze_split or {}).get("tf_to_split")

    keep_mask = np.zeros(E0, dtype=bool)
    owned_split = np.empty(E0, dtype=object)

    if freeze_pairs is not None:
        print(f"[INFO] Frozen split from corr={freeze_split.get('ref_corr_threshold')} "
              f"({len(freeze_pairs)} undirected pairs); new edges → train, never test")
        name_of = genes
        for e, (s, d) in enumerate(zip(src, dst)):
            key = "|".join(sorted([name_of[s], name_of[d]]))
            if key in freeze_pairs:
                owned_split[e] = freeze_pairs[key]
                keep_mask[e] = True
                continue
            if (s in tf_set) and (d in tf_set) and freeze_tf is not None:
                ss = freeze_tf.get(name_of[s])
                ds = freeze_tf.get(name_of[d])
                if ss is not None and ds is not None and ss != ds:
                    owned_split[e] = None
                    keep_mask[e] = False
                    continue
            owned_split[e] = "train"
            keep_mask[e] = True
    else:
        total_edges = E0
        total_pos = int(y_np.sum())
        targets = {
            "train": {"edges": int(round(0.70 * total_edges)), "pos": int(round(0.70 * total_pos))},
            "val":   {"edges": int(round(0.15 * total_edges)), "pos": int(round(0.15 * total_pos))},
            "test":  {"edges": total_edges,                   "pos": total_pos},
        }
        targets["test"]["edges"] -= (targets["train"]["edges"] + targets["val"]["edges"])
        targets["test"]["pos"] -= (targets["train"]["pos"] + targets["val"]["pos"])

        splits = {"train": set(), "val": set(), "test": set()}
        stats = {
            "train": {"edges": 0, "pos": 0},
            "val": {"edges": 0, "pos": 0},
            "test": {"edges": 0, "pos": 0}
        }

        tf_order = sorted(
            tf_to_edges,
            key=lambda tf: (-len(tf_to_edges[tf]), -int(sum(y_np[e] for e in tf_to_edges[tf])))
        )

        for tf in tf_order:
            n_e = len(tf_to_edges[tf])
            n_p = int(sum(y_np[e] for e in tf_to_edges[tf]))
            best_split, best_score = None, float("inf")

            for split in ("train", "val", "test"):
                frac_e = stats[split]["edges"] / (targets[split]["edges"] or 1)
                frac_p = stats[split]["pos"] / (targets[split]["pos"] or 1)
                score = frac_e + frac_p
                if score < best_score:
                    best_score, best_split = score, split

            splits[best_split].add(tf)
            stats[best_split]["edges"] += n_e
            stats[best_split]["pos"] += n_p

        tf_to_split = {tf: split for split in splits for tf in splits[split]}

        undirected_gene_gene = defaultdict(list)
        for e, (s, d) in enumerate(zip(src, dst)):
            if (s not in tf_set) and (d not in tf_set):
                undirected_gene_gene[frozenset((s, d))].append(e)

        all_gg = list(undirected_gene_gene.keys())
        rng = np.random.default_rng(seed)
        rng.shuffle(all_gg)

        n_train = int(0.7 * len(all_gg))
        n_val = int(0.15 * len(all_gg))
        gg_split = {}

        for i, key in enumerate(all_gg):
            bucket = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
            for e in undirected_gene_gene[key]:
                gg_split[e] = bucket

        for e, (s, d) in enumerate(zip(src, dst)):
            if (s in tf_set) and (d in tf_set):
                if tf_to_split.get(s) != tf_to_split.get(d):
                    owned_split[e] = None
                    keep_mask[e] = False
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

    # CHANGED [A1-abs-corr]: removed lowcorr_pos drop of test positives with signed
    # corr < threshold. Under abs filter, negatively correlated positives are valid.
    # ORIGINAL:
    #   lowcorr_pos = (y_np == 1) & (corr_vals < corr_threshold)
    #   is_test = (owned_split == "test")
    #   drop_test_lowcorr_pos = lowcorr_pos & is_test
    #   if drop_test_lowcorr_pos.any():
    #       print(...)
    #       keep_mask[drop_test_lowcorr_pos] = False
    #       owned_split[drop_test_lowcorr_pos] = None

    src = src[keep_mask]
    dst = dst[keep_mask]
    corr_vals = corr_vals[keep_mask]
    bind_vals = bind_vals[keep_mask]
    y_np = y_np[keep_mask]
    owned_split = owned_split[keep_mask]
    if bind2_dict is not None:
        bind2_vals = bind2_vals[keep_mask]

    E = len(src)

    train_mask = torch.tensor([s == "train" for s in owned_split], dtype=torch.bool)
    val_mask = torch.tensor([s == "val" for s in owned_split], dtype=torch.bool)
    test_mask = torch.tensor([s == "test" for s in owned_split], dtype=torch.bool)

    train_mask_bool = (owned_split == "train")
    val_mask_bool = (owned_split == "val")
    test_mask_bool = (owned_split == "test")

    # CHANGED [A1-abs-corr]: use abs in check.
    # ORIGINAL: bad_test = int(((y_np == 1) & (corr_vals < corr_threshold) & test_mask_bool).sum())
    #           print("[CHECK] low-corr positives in TEST:", bad_test)
    bad_test = int(((y_np == 1) & (np.abs(corr_vals) < corr_threshold) & test_mask_bool).sum())
    print("[CHECK] low-|corr| positives in TEST:", bad_test)

    # CHANGED [A11-wire-edge-norm]: compute normalised edge features and use them as
    # primary edge_attr (raw previously fed to model; EF_norm was stored unused).
    # ORIGINAL: edge_attr built from raw [corr_vals, bind_vals, ...]; EF_norm unused.
    bind_use = bind_vals.astype(np.float32, copy=True)
    if ablate_tf_bind:
        bind_use[:] = 0.0
    edge_cols_norm = [corr_vals.astype(np.float32, copy=False), bind_use]
    if bind2_dict is not None:
        bind2_use = bind2_vals.astype(np.float32, copy=True)
        if ablate_motif:
            bind2_use[:] = 0.0
        edge_cols_norm.append(bind2_use)

    EF = np.stack(edge_cols_norm, axis=1).astype(np.float32, copy=False)
    EF[:, 0] = np.arctanh(np.clip(EF[:, 0], -0.999999, 0.999999))
    EF[:, 1] = _log1p_signed(EF[:, 1])
    if EF.shape[1] >= 3:
        EF[:, 2] = _log1p_signed(EF[:, 2])

    eps = 1e-6
    if train_mask_bool.any():
        mu = EF[train_mask_bool].mean(axis=0)
        sd = EF[train_mask_bool].std(axis=0)
        sd = np.where(sd < eps, 1.0, sd)
    else:
        mu = np.zeros(EF.shape[1], dtype=np.float32)
        sd = np.ones(EF.shape[1], dtype=np.float32)

    EF_norm = (EF - mu) / sd
    EF_norm = np.clip(EF_norm, -5.0, 5.0)

    edge_norm_stats = {
        "mu": mu.astype(np.float32),
        "sd": sd.astype(np.float32),
        "transform": "corr->atanh; bind->signed log1p; zscore(train); clip±5"
    }

    # CHANGED [A8-degree-leakage]: degree from train_mask edges only (not all splits).
    # ORIGINAL: for s, d in zip(src, dst): degrees[s/d] += 1
    train_src = src[train_mask_bool]
    train_dst = dst[train_mask_bool]
    degrees = np.zeros(N, dtype=np.float32)
    for s, d in zip(train_src, train_dst):
        degrees[s] += 1
        degrees[d] += 1

    # CHANGED [A7-tf-indicator]: gate behind use_tf_indicator
    # ORIGINAL: always appended tf_indicator
    go_use = go_mat.astype(np.float32, copy=False)
    if ablate_go:
        go_use = np.zeros_like(go_use)
    feats = [expr_np, go_use, degrees[:, None]]
    if use_tf_indicator:
        tf_indicator = np.zeros((N, 1), dtype=np.float32)
        if tf_indices:
            tf_indicator[np.array(tf_indices, dtype=int)] = 1.0
        feats.append(tf_indicator)

    if anno_csv_path is not None:
        # CHANGED [A5-log-encoder]: MiniLM default
        # ORIGINAL: model_name = llama_model or "meta-llama/Llama-3.2-1B"
        model_name = llama_model or "sentence-transformers/all-MiniLM-L6-v2"
        text_emb = _build_llama_text_embeddings(
            anno_csv_path=anno_csv_path,
            genes=genes,
            model_name=model_name,
            batch_size=llama_batch_size,
            max_tokens=llama_max_tokens,
            cache_key=os.path.basename(expr_path),
        )
        if ablate_text:
            text_emb = np.zeros_like(text_emb)
        feats.append(text_emb.astype(np.float32, copy=False))

    X = np.hstack(feats)
    x_tensor = torch.tensor(X, dtype=torch.float32)
    if ablate_go or ablate_text or ablate_tf_bind or ablate_motif:
        print(
            f"[ABLATION] go={ablate_go} text={ablate_text} "
            f"tf_bind={ablate_tf_bind} motif={ablate_motif}"
        )

    # CHANGED [A11-wire-edge-norm]: edge_attr = normalised; raw kept as edge_attr_raw
    # ORIGINAL: edge_attr from raw stack; data.edge_attr_norm = EF_norm (unused by model)
    edge_attr = torch.tensor(EF_norm, dtype=torch.float32)
    edge_attr_raw = torch.tensor(
        np.stack([corr_vals, bind_vals] + ([bind2_vals] if bind2_dict is not None else []), axis=1),
        dtype=torch.float32,
    )

    y_tensor = torch.tensor(y_np, dtype=torch.float32)

    def _nodes_in(mask_t: torch.Tensor):
        idx = np.nonzero(mask_t.numpy())[0]
        nodes = set()
        for e in idx:
            nodes.add(int(src[e]))
            nodes.add(int(dst[e]))
        return nodes

    train_nodes = _nodes_in(train_mask)
    val_nodes = _nodes_in(val_mask)
    test_nodes = _nodes_in(test_mask)

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
        x=x_tensor,
        edge_index=torch.tensor(np.vstack([src, dst]), dtype=torch.long),
        edge_attr=edge_attr,
        y=y_tensor,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    data.gene_names = genes
    data.tf_indices = tf_indices
    data.expr_dim = expr_np.shape[1]
    data.edge_norm_stats = edge_norm_stats
    # CHANGED [A11-wire-edge-norm]: raw kept for inspection; model reads normalised edge_attr
    # ORIGINAL: data.edge_attr_norm = torch.tensor(EF_norm, dtype=torch.float32)
    data.edge_attr_raw = edge_attr_raw

    return data

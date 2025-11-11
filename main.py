from data import *
from model import *
from train import *
from eval import *
from data import _sanitize_graph, extract_tf_subgraphs
import os
import random
import argparse
import numpy as np
import torch
from torch_geometric.utils import k_hop_subgraph
import re

def seed_everything(seed: int = 42):
    """
    Fully deterministic setup for reproducible experiments.
    Call this ONCE at the very start of your program, before any CUDA operations.
    """
    import os, random, numpy as np, torch

    # --- Environment variables ---
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    # --- Core seeds ---
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- Deterministic behavior across threads and kernels ---
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

# --- in main() ---
seed_everything(42)

seed = 42

def parse_args():
    p = argparse.ArgumentParser(
        description="Run GraphTransformer: load → pretrain → finetune → evaluate"
    )
    # REQUIRED
    p.add_argument("--expr_path",  required=True, help="Path to <BASE>_ExpressionData.csv")
    p.add_argument("--label_path", required=True, help="Path to <BASE>_refNetwork.csv")
    p.add_argument("--tf_path",    required=True, help="Path to <BASE>_TFs.csv")

    # OPTIONAL
    p.add_argument("--priors_path",   default="gene_go_pairs.csv",
                   help="Path to gene_go_pairs.csv (default: gene_go_pairs.csv)")
    p.add_argument("--tf_bind_path",  default="tf_binding_logodds.csv",
                   help="Path to tf_binding_logodds.csv (default: tf_binding_logodds.csv)")
    p.add_argument("--corr_threshold", type=float, default=0.1)
    p.add_argument("--max_epochs",     type=int,   default=1000)
    p.add_argument("--patience",       type=int,   default=5)
    p.add_argument("--hops_tf",        type=int,   default=2)
    p.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--hidden_dim",     type=int,   default=128)
    p.add_argument("--heads",          type=int,   default=8)
    p.add_argument("--num_layers",     type=int,   default=1)
    p.add_argument("--dropout",        type=float, default=0.1)
    p.add_argument("--base_name",      default=None,
                   help="Base name for checkpoints/outputs; default: derived from expr file stem")
    return p.parse_args()

def infer_base_name(expr_path: str, label_path: str, tf_path: str, explicit: str | None = None) -> str:
    if explicit:
        return explicit

    def stem(p: str) -> str:
        return os.path.splitext(os.path.basename(p))[0]

    def strip_known_expr(s: str) -> str:
        for suf in ["_ExpressionData", "_Expression", "_expr", "_EXP", "_exp"]:
            if s.endswith(suf):
                return s[: -len(suf)]
        return s

    def strip_known_label(s: str) -> str:
        return re.sub(r"_refNetwork$", "", s)

    def strip_known_tf(s: str) -> str:
        return re.sub(r"_TFs?$", "", s)

    s1 = strip_known_expr(stem(expr_path))
    s2 = strip_known_label(stem(label_path))
    s3 = strip_known_tf(stem(tf_path))

    # Prefer a token-wise common prefix; fall back to expr-derived name
    t1, t2, t3 = s1.split("_"), s2.split("_"), s3.split("_")
    common = []
    for a, b, c in zip(t1, t2, t3):
        if a == b == c:
            common.append(a)
        else:
            break
    return "_".join(common) if common else s1

def main():
    args = parse_args()
    seed_everything(42)

    device = torch.device(args.device)
    print(f"Using device: {device}\n")

    # derive clean base name
    base_name = infer_base_name(args.expr_path, args.label_path, args.tf_path, args.base_name)
    print(f"[INFO] Base name: {base_name}")

    # Build model (GraphTransformer loads data internally)
    model = GraphTransformer(
        args.expr_path,
        args.label_path,
        args.tf_path,
        args.priors_path,
        args.tf_bind_path,
        args.corr_threshold,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    print(f"[CONFIG] hidden_dim={model.node_lin[1].in_features//2} "
          f"heads={getattr(model.convs[0], 'heads', 'N/A')} "
          f"num_layers={len(model.convs)} "
          f"dropout={model.dropout}")

    # sanitize and prep data
    g_main = _sanitize_graph(model.data)
    ensure_pos_emb_capacity(model, g_main.x.size(0))

    # build TF-centric subgraphs/pools
    tf_subgraphs = extract_tf_subgraphs(g_main, num_hops=args.hops_tf)
    train_tf_pools = [tf_subgraphs]

    # ─────────────── Stage A: Pretraining ───────────────
    if hasattr(model, "classifier"):
        for p in model.classifier.parameters(): p.requires_grad = False
    if hasattr(model, "bilin"):
        for p in model.bilin.parameters(): p.requires_grad = False
    if hasattr(model, "cls_mix"):
        model.cls_mix.requires_grad = False
    if hasattr(model, "freeze_encoder"):
        model.freeze_encoder(False)

    print("\n=== Stage A: Pretraining on masked gene expression ===")
    _ = train_expr_only(
        model, train_tf_pools, device,
        max_epochs=50, patience=args.patience,
        MASK_RATIO_TRAIN=0.1, MASK_RATIO_VAL=0.1,
        w_mse=1.0, w_pcc=1.0, base_lr=model.lr, wd=1e-4
    )

    # ─────────────── Stage B: Fine-tuning edges ───────────────
    if hasattr(model, "classifier"):
        for p in model.classifier.parameters(): p.requires_grad = True
    if hasattr(model, "bilin"):
        for p in model.bilin.parameters(): p.requires_grad = True
    if hasattr(model, "cls_mix"):
        model.cls_mix.requires_grad = True
    if hasattr(model, "reinit_edge_head"):
        model.reinit_edge_head()
    if hasattr(model, "freeze_encoder"):
        model.freeze_encoder(False)

    print("\n=== Stage B: Fine-tuning on edge prediction ===")
    _ = finetune_edge_only(
        model, train_tf_pools, device,
        max_epochs=args.max_epochs, patience=args.patience,
        base_lr_encoder=2e-4,
        base_lr_head=5e-4,
        wd=1e-4
    )

    # ─────────────── Evaluate ───────────────
    model.eval()
    model.data = g_main.to(device)
    bootstrap_out = bootstrap_test(model, g_main, device, base_name=base_name)
    bootstrap_evaluation_best_thresh(f"{base_name}_prediction.csv", 100)


if __name__ == "__main__":
    main()

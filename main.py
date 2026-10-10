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

# CHANGED [A-seed]: removed module-level seed_everything(42) / seed=42.
# ORIGINAL: seed_everything(42); seed = 42

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
    # CHANGED [species-cli]: expose bind2 / text / TF-indicator (model.py already accepts these)
    # ORIGINAL: no --tf_bind2_path / --anno_csv_path / --no_tf_indicator
    p.add_argument("--tf_bind2_path", default=None,
                   help="Optional second motif score table (motif_score.tsv / FIMO TSV)")
    p.add_argument("--anno_csv_path", default=None,
                   help="Optional gene text descriptions CSV (text_description.csv)")
    p.add_argument("--no_tf_indicator", action="store_true",
                   help="Ablate TF-identity node feature shortcut (use_tf_indicator=False)")
    p.add_argument("--ablate_go", action="store_true",
                   help="Zero GO multi-hot node features")
    p.add_argument("--ablate_text", action="store_true",
                   help="Zero MiniLM text embedding node features")
    p.add_argument("--ablate_tf_bind", action="store_true",
                   help="Zero TF-binding-probability edge features")
    p.add_argument("--ablate_motif", action="store_true",
                   help="Zero motif-score edge features")
    p.add_argument("--expr_only", action="store_true",
                   help="Keep expression (+degree/TF-id); zero GO, text, TF-bind, motif")
    p.add_argument("--corr_threshold", type=float, default=0.1)
    p.add_argument("--max_epochs",     type=int,   default=1000)
    p.add_argument("--patience",       type=int,   default=5)
    p.add_argument("--hops_tf",        type=int,   default=2,
                   help="Max hops when hop-filling a TF-centered subgraph to --max_nodes_tf")
    p.add_argument("--max_nodes_tf",   type=int,   default=100,
                   help="Hop-fill each split-isolated TF subgraph until this many nodes; "
                        "an overflowing hop is packed without replacement into extra subgraphs")
    p.add_argument("--mask_ratio",     type=float, default=0.1,
                   help="Stage A expression mask ratio (train and val)")
    p.add_argument("--w_mse", type=float, default=1.0, help="Stage A MSE loss weight")
    p.add_argument("--w_pcc", type=float, default=1.0, help="Stage A Pearson loss weight")
    p.add_argument("--w_mae", type=float, default=1.0, help="Stage A MAE loss weight")
    p.add_argument("--w_bce", type=float, default=1.0,
                   help="Stage B weight on binary cross-entropy (logits)")
    p.add_argument("--w_ap", type=float, default=0.0,
                   help="Stage B weight on the AP surrogate (top-of-list; 0 disables)")
    p.add_argument("--w_pairauc", type=float, default=1.0,
                   help="Stage B weight on the pairwise AUC surrogate (whole-ranking order)")
    p.add_argument("--pair_cap", type=int, default=65536,
                   help="Max pos/neg pairs sampled per subgraph by the pairwise AUC loss")
    p.add_argument("--neg_ratio", type=int, default=50,
                   help="Negatives per positive per subgraph in Stage B; <=0 uses all negatives")
    p.add_argument("--edge_select", default="auroc", choices=["auprc", "auroc", "mean"],
                   help="Validation metric used to pick the Stage B checkpoint")
    p.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--hidden_dim",     type=int,   default=128)
    p.add_argument("--heads",          type=int,   default=8)
    p.add_argument("--num_layers",     type=int,   default=1)
    p.add_argument("--dropout",        type=float, default=0.1)
    p.add_argument("--base_name",      default=None,
                   help="Base name for checkpoints/outputs; default: derived from expr file stem")
    # CHANGED [B2-repro]: --seed for multi-seed ablations
    # ORIGINAL: no --seed argument
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    # CHANGED [reviewer-ablations]
    p.add_argument("--skip_pretrain", action="store_true",
                   help="Skip Stage A; fine-tune from random encoder init "
                        "(or from --init_ckpt if provided)")
    p.add_argument("--init_ckpt", default=None,
                   help="Optional checkpoint to load before Stage A/B "
                        "(model_state_dict). Use with --skip_pretrain to "
                        "warm-start Stage B from a baseline run.")
    p.add_argument("--linear_probe", action="store_true",
                   help="Run Stage A, then freeze encoder and train edge head only")
    p.add_argument("--conv_type", default="transformer",
                   choices=["transformer", "gcn", "gat"],
                   help="Node encoder convolution (architecture ablation)")
    p.add_argument("--freeze_split_path", default=None,
                   help="JSON of pair/TF splits from a reference corr graph (fair corr sweep)")
    p.add_argument("--input_noise_std", type=float, default=0.1,
                   help="Gaussian std added to node features x during training (0 disables)")
    # CHANGED [few-shot]: subsample train positives for Stage B only
    p.add_argument("--label_frac", type=float, default=1.0,
                   help="Fraction of unique train positive edges kept for Stage B "
                        "(1.0 = all). Ignored if --n_shot_pos is set.")
    p.add_argument("--n_shot_pos", type=int, default=None,
                   help="Keep exactly this many unique train positives for Stage B "
                        "(overrides --label_frac). Negatives unchanged.")
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
    # CHANGED [A-seed]
    # ORIGINAL: seed_everything(42)
    seed_everything(args.seed)

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
        dropout=args.dropout,
        seed=args.seed,
        # CHANGED [species-cli]: pass rebuilt priors / text / TF-indicator ablation
        tf_bind2_path=args.tf_bind2_path,
        anno_csv_path=args.anno_csv_path,
        use_tf_indicator=not args.no_tf_indicator,
        conv_type=args.conv_type,
        freeze_split_path=args.freeze_split_path,
        input_noise_std=args.input_noise_std,
        ablate_go=bool(args.ablate_go or args.expr_only),
        ablate_text=bool(args.ablate_text or args.expr_only),
        ablate_tf_bind=bool(args.ablate_tf_bind or args.expr_only),
        ablate_motif=bool(args.ablate_motif or args.expr_only),
    ).to(device)

    if args.init_ckpt:
        ckpt = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[INFO] loaded init_ckpt={args.init_ckpt} "
              f"(missing={len(missing)} unexpected={len(unexpected)})", flush=True)

    print(f"[CONFIG] hidden_dim={args.hidden_dim} conv_type={args.conv_type} "
          f"heads={getattr(model.convs[0], 'heads', 'N/A')} "
          f"num_layers={len(model.convs)} "
          f"dropout={model.dropout} "
          f"corr_threshold={args.corr_threshold} hops_tf={args.hops_tf} "
          f"max_nodes_tf={args.max_nodes_tf} "
          f"mask_ratio={args.mask_ratio} w_mse={args.w_mse} w_pcc={args.w_pcc} w_mae={args.w_mae} "
          f"skip_pretrain={args.skip_pretrain} linear_probe={args.linear_probe} "
          f"tf_indicator={not args.no_tf_indicator} freeze_split={args.freeze_split_path} "
          f"input_noise_std={args.input_noise_std} "
          f"label_frac={args.label_frac} n_shot_pos={args.n_shot_pos} "
          f"w_bce={args.w_bce} w_ap={args.w_ap} w_pairauc={args.w_pairauc} pair_cap={args.pair_cap} "
          f"neg_ratio={args.neg_ratio} edge_select={args.edge_select}")

    # sanitize and prep data; move graph to training device so TF-subgraph
    # BFS + edge induction run on GPU when --device cuda.
    g_main = _sanitize_graph(model.data)
    ensure_pos_emb_capacity(model, g_main.x.size(0))
    g_main = g_main.to(device)
    model.data = g_main
    print(
        f"[INFO] subgraph extract on {device}: "
        f"E={int(g_main.edge_index.size(1))} N={int(g_main.x.size(0))}",
        flush=True,
    )

    # Split-isolated TF-centered subgraphs: hop-fill to max_nodes inside each split
    splits = extract_tf_subgraphs(
        g_main, num_hops=args.hops_tf, max_nodes=args.max_nodes_tf, seed=args.seed
    )
    train_subs, val_subs, test_subs = splits["train"], splits["val"], splits["test"]

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
    if args.skip_pretrain:
        if args.init_ckpt:
            print(f"[INFO] --skip_pretrain: skipping Stage A (warm-start from {args.init_ckpt})")
        else:
            print("[INFO] --skip_pretrain: skipping Stage A (random encoder init)")
    else:
        _ = train_expr_only(
            model, train_subs, device,
            max_epochs=50, patience=args.patience,
            MASK_RATIO_TRAIN=args.mask_ratio, MASK_RATIO_VAL=args.mask_ratio,
            w_mse=args.w_mse, w_pcc=args.w_pcc, w_mae=args.w_mae,
            base_lr=model.lr, wd=1e-4,
            val_subs=val_subs,
        )

    # Few-shot: after Stage A, restrict train positives seen by Stage B
    few_shot_info = apply_few_shot_train_positives(
        train_subs,
        seed=args.seed,
        label_frac=args.label_frac,
        n_shot_pos=args.n_shot_pos,
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
        model.freeze_encoder(bool(args.linear_probe))

    print("\n=== Stage B: Fine-tuning on edge prediction ===")
    if args.linear_probe:
        print("[INFO] --linear_probe: encoder frozen; training edge head only")
    _ = finetune_edge_only(
        model, train_subs, device,
        max_epochs=args.max_epochs, patience=args.patience,
        base_lr_encoder=2e-4,
        base_lr_head=5e-4,
        wd=1e-4,
        freeze_encoder_ft=bool(args.linear_probe),
        val_subs=val_subs,
        w_bce=args.w_bce, w_ap=args.w_ap, w_pairauc=args.w_pairauc, pair_cap=args.pair_cap,
        neg_ratio=args.neg_ratio, edge_select=args.edge_select,
    )

    # CHANGED [A10-checkpoint]: save after Stage B (README expected this file).
    # ORIGINAL: no checkpoint save
    ckpt_path = f"{base_name}_pretrained_weight.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "base_name": base_name,
            "hidden_dim": args.hidden_dim,
            "heads": args.heads,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "corr_threshold": args.corr_threshold,
            "hops_tf": args.hops_tf,
            "max_nodes_tf": args.max_nodes_tf,
            "mask_ratio": args.mask_ratio,
            "w_mse": args.w_mse,
            "w_pcc": args.w_pcc,
            "w_mae": args.w_mae,
            "freeze_split_path": args.freeze_split_path,
            "seed": args.seed,
            "conv_type": args.conv_type,
            "skip_pretrain": args.skip_pretrain,
            "linear_probe": args.linear_probe,
            "use_tf_indicator": not args.no_tf_indicator,
            "input_noise_std": args.input_noise_std,
            "label_frac": args.label_frac,
            "n_shot_pos": args.n_shot_pos,
            "few_shot": few_shot_info,
            "w_bce": args.w_bce,
            "w_ap": args.w_ap,
            "w_pairauc": args.w_pairauc,
            "pair_cap": args.pair_cap,
            "neg_ratio": args.neg_ratio,
            "edge_select": args.edge_select,
            "priors_path": args.priors_path,
            "anno_csv_path": args.anno_csv_path,
        },
        ckpt_path,
    )
    print(f"[INFO] Checkpoint saved → {ckpt_path}")

    # ─────────────── Evaluate ───────────────
    model.eval()
    model.data = g_main.to(device)
    bootstrap_out = bootstrap_test(
        model, g_main, device, base_name=base_name,
        test_subs=test_subs, mask_ratio=args.mask_ratio,
    )

    # F1 threshold from val subgraphs: encode and score inside the val split
    val_probs, val_labels = collect_split_edge_arrays(
        model, val_subs, device, tf_only=True,
        tf_indices=g_main.tf_indices, gene_names=list(g_main.gene_names),
    )
    if val_probs.size == 0:
        val_probs = val_labels = None

    bootstrap_evaluation_best_thresh(
        f"{base_name}_prediction.csv",
        n_bootstraps=100,
        val_probs=val_probs,
        val_labels=val_labels,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

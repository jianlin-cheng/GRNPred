import torch
from torchmetrics.functional import (
    auroc,
    average_precision,
    accuracy,
    precision,
    recall,
    f1_score as tm_f1_score,
    jaccard_index,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    jaccard_score,
    precision_recall_curve
)
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data


from train import _sub_graph, _sub_split_mask, use_data

# Standard EPR@K cutoffs used in GRN benchmark comparisons (plus EPR@gt = #test positives).
DEFAULT_EPR_K = (100, 500, 1000, 2000)


def epr_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
    """Early precision: fraction of positives among the top-k scored edges."""
    n = len(y_true)
    if n == 0:
        return float("nan")
    k = min(int(k), n)
    if k <= 0:
        return float("nan")
    order = np.argsort(-y_prob)
    return float(np.sum(y_true[order[:k]] == 1) / k)


def epr_metric_specs(n_pos: int, n_total: int) -> list[tuple[int, str]]:
    """Return (k, label) pairs for EPR reporting on a scored edge set."""
    specs: list[tuple[int, str]] = []
    seen: set[int] = set()
    for k in DEFAULT_EPR_K:
        if k <= n_total and k not in seen:
            specs.append((k, f"EPR@{k}"))
            seen.add(k)
    if n_pos > 0 and n_pos <= n_total and n_pos not in seen:
        specs.append((n_pos, "EPR@gt"))
    return sorted(specs, key=lambda x: x[0])


def _tf_edge_mask(edge_index, tf_indices):
    tf_set = set(int(i) for i in tf_indices)
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    return torch.tensor(
        [(int(s) in tf_set or int(d) in tf_set) for s, d in zip(src, dst)],
        dtype=torch.bool,
    )


def _mp_data(sub, device):
    """Encode on every coexpression edge in the subset (structure ≠ ChIP split)."""
    return _sub_graph(sub, sub["edge_index"], sub["edge_attr"], sub["y"], device)


def collect_subgraph_edge_rows(model, subgraphs, gene_names, tf_indices, device, include_label=True):
    """Encode on the full coexp subgraph; score only this split's label edges."""
    tf_set = set(int(i) for i in tf_indices)
    src_all, dst_all, prob_all, y_all, tf_all = [], [], [], [], []
    old = model.data
    model.eval()
    try:
        with torch.no_grad():
            for _, sub in subgraphs.items():
                mask = _sub_split_mask(sub)
                if mask.sum().item() == 0:
                    continue
                g = _mp_data(sub, device)
                n_id = sub["subset"].detach().cpu()
                score_ei = sub["edge_index"][:, mask].to(device)
                score_ea = sub["edge_attr"][mask].to(device)
                y_split = sub["y"][mask]
                with use_data(model, g):
                    logits = model.forward(
                        mask_ratio=0.0,
                        task="edge",
                        score_edge_index=score_ei,
                        score_edge_attr=score_ea,
                    )
                logits = torch.nan_to_num(logits.float(), nan=0.0).detach().cpu()
                probs = torch.sigmoid(logits)
                ei = score_ei.detach().cpu()
                y = y_split.detach().cpu() if include_label else None
                center = int(sub.get("center_tf", -1))
                src_g = n_id[ei[0]].numpy()
                dst_g = n_id[ei[1]].numpy()
                src_all.extend(int(v) for v in src_g)
                dst_all.extend(int(v) for v in dst_g)
                prob_all.extend(float(v) for v in probs.numpy())
                tf_all.extend([center] * ei.size(1))
                if include_label:
                    y_all.extend(int(v) for v in y.numpy())
    finally:
        model.data = old
    src_np = np.asarray(src_all, dtype=np.int64)
    dst_np = np.asarray(dst_all, dtype=np.int64)
    probs_np = np.asarray(prob_all, dtype=np.float64)
    center_np = np.asarray(tf_all, dtype=np.int64)
    y_np = np.asarray(y_all, dtype=np.int64) if include_label else None
    return src_np, dst_np, probs_np, y_np, center_np, tf_set, gene_names


def _dedup_tf_role(src_np, dst_np, probs_np, y_np, tf_set, gene_names, include_label, extra_center=None):
    seen_pairs = {}
    keep_indices = []
    for i, (s, d) in enumerate(zip(src_np, dst_np)):
        pair = frozenset((int(s), int(d)))
        if pair in seen_pairs:
            j = seen_pairs[pair]
            sj = src_np[j]
            if int(s) in tf_set and int(sj) not in tf_set:
                keep_indices[keep_indices.index(j)] = i
                seen_pairs[pair] = i
            else:
                probs_np[j] = 0.5 * (probs_np[j] + probs_np[i])
        else:
            seen_pairs[pair] = i
            keep_indices.append(i)

    keep_indices = np.array(keep_indices, dtype=np.int64)
    if keep_indices.size == 0:
        cols = {"Gene1": [], "Gene2": [], "PredProb": [], "CenterTF": []}
        if include_label:
            cols["TrueLabel"] = []
        return pd.DataFrame(cols)

    final_src = src_np[keep_indices]
    final_dst = dst_np[keep_indices]
    final_prob = probs_np[keep_indices]
    if extra_center is not None:
        final_tf = extra_center[keep_indices]
    else:
        final_tf = np.full(len(keep_indices), -1, dtype=np.int64)

    if include_label:
        src_to_pos = set(zip(src_np[y_np == 1], dst_np[y_np == 1]))
        final_y = []
        for i_k, (s, d) in zip(keep_indices, zip(final_src, final_dst)):
            label = int(y_np[i_k])
            if (int(d), int(s)) in src_to_pos:
                label = 1
            final_y.append(label)
        final_y = np.array(final_y, dtype=np.int64)

    tf_involved = np.array([(int(s) in tf_set or int(d) in tf_set)
                            for s, d in zip(final_src, final_dst)])

    center_names = []
    for gid, keep in zip(final_tf, tf_involved):
        if not keep:
            continue
        if 0 <= int(gid) < len(gene_names):
            center_names.append(gene_names[int(gid)])
        else:
            center_names.append("")

    out = {
        "Gene1": [gene_names[i] for i, m in zip(final_src, tf_involved) if m],
        "Gene2": [gene_names[j] for j, m in zip(final_dst, tf_involved) if m],
        "PredProb": final_prob[tf_involved],
        "CenterTF": center_names,
    }
    if include_label:
        out["TrueLabel"] = final_y[tf_involved]
    return pd.DataFrame(out)


def collect_subgraph_expression(model, subgraphs, gene_names, device, mask_ratio=0.1):
    """Predict expression for masked genes inside each split subgraph (Stage-A protocol).

    Only masked genes are reported: unmasked genes keep their expression in the
    encoder input, so their predictions are not held out.
    """
    if mask_ratio <= 0.0:
        print(
            "[EXPR] mask_ratio <= 0: no genes are held out, so no expression "
            "predictions are reported."
        )
        return pd.DataFrame(
            columns=["Gene", "CenterTF", "TrueMean", "PredMean", "MSE", "MAE", "Pearson"]
        ), {"n_values": 0, "mse": float("nan"), "mae": float("nan"), "pearson": float("nan")}

    rows = []
    mse_num, mae_num, n_vals = 0.0, 0.0, 0
    pred_cat, true_cat = [], []
    old = model.data
    model.eval()
    try:
        with torch.no_grad():
            for _, sub in subgraphs.items():
                g = _mp_data(sub, device)
                n_id = sub["subset"].detach().cpu()
                center = int(sub.get("center_tf", -1))
                center_name = gene_names[center] if 0 <= center < len(gene_names) else ""
                with use_data(model, g):
                    expr_pred, mask_nodes = model.forward(
                        mask_ratio=mask_ratio, task="gene_expression"
                    )
                if not mask_nodes.any():
                    continue
                true_expr = torch.nan_to_num(g.x[:, :model.expr_dim].float(), nan=0.0)
                expr_pred = torch.nan_to_num(expr_pred.float(), nan=0.0)

                d = (expr_pred[mask_nodes] - true_expr[mask_nodes]).reshape(-1)
                mse_num += float((d * d).sum().item())
                mae_num += float(d.abs().sum().item())
                n_vals += int(d.numel())
                pred_cat.append(expr_pred[mask_nodes].reshape(-1).detach().cpu())
                true_cat.append(true_expr[mask_nodes].reshape(-1).detach().cpu())

                masked_local = mask_nodes.nonzero(as_tuple=False).view(-1).detach().cpu()
                pred_np = expr_pred.detach().cpu().numpy()
                true_np = true_expr.detach().cpu().numpy()
                for i in masked_local.tolist():
                    p = pred_np[i]
                    t = true_np[i]
                    p0 = p - p.mean()
                    t0 = t - t.mean()
                    den = np.sqrt((p0 * p0).sum() * (t0 * t0).sum())
                    pcc = float((p0 * t0).sum() / (den + 1e-8)) if den > 0 else float("nan")
                    rows.append({
                        "Gene": gene_names[int(n_id[i].item())],
                        "CenterTF": center_name,
                        "TrueMean": float(t.mean()),
                        "PredMean": float(p.mean()),
                        "MSE": float(((p - t) ** 2).mean()),
                        "MAE": float(np.abs(p - t).mean()),
                        "Pearson": pcc,
                    })
    finally:
        model.data = old

    metrics = {"n_values": n_vals, "mse": float("nan"), "mae": float("nan"), "pearson": float("nan")}
    if n_vals > 0:
        metrics["mse"] = mse_num / n_vals
        metrics["mae"] = mae_num / n_vals
        pred_v = torch.cat(pred_cat)
        true_v = torch.cat(true_cat)
        pred_v = pred_v - pred_v.mean()
        true_v = true_v - true_v.mean()
        den = torch.sqrt((pred_v * pred_v).sum() * (true_v * true_v).sum())
        metrics["pearson"] = float((pred_v * true_v).sum() / (den + 1e-8))
    return pd.DataFrame(rows), metrics


def collect_split_edge_arrays(model, subgraphs, device, tf_only=True, tf_indices=None, gene_names=None):
    """Val-threshold helper: PredProb and TrueLabel from encode-in-split subgraphs."""
    names = gene_names if gene_names is not None else ["g"]
    src_np, dst_np, probs_np, y_np, center_np, tf_set, names = collect_subgraph_edge_rows(
        model, subgraphs, names, tf_indices or [], device, include_label=True
    )
    if src_np.size == 0 or y_np is None:
        return np.array([]), np.array([])
    df = _dedup_tf_role(
        src_np, dst_np, probs_np, y_np, tf_set, names, True, extra_center=center_np
    )
    if df.empty:
        return np.array([]), np.array([])
    if not tf_only:
        return df["PredProb"].to_numpy(), df["TrueLabel"].to_numpy()
    return df["PredProb"].to_numpy(), df["TrueLabel"].to_numpy()


def _write_expression_outputs(model, subgraphs, full_data, device, base_name, mask_ratio):
    df_expr, expr_metrics = collect_subgraph_expression(
        model, subgraphs, list(full_data.gene_names), device, mask_ratio=mask_ratio
    )
    expr_path = f"{base_name}_expression_prediction.csv"
    df_expr.to_csv(expr_path, index=False)
    print(f"Saved {len(df_expr)} masked-gene expression predictions → {expr_path}")
    met_path = f"{base_name}_expression_metrics.csv"
    pd.DataFrame([
        {"Metric": "MSE (masked)", "Value": expr_metrics["mse"]},
        {"Metric": "MAE (masked)", "Value": expr_metrics["mae"]},
        {"Metric": "Pearson (masked)", "Value": expr_metrics["pearson"]},
        {"Metric": "n_values", "Value": expr_metrics["n_values"]},
        {"Metric": "n_masked_genes", "Value": len(df_expr)},
        {"Metric": "mask_ratio", "Value": mask_ratio},
    ]).to_csv(met_path, index=False)
    print(
        f"[EXPR] test masked reconstruction: "
        f"MSE={expr_metrics['mse']:.4f} MAE={expr_metrics['mae']:.4f} "
        f"PCC={expr_metrics['pearson']:.4f} n={expr_metrics['n_values']} "
        f"genes={len(df_expr)} (mask_ratio={mask_ratio}) → {met_path}"
    )
    return expr_path


def bootstrap_test(
    model,
    full_data,
    device,
    base_name: str = "prediction",
    test_subs=None,
    mask_ratio: float = 0.1,
):
    if test_subs is None:
        raise ValueError("bootstrap_test requires test_subs (split-isolated TF subgraphs)")

    model.eval()
    gene_names = list(full_data.gene_names)
    src_np, dst_np, probs_np, y_np, center_np, tf_set, gene_names = collect_subgraph_edge_rows(
        model, test_subs, gene_names, full_data.tf_indices, device, include_label=True
    )
    df_out = _dedup_tf_role(
        src_np, dst_np, probs_np, y_np, tf_set, gene_names, True, extra_center=center_np
    )
    filename = f"{base_name}_prediction.csv"
    df_out.to_csv(filename, index=False)
    print(f"Saved {len(df_out)} TF-involving test edges → {filename}")

    if not df_out.empty and "TrueLabel" in df_out.columns:
        pos_pairs = set(zip(df_out[df_out.TrueLabel == 1]["Gene1"],
                            df_out[df_out.TrueLabel == 1]["Gene2"]))
        neg_rows = df_out[df_out.TrueLabel == 0]
        reversed_in_neg = sum(1 for _, r in neg_rows.iterrows()
                              if (r["Gene2"], r["Gene1"]) in pos_pairs)
        if reversed_in_neg > 0:
            print(f"[WARNING] {reversed_in_neg} negative rows are reversals of positive edges.")
        else:
            print("[CHECK] Zero negative rows are reversals of positive edges. Direction labels OK.")

    _write_expression_outputs(model, test_subs, full_data, device, base_name, mask_ratio)
    return filename

def bootstrap_evaluation_best_thresh(
    csv_path: str,
    n_bootstraps: int = 100,
    val_probs: np.ndarray | None = None,
    val_labels: np.ndarray | None = None,
    seed: int = 42,
):
    df = pd.read_csv(csv_path)
    y_true = df["TrueLabel"].values
    y_prob = df["PredProb"].values

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_pos   = len(pos_idx)

    # CHANGED [A3-threshold]: select F1 threshold on validation, not per-test bootstrap.
    if val_probs is not None and val_labels is not None:
        prec_v, rec_v, thr_v = precision_recall_curve(val_labels, val_probs)
        f1_v = 2 * prec_v[:-1] * rec_v[:-1] / (prec_v[:-1] + rec_v[:-1] + 1e-12)
        fixed_thresh = float(thr_v[np.nanargmax(f1_v)])
        print(f"[EVAL] F1 threshold selected on validation set: {fixed_thresh:.4f}")
    else:
        fixed_thresh = 0.5
        print("[EVAL] No validation data provided; using fixed threshold = 0.5")

    boot_metrics = {
        "Accuracy": [], "Precision": [], "Recall": [], "F1 Score": [],
        "ROC AUC": [], "AUPRC (balanced)": [], "Jaccard Index": []
    }
    bs_epr_keys: list[str] = []

    # CHANGED [A11-bootstrap-seed]
    rng = np.random.default_rng(seed)

    for i in range(n_bootstraps):
        # CHANGED [A11-replace-false]: was replace=True
        # ORIGINAL: neg_sample = np.random.choice(neg_idx, size=n_pos, replace=True)
        n_sample = min(n_pos, len(neg_idx))
        neg_sample = rng.choice(neg_idx, size=n_sample, replace=False)
        idx         = np.concatenate([pos_idx, neg_sample])
        y_true_bs   = y_true[idx]
        y_prob_bs   = y_prob[idx]

        # CHANGED [A3-threshold]: fixed threshold, not oracle on this draw
        y_pred_bs = (y_prob_bs >= fixed_thresh).astype(int)

        boot_metrics["Accuracy"].append(accuracy_score(y_true_bs, y_pred_bs))
        boot_metrics["Precision"].append(precision_score(y_true_bs, y_pred_bs, zero_division=0))
        boot_metrics["Recall"].append(recall_score(y_true_bs, y_pred_bs, zero_division=0))
        boot_metrics["F1 Score"].append(f1_score(y_true_bs, y_pred_bs, zero_division=0))
        boot_metrics["Jaccard Index"].append(jaccard_score(y_true_bs, y_pred_bs, zero_division=0))
        boot_metrics["ROC AUC"].append(roc_auc_score(y_true_bs, y_prob_bs))
        boot_metrics["AUPRC (balanced)"].append(average_precision_score(y_true_bs, y_prob_bs))

        if i == 0:
            for k, label in epr_metric_specs(int(y_true_bs.sum()), len(y_true_bs)):
                key = f"{label} (balanced)"
                bs_epr_keys.append(key)
                boot_metrics[key] = []
        for key in bs_epr_keys:
            label = key.replace(" (balanced)", "")
            k = n_pos if label == "EPR@gt" else int(label.split("@")[1])
            boot_metrics[key].append(epr_at_k(y_true_bs, y_prob_bs, k))

    print(f"Balanced-bootstrap evaluation ({n_bootstraps} rounds, threshold={fixed_thresh:.4f}):")
    rows = []
    metric_order = [
        "Accuracy", "Precision", "Recall",
        "F1 Score", "ROC AUC", "AUPRC (balanced)", "Jaccard Index",
        *bs_epr_keys,
    ]
    for name in metric_order:
        arr = np.array(boot_metrics[name])
        print(f"{name:22s} mean={arr.mean():.4f}  std={arr.std():.4f}")
        rows.append({"Metric": name, "Mean": round(float(arr.mean()), 4),
                     "Std": round(float(arr.std()), 4)})

    # CHANGED [A11-csv-output]
    out_csv = csv_path.replace("_prediction.csv", "_prediction_metrics.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[EVAL] Balanced-bootstrap metrics saved → {out_csv}")

    # CHANGED [A6-unbalanced-eval]
    print(f"\nFull-imbalanced evaluation (natural prevalence "
          f"{y_true.mean():.4f}, n={len(y_true)}):")
    imb_rows = []
    imb_specs = [
        ("ROC AUC (imbalanced)", roc_auc_score(y_true, y_prob)),
        ("AUPRC (imbalanced)", average_precision_score(y_true, y_prob)),
        ("F1 @ thr (imbalanced)", f1_score(
            y_true, (y_prob >= fixed_thresh).astype(int), zero_division=0)),
    ]
    for k, label in epr_metric_specs(n_pos, len(y_true)):
        imb_specs.append((f"{label} (imbalanced)", epr_at_k(y_true, y_prob, k)))
    for name, val in imb_specs:
        print(f"{name:30s} {val:.4f}")
        imb_rows.append({"Metric": name, "Value": round(float(val), 4)})

    imb_csv = csv_path.replace("_prediction.csv", "_prediction_metrics_imbalanced.csv")
    pd.DataFrame(imb_rows).to_csv(imb_csv, index=False)
    print(f"[EVAL] Imbalanced metrics saved → {imb_csv}")

    return {"balanced": {r["Metric"]: r["Mean"] for r in rows},
            "imbalanced": {r["Metric"]: r["Value"] for r in imb_rows},
            "threshold": fixed_thresh}


def bootstrap_test_no_label(
    model,
    full_data,
    device,
    base_name: str = "inference",
    test_subs=None,
    mask_ratio: float = 0.1,
):
    """Score test-split TF subgraphs and write edge + expression predictions."""
    if test_subs is None:
        raise ValueError("bootstrap_test_no_label requires test_subs")

    model.eval()
    gene_names = list(full_data.gene_names)
    src_np, dst_np, probs_np, _, center_np, tf_set, gene_names = collect_subgraph_edge_rows(
        model, test_subs, gene_names, full_data.tf_indices, device, include_label=False
    )
    df_out = _dedup_tf_role(
        src_np, dst_np, probs_np, None, tf_set, gene_names, False, extra_center=center_np
    )
    filename = f"{base_name}_prediction.csv"
    df_out.to_csv(filename, index=False)
    print(f"[INFERENCE] Saved {len(df_out)} TF-involving predictions → {filename}")
    _write_expression_outputs(model, test_subs, full_data, device, base_name, mask_ratio)
    return filename

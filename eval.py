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

def bootstrap_test(model, full_data, device, base_name: str = "prediction"):


    model.eval()

    # ─── 1) Build train-only graph ─────────────────────────────────────────────
    train_mask = full_data.train_mask
    train_data = Data(
        x          = full_data.x,
        edge_index = full_data.edge_index[:, train_mask],
        edge_attr  = full_data.edge_attr[train_mask]
    )
    model.data = train_data.to(device)

    # ─── 2) Message-passing to get node embeddings (mirror model._encode) ─────
    with torch.no_grad():
        x    = model.data.x.to(device)
        idx  = torch.arange(x.size(0), device=device)
        p    = model.pos_emb(idx)
        x    = torch.cat([x, p], dim=1)
        h    = model.node_lin(x)

        conv_ei = model.data.edge_index.to(device)
        conv_ea = model.data.edge_attr.to(device)

        for layer, (conv, norm) in enumerate(zip(model.convs, model.norms)):
            e_drop = F.dropout(conv_ea, p=model.dropout, training=True)

            h_in = h
            h    = conv(h, conv_ei, e_drop)
            h    = norm(h + h_in)
            h    = F.relu(h)

            row, col = conv_ei
            h_i, h_j = h[row], h[col]
            e_new    = model.edge_updater(torch.cat([h_i, h_j, e_drop], dim=-1))

            if hasattr(model, "edge_gates"):
                gate = torch.sigmoid(model.edge_gates[layer])
            else:
                gate = torch.sigmoid(model.edge_gate)

            conv_ea = e_drop + gate * e_new

    test_mask = full_data.test_mask
    ei_test   = full_data.edge_index[:, test_mask].to(device)
    src, dst  = ei_test
    h_i, h_j  = h[src], h[dst]


    had   = h_i * h_j
    diff1 = (h_i - h_j).abs()
    edge_repr = torch.cat([h_i, h_j, had, diff1], dim=-1)

    with torch.no_grad():
    
        mlp_score = model.classifier(edge_repr).squeeze(-1)

        if hasattr(model, "bilin") and (model.bilin is not None) and hasattr(model, "cls_mix"):
            bil_score = model.bilin(h_i, h_j).squeeze(-1)
            mix = torch.sigmoid(model.cls_mix)
            logits = mix * mlp_score + (1.0 - mix) * bil_score
        else:
            logits = mlp_score

        # cosine similarity term
        cos = F.cosine_similarity(h_i, h_j, dim=-1)
        if hasattr(model, "cos_scale"):
            bias  = getattr(model, "cos_bias", torch.tensor(0., device=device))
            logits = logits + model.cos_scale * (cos + bias)
        else:
            logits = logits + torch.tanh(model.cos_mix) * cos

        # optional class-prior bias
        if hasattr(model, "logit_bias"):
            logits = logits + model.logit_bias

        probs = torch.sigmoid(logits).detach().cpu()


    y_true   = full_data.y[test_mask].long().detach().cpu()
    gene_names = full_data.gene_names
    tf_set     = set(full_data.tf_indices)

    src_np   = src.cpu().numpy()
    dst_np   = dst.cpu().numpy()
    probs_np = probs.numpy()

    tf_edge_mask = np.array([(s in tf_set) or (d in tf_set) for s, d in zip(src_np, dst_np)])

    undir_mask = src_np < dst_np
    keep_mask = tf_edge_mask & undir_mask

    df_out = pd.DataFrame({
        "Gene1":     [gene_names[i] for i, m in zip(src_np, keep_mask) if m],
        "Gene2":     [gene_names[j] for j, m in zip(dst_np, keep_mask) if m],
        "TrueLabel": y_true.numpy()[keep_mask],
        "PredProb":  probs_np[keep_mask],
    })

    filename = f"{base_name}_prediction.csv"
    df_out.to_csv(filename, index=False)
    print(f"Saved {len(df_out)} TF-involving test edges → {filename}")

    return filename

def bootstrap_evaluation_best_thresh(csv_path: str, n_bootstraps: int = 100):
    df = pd.read_csv(csv_path)
    y_true = df["TrueLabel"].values
    y_prob = df["PredProb"].values

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_pos   = len(pos_idx)

    boot_metrics = {
        "Accuracy": [], "Precision": [], "Recall": [], "F1 Score": [],
        "ROC AUC": [], "Average Precision": [], "Jaccard Index": []
    }
    best_thresholds = []

    # ── Bootstrap loop
    for i in range(n_bootstraps):
        # 1) sample positives & negatives
        neg_sample = np.random.choice(neg_idx, size=n_pos, replace=True)
        idx         = np.concatenate([pos_idx, neg_sample])
        y_true_bs   = y_true[idx]
        y_prob_bs   = y_prob[idx]

        # 2) find best‐F1 threshold for this bootstrap
        prec, rec, thr = precision_recall_curve(y_true_bs, y_prob_bs)
        f1_scores      = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
        best_i         = np.nanargmax(f1_scores)
        thresh_bs      = thr[best_i]
        best_thresholds.append(thresh_bs)

        # 3) binarize at that threshold
        y_pred_bs = (y_prob_bs >= thresh_bs).astype(int)

        # 4) compute & store metrics
        boot_metrics["Accuracy"].append(accuracy_score(y_true_bs, y_pred_bs))
        boot_metrics["Precision"].append(precision_score(y_true_bs, y_pred_bs, zero_division=0))
        boot_metrics["Recall"].append(recall_score(y_true_bs, y_pred_bs, zero_division=0))
        boot_metrics["F1 Score"].append(f1_score(y_true_bs, y_pred_bs, zero_division=0))
        boot_metrics["Jaccard Index"].append(jaccard_score(y_true_bs, y_pred_bs, zero_division=0))
        boot_metrics["ROC AUC"].append(roc_auc_score(y_true_bs, y_prob_bs))
        boot_metrics["Average Precision"].append(average_precision_score(y_true_bs, y_prob_bs))

    avg_thresh = np.mean(best_thresholds)
    std_thresh = np.std(best_thresholds)


   
    print(f"Bootstrapped evaluation over {n_bootstraps} runs:")
    for name in [
        "Accuracy", "Precision", "Recall",
        "F1 Score", "ROC AUC", "Average Precision", "Jaccard Index"
    ]:
        arr = np.array(boot_metrics[name])
        print(f"{name:17s} mean={arr.mean():.4f}  std={arr.std():.4f}")




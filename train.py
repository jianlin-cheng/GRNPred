import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from contextlib import contextmanager
# CHANGED [A10-nullcontext]: missing import caused ImportError on CPU-only runs.
# ORIGINAL: (no import; nullcontext used without import)
from contextlib import nullcontext
from torch_geometric.data import Data
from data import *
from model import *
from losses import *
from torch import amp
import torch.nn.functional as F


def _sub_split_mask(sub):
    name = sub.get("split", "train")
    key = f"{name}_mask"
    if key in sub:
        return sub[key]
    return sub["train_mask"]


def apply_few_shot_train_positives(
    train_subs,
    seed: int,
    label_frac: float | None = None,
    n_shot_pos: int | None = None,
):
    """Keep only a subset of train-split positive edges for Stage B.

    Negatives in train_mask are unchanged (Stage B still samples via neg_ratio).
    Val/test masks are not touched. Unique directed (src, dst) pairs are sampled
    once globally so the same few-shot set is shared across TF subgraphs.
    """
    if n_shot_pos is None and (label_frac is None or float(label_frac) >= 1.0):
        return {
            "enabled": False,
            "n_train_pos_full": None,
            "n_train_pos_kept": None,
            "label_frac": label_frac,
            "n_shot_pos": n_shot_pos,
        }
    if n_shot_pos is not None and n_shot_pos < 0:
        raise ValueError(f"n_shot_pos must be >= 0, got {n_shot_pos}")
    if label_frac is not None and not (0.0 <= float(label_frac) <= 1.0):
        raise ValueError(f"label_frac must be in [0, 1], got {label_frac}")

    pairs = []
    for sub in train_subs.values():
        mask = sub["train_mask"]
        y = sub["y"]
        ei = sub["edge_index"]
        n_id = sub["n_id"]
        pos = torch.nonzero(mask & (y == 1), as_tuple=False).view(-1)
        for eidx in pos.tolist():
            s = int(n_id[int(ei[0, eidx])])
            d = int(n_id[int(ei[1, eidx])])
            pairs.append((s, d))
    # Stable unique then shuffle with experiment seed
    unique = list(dict.fromkeys(pairs))
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(unique))
    unique = [unique[i] for i in order]

    n_full = len(unique)
    if n_shot_pos is not None:
        k = min(int(n_shot_pos), n_full)
    else:
        frac = float(label_frac)
        k = int(round(n_full * frac)) if frac > 0 else 0
        k = min(max(k, 0), n_full)
        # Avoid accidental empty Stage B when frac>0 but round→0
        if frac > 0 and n_full > 0 and k == 0:
            k = 1
    keep = set(unique[:k])

    for sub in train_subs.values():
        mask = sub["train_mask"].clone()
        y = sub["y"]
        ei = sub["edge_index"]
        n_id = sub["n_id"]
        pos = torch.nonzero(mask & (y == 1), as_tuple=False).view(-1)
        for eidx in pos.tolist():
            s = int(n_id[int(ei[0, eidx])])
            d = int(n_id[int(ei[1, eidx])])
            if (s, d) not in keep:
                mask[eidx] = False
        sub["train_mask"] = mask

    info = {
        "enabled": True,
        "n_train_pos_full": n_full,
        "n_train_pos_kept": k,
        "label_frac": None if n_shot_pos is not None else float(label_frac),
        "n_shot_pos": int(n_shot_pos) if n_shot_pos is not None else None,
        "kept_frac": (float(k) / float(n_full)) if n_full else 0.0,
    }
    print(
        f"[INFO] few-shot Stage B: kept {k}/{n_full} train positives "
        f"(label_frac={info['label_frac']}, n_shot_pos={info['n_shot_pos']})",
        flush=True,
    )
    return info


@contextmanager
def use_data(model, data_obj):
    """Temporarily set model.data to data_obj and restore afterward (even on error)."""
    _old = model.data
    model.data = data_obj
    try:
        yield
    finally:
        model.data = _old


def _sub_graph(sub, ei, ea, y, device):
    n_id = sub["subset"]
    kwargs = dict(
        x=sub["x"],
        edge_index=ei,
        edge_attr=ea,
        y=y,
        n_id=n_id,
        subset=n_id,
    )
    if sub.get("hop_dist") is not None:
        kwargs["hop_dist"] = sub["hop_dist"]
    return Data(**kwargs).to(device)


def _mp_graph(sub, device):
    """Encode on every coexpression edge in the subset (structure ≠ ChIP split)."""
    return _sub_graph(sub, sub["edge_index"], sub["edge_attr"], sub["y"], device)


def _edge_rank_loss(logits, y_edge, w_bce=1.0, w_ap=0.0, w_pairauc=1.0, pair_cap=65536):
    """BCE + pairwise AUC, with optional AP.

    BCE calibrates scores. Pairwise AUC orders the full list (AUROC). AP only
    cares about the head and will bury the tail if used alone.
    """
    terms = []
    if w_bce > 0:
        terms.append(w_bce * F.binary_cross_entropy_with_logits(logits, y_edge))
    if w_pairauc > 0:
        pr_loss = pairwise_auc_loss(logits, y_edge, max_pairs=pair_cap)
        if pr_loss is not None:
            terms.append(w_pairauc * pr_loss)
    if w_ap > 0:
        ap_loss = fast_ap_loss(logits, y_edge)
        if ap_loss is not None:
            terms.append(w_ap * ap_loss)
    if not terms:
        return None
    stacked = terms[0]
    for t in terms[1:]:
        stacked = stacked + t
    return stacked


def train_expr_only(model, train_subs, device, max_epochs, patience,
                    MASK_RATIO_TRAIN=0.2, MASK_RATIO_VAL=0.1,
                    # CHANGED [A11-w_mse]: default 1.0 to match equal-weighting claim.
                    # ORIGINAL: w_mse=0.0
                    w_mse=1.0, w_pcc=1.0, w_mae=1.0, base_lr=5e-4, wd=1e-4,
                    val_subs=None):

    amp_ctx = (lambda: autocast()) if (hasattr(device, "type") and device.type == "cuda") else nullcontext

    optimizer = AdamW([
        {"params": filter(lambda p: p.requires_grad, model.parameters()), "lr": base_lr}
    ], weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6
    )
    scaler = GradScaler(enabled=(hasattr(device, "type") and device.type == "cuda"))
    mse_loss = torch.nn.MSELoss()
    mae_loss = torch.nn.L1Loss()

    best_val = float('inf')
    best_state = None
    bad_epochs = 0
    val_subs = {} if val_subs is None else val_subs

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss, count = 0.0, 0
        for _, sub in train_subs.items():
            edge_mask = _sub_split_mask(sub)
            if edge_mask.sum().item() == 0:
                continue

            sub_edge = _mp_graph(sub, device)

            optimizer.zero_grad(set_to_none=True)
            with amp_ctx():
                with use_data(model, sub_edge):
                    expr_pred, mask_nodes = model.forward(
                        mask_ratio=MASK_RATIO_TRAIN, task="gene_expression"
                    )

                    touched = torch.zeros(sub_edge.x.size(0), dtype=torch.bool, device=device)
                    r, c = sub_edge.edge_index
                    if r.numel() > 0:
                        touched[r] = True
                        touched[c] = True
                    node_mask = touched & mask_nodes

                    true_expr = torch.nan_to_num(sub_edge.x[:, :model.expr_dim].float(), nan=0.0)
                    expr_pred = torch.nan_to_num(expr_pred.float(),               nan=0.0)

                    if node_mask.any():
                        pred_m = expr_pred[node_mask]
                        true_m = true_expr[node_mask]
                    else:
                        ri = torch.randint(0, sub_edge.x.size(0), (1,), device=device)
                        pred_m = expr_pred[ri]; true_m = true_expr[ri]

                    mse = mse_loss(pred_m, true_m)
                    pcc = pearson_corr_loss(pred_m, true_m)
                    mae = mae_loss(pred_m, true_m)
                    loss = w_mse*mse + w_pcc*pcc + w_mae*mae

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += float(loss); count += 1

        avg_train = total_loss / max(count, 1)

        model.eval()
        val_loss, val_count = 0.0, 0
        with torch.no_grad():
            for _, sub in val_subs.items():
                edge_mask = _sub_split_mask(sub)
                if edge_mask.sum().item() == 0:
                    continue

                sub_edge = _mp_graph(sub, device)

                with amp_ctx():
                    with use_data(model, sub_edge):
                        expr_pred, mask_nodes = model.forward(
                            mask_ratio=MASK_RATIO_VAL, task="gene_expression"
                        )

                    touched = torch.zeros(sub_edge.x.size(0), dtype=torch.bool, device=device)
                    r, c = sub_edge.edge_index
                    if r.numel() > 0:
                        touched[r] = True
                        touched[c] = True
                    node_mask = touched & mask_nodes
                    if not node_mask.any():
                        continue

                    true_expr = torch.nan_to_num(sub_edge.x[:, :model.expr_dim].float(), nan=0.0)
                    expr_pred = torch.nan_to_num(expr_pred.float(),               nan=0.0)
                    pred_m, true_m = expr_pred[node_mask], true_expr[node_mask]

                    mse = mse_loss(pred_m, true_m)
                    pcc = pearson_corr_loss(pred_m, true_m)
                    mae = mae_loss(pred_m, true_m)
                    loss = w_mse*mse + w_pcc*pcc + w_mae*mae

                    val_loss += float(loss); val_count += 1

        avg_val = val_loss / max(val_count, 1) if val_count else avg_train
        print(f"[Pretrain-Expr] Epoch {epoch:03d} — train: {avg_train:.4f} — val: {avg_val:.4f}")
        scheduler.step(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("[Pretrain-Expr] Early stop.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val

def _val_edge_metrics(model, val_subs, device):
    """AUPRC and AUROC over every validation edge, deduplicated by global (src, dst).

    Scores all edges rather than a resampled 1:50 subset so the model-selection
    signal is deterministic across epochs. Returns None when validation has no
    usable edges or only one class.
    """
    if not val_subs:
        return None
    model.eval()
    keys, probs, labels = [], [], []
    with torch.no_grad():
        for _, sub in val_subs.items():
            edge_mask = _sub_split_mask(sub)
            if edge_mask.sum().item() == 0:
                continue
            ei_all = sub["edge_index"][:, edge_mask]
            ea_all = sub["edge_attr"][edge_mask]
            y_all = sub["y"][edge_mask]
            n_id = sub["subset"].to(ei_all.device)
            encode_graph = _mp_graph(sub, device)
            with use_data(model, encode_graph):
                logits = model.forward(
                    mask_ratio=0.0,
                    task="edge",
                    score_edge_index=ei_all.to(device),
                    score_edge_attr=ea_all.to(device),
                )
            p = torch.sigmoid(torch.nan_to_num(logits.float(), nan=0.0)).cpu().numpy()
            src = n_id[ei_all[0]].cpu().numpy().astype(np.int64)
            dst = n_id[ei_all[1]].cpu().numpy().astype(np.int64)
            keys.append(np.stack([src, dst], axis=1))
            probs.append(p)
            labels.append(y_all.cpu().numpy())

    if not probs:
        return None
    pair = np.concatenate(keys, axis=0)
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    stride = int(pair.max()) + 1 if pair.size else 1
    flat = pair[:, 0] * stride + pair[:, 1]
    # Keep the highest-scoring copy of each pair; pairs recur across TF subgraphs.
    order = np.lexsort((-probs, flat))
    flat, probs, labels = flat[order], probs[order], labels[order]
    first = np.ones(flat.shape[0], dtype=bool)
    first[1:] = flat[1:] != flat[:-1]
    probs, labels = probs[first], labels[first]
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return None
    return {
        "auprc": float(average_precision_score(labels, probs)),
        "auroc": float(roc_auc_score(labels, probs)),
    }


def finetune_edge_only(model, train_subs, device, max_epochs, patience,
                       base_lr_encoder=2e-4, base_lr_head=5e-4, wd=1e-4,
                       freeze_encoder_ft=False, val_subs=None,
                       w_bce=1.0, w_ap=0.0, w_pairauc=1.0, pair_cap=65536, neg_ratio=50,
                       edge_select="auroc"):

    FREEZE_EPOCHS = 0
    # CHANGED [A10-patience]: removed hardcoded override of CLI patience.
    # ORIGINAL: patience = 10
    # CHANGED [linear-probe]: freeze_encoder_ft keeps Stage-A encoder frozen in Stage B.

    enc_params = []
    for m in [model.pos_emb, model.node_lin, model.convs, model.norms, model.edge_updater]:
        enc_params += list(m.parameters())

    # CHANGED [decoder-ablation]: head params depend on edge_decoder
    # ORIGINAL: always classifier + bilin + cls_mix
    if getattr(model, "edge_decoder", "mlp") == "ip_mean_edgeattr":
        head_params = [model.ip_edge_alpha, model.ip_edge_bias]
    else:
        head_params = list(model.classifier.parameters())
        if hasattr(model, "bilin") and model.bilin is not None:
            head_params += list(model.bilin.parameters())
        if hasattr(model, "cls_mix") and model.cls_mix is not None:
            head_params += [model.cls_mix]
        if hasattr(model, "cos_mix") and model.cos_mix is not None:
            head_params += [model.cos_mix]

    enc_params = [p for p in enc_params if p.requires_grad]
    head_params = [p for p in head_params if p.requires_grad]
    param_groups = []
    if enc_params:
        param_groups.append({"params": enc_params, "lr": base_lr_encoder})
    if head_params:
        param_groups.append({"params": head_params, "lr": base_lr_head})
    optimizer = AdamW(param_groups, weight_decay=wd)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-6
    )
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu")
    val_subs = {} if val_subs is None else val_subs

    best_score = -float('inf')
    best_state = None
    bad_epochs = 0

    for epoch in range(1, max_epochs + 1):
        if hasattr(model, "freeze_encoder"):
            model.freeze_encoder(True if freeze_encoder_ft else (epoch <= FREEZE_EPOCHS))

        model.train()
        train_loss, train_count = 0.0, 0

        for _, sub in train_subs.items():
            edge_mask = _sub_split_mask(sub)
            if edge_mask.sum().item() == 0:
                continue
            y_all     = sub["y"][edge_mask]
            eidx_all  = torch.nonzero(edge_mask, as_tuple=False).view(-1)
            pos_loc   = torch.nonzero(y_all == 1, as_tuple=False).view(-1)
            neg_loc   = torch.nonzero(y_all == 0, as_tuple=False).view(-1)
            if pos_loc.numel() == 0 or neg_loc.numel() == 0:
                continue
            n_neg = (neg_loc.size(0) if neg_ratio <= 0
                     else min(neg_loc.size(0), pos_loc.size(0) * neg_ratio))
            perm = torch.randperm(neg_loc.size(0), device=neg_loc.device)
            neg_sample = neg_loc[perm[:n_neg]]
            batch_idx = torch.cat([pos_loc, neg_sample], dim=0)
            sel_global = eidx_all[batch_idx]

            encode_graph = _mp_graph(sub, device)
            score_ei = sub["edge_index"][:, sel_global].to(device)
            score_ea = sub["edge_attr"][sel_global].to(device)
            y_edge = sub["y"][sel_global].float().to(device)

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(device_type=device.type):
                with use_data(model, encode_graph):
                    logits = model.forward(
                        mask_ratio=0.0,
                        task="edge",
                        score_edge_index=score_ei,
                        score_edge_attr=score_ea,
                    )
                    logits = torch.nan_to_num(logits.float(), nan=0.0)
                    loss = _edge_rank_loss(
                        logits, y_edge,
                        w_bce=w_bce, w_ap=w_ap, w_pairauc=w_pairauc, pair_cap=pair_cap,
                    )
                    if loss is None:
                        continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += float(loss)
            train_count += 1

        avg_train = train_loss / max(train_count, 1)

        val_m = _val_edge_metrics(model, val_subs, device)
        if val_m is None:
            # No usable validation signal; fall back to train loss so the loop still ends.
            score = -avg_train
            print(f"[Finetune-Edge] Epoch {epoch:03d} — train: {avg_train:.4f} — val: n/a")
        else:
            if edge_select == "auroc":
                score = val_m["auroc"]
            elif edge_select == "mean":
                score = 0.5 * (val_m["auprc"] + val_m["auroc"])
            else:
                score = val_m["auprc"]
            print(
                f"[Finetune-Edge] Epoch {epoch:03d} — train: {avg_train:.4f} "
                f"— val AUPRC: {val_m['auprc']:.4f} — val AUROC: {val_m['auroc']:.4f}"
            )
        scheduler.step(score)

        if score > best_score:
            best_score = score
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("[Finetune-Edge] Early stop.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[Finetune-Edge] Restored best epoch by val {edge_select}: {best_score:.4f}")
    return best_score

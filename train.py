import torch
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from contextlib import contextmanager
from torch_geometric.data import Data
from data import *
from model import *
from losses import *
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import amp

@contextmanager
def use_data(model, data_obj):
    """Temporarily set model.data to data_obj and restore afterward (even on error)."""
    _old = model.data
    model.data = data_obj
    try:
        yield
    finally:
        model.data = _old

def train_expr_only(model, train_tf_pools, device, max_epochs, patience,
                    MASK_RATIO_TRAIN=0.2, MASK_RATIO_VAL=0.1,
                    w_mse=0.0, w_pcc=1.0, base_lr=5e-4, wd=1e-4):

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

    # warm-up masks (build/cache split-specific fixed node masks inside model)
    with use_data(model, model.data.to(device)):
        _ = model.forward(mask_ratio=0.1, task="gene_expression", split="test")
        _ = model.forward(mask_ratio=0.1, task="gene_expression", split="val")

    for epoch in range(1, max_epochs + 1):
        # --------------------------- TRAIN (use ALL train edges) ---------------------------
        model.train()
        total_loss, count = 0.0, 0
        for tf_subgraphs in train_tf_pools:
            for _, sub in tf_subgraphs.items():
                if sub["train_mask"].sum().item() == 0:
                    continue

                # === CHANGED: take ALL edges in the train split (no pos/neg sampling) ===
                train_ei   = sub["edge_index"][:, sub["train_mask"]]
                train_attr = sub["edge_attr"][sub["train_mask"]]
                train_y    = sub["y"][sub["train_mask"]]

                sub_edge = Data(
                    x=sub["x"],
                    edge_index=train_ei,
                    edge_attr=train_attr,
                    y=train_y
                ).to(device)

                optimizer.zero_grad(set_to_none=True)
                with amp_ctx():
                    with use_data(model, sub_edge):
                        expr_pred, mask_nodes = model.forward(
                            mask_ratio=MASK_RATIO_TRAIN, task="gene_expression"
                        )

                        # restrict loss to nodes touched by current edges AND masked
                        touched = torch.zeros(sub_edge.x.size(0), dtype=torch.bool, device=device)
                        r, c = sub_edge.edge_index
                        touched[r] = True; touched[c] = True
                        node_mask = touched & mask_nodes

                        true_expr = torch.nan_to_num(sub_edge.x[:, :model.expr_dim].float(), nan=0.0)
                        expr_pred = torch.nan_to_num(expr_pred.float(),               nan=0.0)

                        if node_mask.any():
                            pred_m = expr_pred[node_mask]
                            true_m = true_expr[node_mask]
                        else:
                            # fallback to at least one node to avoid empty loss
                            ri = torch.randint(0, sub_edge.x.size(0), (1,), device=device)
                            pred_m = expr_pred[ri]; true_m = true_expr[ri]

                        mse = mse_loss(pred_m, true_m)
                        pcc = pearson_corr_loss(pred_m, true_m)
                        mae = mae_loss(pred_m, true_m)
                        loss = w_mse*mse + w_pcc*pcc + 1.0*mae

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer); scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                total_loss += float(loss); count += 1

        avg_train = total_loss / max(count, 1)

        # ------------------------- VALIDATION  -------------------------
        model.eval()
        val_loss, val_count = 0.0, 0
        with torch.no_grad():
            for tf_subgraphs in train_tf_pools:
                for _, sub in tf_subgraphs.items():
                    if sub["val_mask"].sum().item() == 0:
                        continue

                    val_ei   = sub["edge_index"][:, sub["val_mask"]]
                    val_attr = sub["edge_attr"][sub["val_mask"]]
                    val_y    = sub["y"][sub["val_mask"]]

                    sub_edge = Data(
                        x=sub["x"],
                        edge_index=val_ei,
                        edge_attr=val_attr,
                        y=val_y
                    ).to(device)

                    with amp_ctx():
                        with use_data(model, sub_edge):
                            expr_pred, mask_nodes = model.forward(
                                mask_ratio=MASK_RATIO_VAL, task="gene_expression", split="val"
                            )

                        touched = torch.zeros(sub_edge.x.size(0), dtype=torch.bool, device=device)
                        r, c = sub_edge.edge_index
                        touched[r] = True; touched[c] = True
                        node_mask = touched & mask_nodes
                        if not node_mask.any():
                            continue

                        true_expr = torch.nan_to_num(sub_edge.x[:, :model.expr_dim].float(), nan=0.0)
                        expr_pred = torch.nan_to_num(expr_pred.float(),               nan=0.0)
                        pred_m, true_m = expr_pred[node_mask], true_expr[node_mask]

                        mse = mse_loss(pred_m, true_m)
                        pcc = pearson_corr_loss(pred_m, true_m)
                        mae = mae_loss(pred_m, true_m)
                        loss = w_mse*mse + w_pcc*pcc + 1.0*mae

                        val_loss += float(loss); val_count += 1

        avg_val = val_loss / max(val_count, 1) if val_count else float('inf')
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

def finetune_edge_only(model, train_tf_pools, device, max_epochs, patience,
                       base_lr_encoder=2e-4, base_lr_head=5e-4, wd=1e-4):

    FREEZE_EPOCHS = 0   
    patience = 10       

    enc_params = []
    for m in [model.pos_emb, model.node_lin, model.convs, model.norms, model.edge_updater]:
        enc_params += list(m.parameters())

    head_params = list(model.classifier.parameters())
    if hasattr(model, "bilin") and model.bilin is not None:
        head_params += list(model.bilin.parameters())
    if hasattr(model, "cls_mix") and model.cls_mix is not None:
        head_params += [model.cls_mix]

    optimizer = AdamW([
        {"params": enc_params, "lr": base_lr_encoder},
        {"params": head_params, "lr": base_lr_head},
    ], weight_decay=wd)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6
    )
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu")

    best_val = float('inf')
    best_state = None
    bad_epochs = 0

    for epoch in range(1, max_epochs + 1):
        if hasattr(model, "freeze_encoder"):
            model.freeze_encoder(epoch <= FREEZE_EPOCHS)

        model.train()
        train_loss, train_count = 0.0, 0

        for tf_subgraphs in train_tf_pools:
            for _, sub in tf_subgraphs.items():
                if sub["train_mask"].sum().item() == 0:
                    continue
                edge_mask = sub["train_mask"]
                y_all     = sub["y"][edge_mask]
                eidx_all  = torch.nonzero(edge_mask, as_tuple=False).view(-1)
                pos_loc   = torch.nonzero(y_all == 1, as_tuple=False).view(-1)
                neg_loc   = torch.nonzero(y_all == 0, as_tuple=False).view(-1)
                if pos_loc.numel() == 0 or neg_loc.numel() == 0:
                    continue
                n_neg = min(neg_loc.size(0), pos_loc.size(0) * 50)
                perm = torch.randperm(neg_loc.size(0), device=neg_loc.device)
                neg_sample = neg_loc[perm[:n_neg]]
                batch_idx = torch.cat([pos_loc, neg_sample], dim=0)
                sel_global = eidx_all[batch_idx]

                sub_edge = Data(
                    x=sub["x"], edge_index=sub["edge_index"][:, sel_global],
                    edge_attr=sub["edge_attr"][sel_global], y=sub["y"][sel_global],
                ).to(device)

                optimizer.zero_grad(set_to_none=True)
                with amp.autocast(device_type=device.type):
                    with use_data(model, sub_edge):
                        logits = model.forward(mask_ratio=0.0, task="edge")
                        logits = torch.nan_to_num(logits.float(), nan=0.0)
                        y_edge = sub_edge.y.float().to(logits.device)


                        pr_loss = pairwise_auc_loss(logits, y_edge, max_pairs=65536)
                        ap_loss = fast_ap_loss(logits, y_edge)

                        if pr_loss is None and ap_loss is None:
                            continue
                        elif pr_loss is None:
                            loss = ap_loss
                        elif ap_loss is None:
                            loss = pr_loss
                        else:
                            loss = 0.5 * ap_loss + 0.5 * pr_loss
              
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()

                train_loss += float(loss)
                train_count += 1

        avg_train = train_loss / max(train_count, 1)

        # ---------------- Validation  ----------------
        model.eval()
        val_loss, val_count = 0.0, 0
        with torch.no_grad():
            for tf_subgraphs in train_tf_pools:
                for _, sub in tf_subgraphs.items():
                    if sub["val_mask"].sum().item() == 0:
                        continue
                    val_ei   = sub["edge_index"][:, sub["val_mask"]]
                    val_attr = sub["edge_attr"][sub["val_mask"]]
                    val_y    = sub["y"][sub["val_mask"]]
                    sub_edge = Data(x=sub["x"], edge_index=val_ei, edge_attr=val_attr, y=val_y).to(device)

                    with amp.autocast(device_type=device.type):
                        with use_data(model, sub_edge):
                            logits = model.forward(mask_ratio=0.0, task="edge")
                            logits = torch.nan_to_num(logits.float(), nan=0.0)
                            y_val  = sub_edge.y.float().to(logits.device)

                            pr_val = pairwise_auc_loss(logits, y_val, max_pairs=65536)
                            ap_val = fast_ap_loss(logits, y_val)

                            if pr_val is None and ap_val is None:
                                loss = torch.tensor(0.0, device=logits.device)
                            elif pr_val is None:
                                loss = ap_val
                            elif ap_val is None:
                                loss = pr_val
                            else:
                                loss = 0.5 * ap_val + 0.5 * pr_val

                    val_loss += float(loss)
                    val_count += 1

        avg_val = val_loss / max(val_count, 1) if val_count else float('inf')
        print(f"[Finetune-Edge] Epoch {epoch:03d} — train: {avg_train:.4f} — val: {avg_val:.4f}")
        scheduler.step(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("[Finetune-Edge] Early stop.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val

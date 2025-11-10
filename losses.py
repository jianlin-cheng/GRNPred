import torch
import torch.nn.functional as F

def pairwise_auc_loss(logits: torch.Tensor, labels: torch.Tensor, max_pairs: int = 4096):

    logits = logits.view(-1)
    labels = labels.view(-1)

    pos_idx = torch.nonzero(labels == 1, as_tuple=False).view(-1)
    neg_idx = torch.nonzero(labels == 0, as_tuple=False).view(-1)
    if pos_idx.numel() == 0 or neg_idx.numel() == 0:
        return None  

    s_pos = logits[pos_idx]            
    s_neg = logits[neg_idx]            

    P, N = s_pos.numel(), s_neg.numel()
    num_pairs = int(min(P * N, max_pairs))

    p_sel = torch.randint(0, P, (num_pairs,), device=logits.device)
    n_sel = torch.randint(0, N, (num_pairs,), device=logits.device)

    diff = s_pos[p_sel] - s_neg[n_sel]   
    return F.softplus(-diff).mean()     

def pearson_corr_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:

    x = pred.float().reshape(-1)
    y = target.float().reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    num = (x * y).sum()
    den = torch.sqrt((x.pow(2).sum() + eps) * (y.pow(2).sum() + eps))
    rho = num / (den + eps)
    return 1.0 - rho

def fast_ap_loss(scores: torch.Tensor, labels: torch.Tensor, eps: float = 1e-8):
    if scores.ndim != 1 or labels.ndim != 1:
        raise ValueError(f"fast_ap_loss expects 1D tensors, got {scores.shape=} {labels.shape=}")
    if scores.numel() != labels.numel():
        raise ValueError(f"Length mismatch: {scores.numel()} vs {labels.numel()}")

    y = torch.nan_to_num(labels.float(), nan=0.0).clamp_(0.0, 1.0).to(scores.device)
    s = torch.nan_to_num(scores.float(),  nan=0.0)

    s_sorted, idx = torch.sort(s, descending=True)
    y_sorted = y[idx]

    pos = y_sorted.sum()
    N   = y_sorted.numel()
    if pos <= 0 or pos >= N:
        return None

    cum_pos = torch.cumsum(y_sorted, dim=0)
    ranks   = torch.arange(1, N + 1, device=y_sorted.device, dtype=y_sorted.dtype)
    prec    = cum_pos / (ranks + eps)
    ap = (prec * y_sorted).sum() / (pos + eps)
    return 1.0 - ap
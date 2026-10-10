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


# CHANGED [A4-ap-loss]: differentiable soft-rank AP surrogate.
# ORIGINAL used torch.sort permutation only (s_sorted unused) → exact zero gradient.
def fast_ap_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 1e-4,
    temperature: float = 0.1,
    max_n: int = 4096,
):
    """Differentiable soft-rank average-precision surrogate loss.

    Soft-rank AP is O(N²). Cap N (keep all positives, subsample negatives)
    so a 50:1 train batch (~24k) cannot allocate a ~2GB score matrix.
    """
    if scores.ndim != 1 or labels.ndim != 1:
        raise ValueError(
            f"fast_ap_loss expects 1-D tensors, got {scores.shape=} {labels.shape=}"
        )
    if scores.numel() != labels.numel():
        raise ValueError(f"Length mismatch: {scores.numel()} vs {labels.numel()}")

    y = torch.nan_to_num(labels.float(), nan=0.0).clamp_(0.0, 1.0).to(scores.device)
    s = torch.nan_to_num(scores.float(), nan=0.0)

    if s.numel() > max_n:
        pos_idx = torch.nonzero(y > 0.5, as_tuple=False).view(-1)
        neg_idx = torch.nonzero(y <= 0.5, as_tuple=False).view(-1)
        n_pos = pos_idx.numel()
        if n_pos == 0:
            return None
        if n_pos >= max_n:
            perm_p = torch.randperm(n_pos, device=s.device)[: max_n - 1]
            keep = pos_idx[perm_p]
        else:
            n_neg = min(neg_idx.numel(), max_n - n_pos)
            perm_n = torch.randperm(neg_idx.numel(), device=s.device)[:n_neg]
            keep = torch.cat([pos_idx, neg_idx[perm_n]], dim=0)
        s = s[keep]
        y = y[keep]

    pos = y.sum()
    N   = y.numel()
    if pos <= 0 or pos >= N:
        return None

    diff = (s.unsqueeze(0) - s.unsqueeze(1)) / temperature
    soft_ranks = 1.0 + torch.sigmoid(diff).sum(dim=1)

    rank_i = soft_ranks.unsqueeze(1)
    rank_j = soft_ranks.unsqueeze(0)
    pos_at_rank_i = (y.unsqueeze(0) * torch.sigmoid(
        (rank_i - rank_j) / temperature
    )).sum(dim=1)

    soft_prec = pos_at_rank_i / (soft_ranks + eps)
    ap = (soft_prec * y).sum() / (pos + eps)

    return 1.0 - ap

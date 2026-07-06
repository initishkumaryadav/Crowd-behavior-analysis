"""Loss combination and ADE/FDE metrics (survey Eq. 32)."""
import torch


def total_loss(losses: dict, cfg) -> torch.Tensor:
    return (
        losses["recon"]
        + losses["dest"]
        + cfg.kl_weight_goal * losses["kl_goal"]
        + cfg.kl_weight_alpha * losses["kl_alpha"]
    )


@torch.no_grad()
def ade_fde(pred, gt):
    """pred, gt: (N, pred_len, 2). Returns (ADE, FDE) averaged over N.

    ADE = mean over all timesteps of ||pred - gt||;  FDE = error at final step.
    """
    err = (pred - gt).norm(dim=-1)          # (N, pred_len)
    ade = err.mean(dim=1).mean().item()
    fde = err[:, -1].mean().item()
    return ade, fde


@torch.no_grad()
def best_of_n_ade_fde(samples, gt):
    """samples: (S, N, pred_len, 2) — S stochastic rollouts. gt: (N, pred_len, 2).

    For each trajectory pick the sample with the lowest ADE (standard best-of-20).
    Returns (minADE, minFDE) averaged over N.
    """
    err = (samples - gt.unsqueeze(0)).norm(dim=-1)     # (S, N, pred_len)
    ade = err.mean(dim=2)                               # (S, N)
    best = ade.argmin(dim=0)                            # (N,) best sample per traj
    idx = torch.arange(gt.size(0), device=gt.device)
    min_ade = ade[best, idx].mean().item()
    fde = err[:, :, -1]                                 # (S, N)
    min_fde = fde[best, idx].mean().item()
    return min_ade, min_fde

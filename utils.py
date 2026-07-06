"""Small shared helpers: seeding, device, neighbour masking."""
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def neighbour_mask(pos: torch.Tensor, thresh: float) -> torch.Tensor:
    """Boolean (N, N) mask of who is a neighbour of whom at one timestep.

    A pedestrian j is a neighbour of i if 0 < ||p_i - p_j|| < thresh
    (paper: neighbour set Omega_i defined by a distance threshold). The
    diagonal (self) is always False.

    Args:
        pos: (N, 2) positions of N agents at a single timestep.
        thresh: distance threshold in the same units as pos.
    Returns:
        (N, N) bool tensor; entry [i, j] True if j is a neighbour of i.
    """
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)      # (N, N, 2)
    dist = diff.norm(dim=-1)                          # (N, N)
    mask = (dist < thresh) & (dist > 1e-6)
    return mask

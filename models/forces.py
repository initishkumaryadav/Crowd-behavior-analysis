"""Social-force modules for NSP-SFM (survey Eqs. 12-14).

Each force returns an acceleration contribution (p-double-dot) in R^2. Neural
networks predict the *parameters* of otherwise physically-explicit forces, which
is the core idea of Neural Social Physics: keep the SFM structure, learn the
knobs that previously had to be hand-tuned.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim, hidden, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class GoalForce(nn.Module):
    r"""F_goal = (1/tau) (v_des - p_dot),  v_des = (dest - p)/((T-t) dt),  tau = NN(q, dest).

    tau is a learned relaxation time (kept positive via softplus). q = [p, p_dot].
    """
    def __init__(self, hidden: int):
        super().__init__()
        self.tau_net = _mlp(4 + 2, hidden, 1)  # input: (p, p_dot, dest)

    def forward(self, p, p_dot, dest, steps_left, dt):
        # steps_left = (T - t): integer >= 1, remaining steps to reach dest.
        # ETH/UCY positions are already per-scene normalized, so we feed them
        # to the network directly (v_des is the world-space physics term).
        q = torch.cat([p, p_dot, dest], dim=-1)          # (N, 6)
        tau = F.softplus(self.tau_net(q)) + 1e-2         # (N, 1), strictly positive
        v_des = (dest - p) / (steps_left * dt)           # (N, 2)
        return (v_des - p_dot) / tau                     # (N, 2)


class CollisionForce(nn.Module):
    r"""Inter-agent repulsion (Eq. 13).

    Potential  U_ij = r_col * k_ij * exp(-||r_ij|| / r_col),  r_ij = p_i - p_j.
    Force on i  F_i = sum_j -grad_{r_ij} U_ij
                    = sum_j k_ij * exp(-||r_ij||/r_col) * (r_ij / ||r_ij||).
    k_ij = softplus(NN(q_i, q_j)) keeps the potential repulsive.
    """
    def __init__(self, hidden: int, r_col: float):
        super().__init__()
        self.r_col = r_col
        self.k_net = _mlp(4 + 4, hidden, 1)   # input: (q_i, q_j) = (p_i,pd_i, p_j,pd_j)

    def k_ij(self, qi, qj):
        return F.softplus(self.k_net(torch.cat([qi, qj], dim=-1)))  # (..., 1)

    def forward(self, p, p_dot, seq_start_end):
        """p, p_dot: (N, 2). Returns per-agent collision acceleration (N, 2)."""
        out = torch.zeros_like(p)
        q = torch.cat([p, p_dot], dim=-1)                 # (N, 4)
        for s, e in seq_start_end:
            n = e - s
            if n < 2:
                continue
            pi = p[s:e]                                    # (n, 2)
            qi = q[s:e]                                    # (n, 4)
            r = pi.unsqueeze(1) - pi.unsqueeze(0)          # (n, n, 2) = p_i - p_j
            dist = r.norm(dim=-1, keepdim=True)            # (n, n, 1)
            eye = torch.eye(n, dtype=torch.bool, device=p.device).unsqueeze(-1)
            safe = dist.clamp_min(1e-6)
            direction = r / safe                           # unit vector i<-j
            qi_e = qi.unsqueeze(1).expand(n, n, 4)
            qj_e = qi.unsqueeze(0).expand(n, n, 4)
            k = self.k_ij(qi_e.reshape(-1, 4), qj_e.reshape(-1, 4)).reshape(n, n, 1)
            mag = k * torch.exp(-safe / self.r_col)        # (n, n, 1)
            contrib = mag * direction                      # (n, n, 2)
            contrib = contrib.masked_fill(eye, 0.0)        # no self-interaction
            out[s:e] = contrib.sum(dim=1)                  # sum over neighbours j
        return out

    # --- explicit potential, used only for the gradient unit-test ---
    def potential(self, pi, pj, qi, qj):
        rij = pi - pj
        dist = rij.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        k = self.k_ij(qi, qj)
        return (self.r_col * k * torch.exp(-dist / self.r_col)).squeeze(-1)


class EnvForce(nn.Module):
    r"""Environmental repulsion (Eq. 14): F_env = k_env * (p - p_obs)/||p - p_obs||^2.

    Requires nearest-obstacle positions p_obs. Disabled for datasets without an
    obstacle map (ETH/UCY); see README.
    """
    def __init__(self):
        super().__init__()
        self.k_env = nn.Parameter(torch.tensor(1.0))

    def forward(self, p, p_obs):
        d = p - p_obs
        denom = d.norm(dim=-1, keepdim=True).clamp_min(1e-6) ** 2
        return self.k_env * d / denom

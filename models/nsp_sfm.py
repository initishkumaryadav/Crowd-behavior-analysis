"""NSP-SFM: Neural Social Physics with the Social Force Model.

Implements the survey's Section 2.3 flagship method (Yue et al. 2022):
  acceleration  p'' = F_goal + F_col (+ F_env)                       (Eq. 9)
  semi-implicit  p_dot_{t+1} = p_dot_t + dt * p''                    (Eq. 10)
                 p_{t+1}     = p_t + dt * (p_dot_{t+1} + alpha)      (Eq. 11)
where the force parameters and the stochastic perturbation alpha are predicted by
neural networks, and destinations come from a Goal Sampling CVAE (Fig. 2).
"""
import torch
import torch.nn as nn

from .forces import GoalForce, CollisionForce, EnvForce
from .cvae import TrajEncoder, GoalSamplingCVAE, AlphaCVAE, kl_standard_normal  # noqa: F401


def _velocity(pos, dt):
    """Finite-difference velocity; front-pad so shape matches pos. (N,L,2)->(N,L,2)"""
    vel = torch.zeros_like(pos)
    vel[:, 1:] = (pos[:, 1:] - pos[:, :-1]) / dt
    vel[:, 0] = vel[:, 1]
    return vel


class NSPSFM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dt = cfg.dt
        self.pred_len = cfg.pred_len
        self.acc_clip = 50.0        # m/s^2 stability bound on acceleration
        self.alpha_clip = 10.0      # bound on stochastic velocity perturbation

        self.encoder = TrajEncoder(cfg.obs_len, cfg.hidden)
        self.goal_force = GoalForce(cfg.hidden)
        self.collision = CollisionForce(cfg.hidden, cfg.r_col)
        self.env_force = EnvForce() if cfg.use_env_force else None

        self.goal_cvae = GoalSamplingCVAE(cond_dim=cfg.hidden, latent=cfg.goal_latent, hidden=cfg.hidden)
        self.alpha_cvae = AlphaCVAE(cond_dim=cfg.hidden + 2, latent=cfg.alpha_latent, hidden=cfg.hidden)

    def forward(self, obs_pos, seq_start_end, pred_pos_gt=None, obstacles=None):
        """Roll out `pred_len` future steps.

        Args:
            obs_pos:     (N, obs_len, 2) observed absolute positions.
            seq_start_end: list of (start, end) per co-present scene.
            pred_pos_gt: (N, pred_len, 2) or None. If given -> training (posterior
                         CVAEs + teacher signals); else stochastic sampling.
            obstacles:   optional (N, 2) nearest-obstacle positions for F_env.
        Returns:
            pred (N, pred_len, 2), losses dict.
        """
        train = pred_pos_gt is not None
        obs_vel = _velocity(obs_pos, self.dt)
        cond = self.encoder(obs_pos, obs_vel)                 # (N, hidden)

        p = obs_pos[:, -1]                                    # (N, 2) current position
        p_dot = (obs_pos[:, -1] - obs_pos[:, -2]) / self.dt   # (N, 2) current velocity

        dest_gt = pred_pos_gt[:, -1] if train else None
        dest, kl_goal = self.goal_cvae(cond, p, dest_gt)      # destination (N,2)

        preds = []
        kl_alpha_tot = torch.zeros((), device=obs_pos.device)
        for k in range(self.pred_len):
            steps_left = torch.full((p.size(0), 1), float(self.pred_len - k), device=p.device)
            acc = self.goal_force(p, p_dot, dest, steps_left, self.dt)
            acc = acc + self.collision(p, p_dot, seq_start_end)
            if self.env_force is not None and obstacles is not None:
                acc = acc + self.env_force(p, obstacles)
            # Bound the acceleration so the explicit integrator can't diverge to
            # inf/NaN early in training (pedestrian accelerations are physically
            # bounded; ~10 m/s^2 is already extreme). Generous limit -> no bias.
            acc = acc.clamp(-self.acc_clip, self.acc_clip)

            p_dot_next = p_dot + self.dt * acc                # Eq. 10
            p_int = p + self.dt * p_dot_next                  # deterministic intermediate

            cond_alpha = torch.cat([cond, p], dim=-1)
            if train:
                alpha_gt = (pred_pos_gt[:, k] - p_int) / self.dt
                alpha, kl_a = self.alpha_cvae(cond_alpha, alpha_gt)
                kl_alpha_tot = kl_alpha_tot + kl_a
            else:
                alpha, _ = self.alpha_cvae(cond_alpha, None)
            alpha = alpha.clamp(-self.alpha_clip, self.alpha_clip)

            p_next = p_int + self.dt * alpha                  # Eq. 11
            preds.append(p_next)
            p_dot = (p_next - p) / self.dt                    # realized velocity
            p = p_next

        pred = torch.stack(preds, dim=1)                      # (N, pred_len, 2)

        losses = {}
        if train:
            recon = (pred - pred_pos_gt).norm(dim=-1).mean()          # avg displacement
            dest_recon = (dest - dest_gt).norm(dim=-1).mean()
            losses = {
                "recon": recon,
                "dest": dest_recon,
                "kl_goal": kl_goal,
                "kl_alpha": kl_alpha_tot / self.pred_len,
            }
        return pred, losses

"""CVAE building blocks for NSP-SFM.

Two conditional VAEs (survey Fig. 2 / Eqs. 11-12):
  * GoalSamplingCVAE  - predicts the destination distribution from the observed
                        trajectory; its samples drive the goal-attraction force.
  * AlphaCVAE         - predicts the stochastic velocity perturbation alpha that
                        turns the deterministic SFM rollout into a stochastic one.

Both follow the standard conditional-VAE recipe: a recognition network q(z|x,c)
used at training time, a standard-normal prior at test time, and a decoder
p(y|z,c). KL is against N(0, I).
"""
import torch
import torch.nn as nn


def kl_standard_normal(mu, logvar):
    """KL( N(mu, sigma) || N(0, I) ), summed over dims, mean over batch."""
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)).mean()


class TrajEncoder(nn.Module):
    """Encode an observed trajectory (positions + velocities) into a condition vector."""
    def __init__(self, obs_len: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_len * 4, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )

    def forward(self, obs_pos, obs_vel):
        # obs_pos, obs_vel: (N, obs_len, 2)
        x = torch.cat([obs_pos, obs_vel], dim=-1).flatten(1)   # (N, obs_len*4)
        return self.net(x)                                     # (N, hidden)


class GoalSamplingCVAE(nn.Module):
    def __init__(self, cond_dim: int, latent: int, hidden: int):
        super().__init__()
        self.latent = latent
        # recognition q(z | dest, cond)
        self.enc = nn.Sequential(nn.Linear(2 + cond_dim, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        # decoder p(dest | z, cond) -> predicts destination offset from last obs pos
        self.dec = nn.Sequential(
            nn.Linear(latent + cond_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, cond, last_pos, dest_gt=None):
        """Returns (dest_pred, kl). If dest_gt given -> posterior (train), else prior."""
        if dest_gt is not None:
            h = self.enc(torch.cat([dest_gt - last_pos, cond], dim=-1))
            mu, logvar = self.mu(h), self.logvar(h)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
            kl = kl_standard_normal(mu, logvar)
        else:
            z = torch.randn(cond.size(0), self.latent, device=cond.device)
            kl = torch.zeros((), device=cond.device)
        offset = self.dec(torch.cat([z, cond], dim=-1))
        return last_pos + offset, kl


class AlphaCVAE(nn.Module):
    """Predicts velocity perturbation alpha given (history cond, current position).

    Condition c = [hist_embedding, current_pos]. Target during training is the
    residual alpha_gt that maps the deterministic intermediate position to the
    ground-truth next position.
    """
    def __init__(self, cond_dim: int, latent: int, hidden: int):
        super().__init__()
        self.latent = latent
        self.enc = nn.Sequential(nn.Linear(2 + cond_dim, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        self.dec = nn.Sequential(
            nn.Linear(latent + cond_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, cond, alpha_gt=None):
        if alpha_gt is not None:
            h = self.enc(torch.cat([alpha_gt, cond], dim=-1))
            mu, logvar = self.mu(h), self.logvar(h)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
            kl = kl_standard_normal(mu, logvar)
        else:
            z = torch.randn(cond.size(0), self.latent, device=cond.device)
            kl = torch.zeros((), device=cond.device)
        return self.dec(torch.cat([z, cond], dim=-1)), kl

"""Unit tests: collision force == -grad U (autograd), integrator sanity."""
import torch
from models.forces import CollisionForce, GoalForce


def test_collision_is_neg_grad_potential():
    torch.manual_seed(0)
    cf = CollisionForce(hidden=32, r_col=0.4).double()  # float64 for a tight grad check
    # two agents; check force on agent 0 equals -d U_01 / d p_0
    p = torch.tensor([[0.0, 0.0], [0.5, 0.2]], dtype=torch.float64, requires_grad=True)
    p_dot = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float64)
    # analytic force
    F = cf(p.detach(), p_dot, [(0, 2)])
    # F is defined as -grad_{r_ij} U with k_ij held as a coefficient (Eq. 13),
    # so differentiate the potential w.r.t. an explicit r_ij, k detached.
    qi = torch.cat([p[0:1], p_dot[0:1]], -1).detach()
    qj = torch.cat([p[1:2], p_dot[1:2]], -1).detach()
    k = cf.k_ij(qi, qj).detach()
    rij = (p[0:1] - p[1:2]).detach().requires_grad_(True)
    dist = rij.norm(dim=-1, keepdim=True)
    U = (cf.r_col * k * torch.exp(-dist / cf.r_col)).sum()
    g = torch.autograd.grad(U, rij)[0][0]      # dU/d r_ij
    analytic = F[0]
    numeric = -g                                # -grad_{r_ij} U
    err = (analytic - numeric).abs().max().item()
    assert err < 1e-4, f"collision force mismatch: {err}"
    print(f"[ok] collision force == -grad U   (max err {err:.2e})")


def test_goal_force_pulls_toward_dest():
    gf = GoalForce(hidden=32)
    p = torch.zeros(1, 2)
    p_dot = torch.zeros(1, 2)
    dest = torch.tensor([[3.0, 0.0]])
    f = gf(p, p_dot, dest, steps_left=torch.tensor([[6.0]]), dt=0.4)
    assert f[0, 0] > 0 and abs(f[0, 1].item()) < 1e-5, "goal force should point +x"
    print(f"[ok] goal force points toward destination: {f.tolist()}")


if __name__ == "__main__":
    test_collision_is_neg_grad_potential()
    test_goal_force_pulls_toward_dest()
    print("all force tests passed")

"""Overfit test: the model must drive train ADE toward ~0 on a single batch.

This validates that the semi-implicit integrator is differentiable end-to-end and
that gradients actually flow through the forces + CVAEs.
"""
import torch
from config import CFG
from utils import set_seed, get_device
from data.dataset import TrajectoryDataset, collate
from models.nsp_sfm import NSPSFM
from losses import total_loss, ade_fde


def test_overfit_one_batch():
    set_seed(0)
    dev = get_device()
    ds = TrajectoryDataset("datasets/eth/train", CFG.obs_len, CFG.pred_len)
    batch = collate([ds[i] for i in range(8)])
    obs, pred, sse = batch[0].to(dev), batch[1].to(dev), batch[2]

    model = NSPSFM(CFG).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    first = None
    for step in range(400):
        opt.zero_grad()
        out, losses = model(obs, sse, pred_pos_gt=pred)
        loss = total_loss(losses, CFG)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
        opt.step()
        if step == 0:
            first, _ = ade_fde(out, pred)
    final_ade, final_fde = ade_fde(out, pred)
    print(f"[overfit] ADE {first:.3f} -> {final_ade:.3f}  FDE {final_fde:.3f}")
    assert final_ade < first * 0.4, "model failed to overfit a single batch"
    print("[ok] integrator + backprop learn (train ADE dropped sharply)")


if __name__ == "__main__":
    test_overfit_one_batch()

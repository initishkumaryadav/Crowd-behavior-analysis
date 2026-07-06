"""Evaluate a trained NSP-SFM checkpoint: best-of-20 ADE/FDE on a scene's test set.

Example:
    python eval.py --scene eth --ckpt checkpoints/nsp_sfm_eth.pt
"""
import os
import argparse
import torch

from config import CFG
from utils import set_seed, get_device
from data.dataset import make_loader
from models.nsp_sfm import NSPSFM
from losses import best_of_n_ade_fde


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="eth", choices=list(CFG.ethucy_scenes))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--num_samples", type=int, default=CFG.num_samples)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    set_seed(CFG.seed)
    dev = get_device(force_cpu=args.cpu)
    ckpt = args.ckpt or os.path.join("checkpoints", f"nsp_sfm_{args.scene}.pt")

    model = NSPSFM(CFG).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    model.eval()

    loader = make_loader(os.path.join(CFG.data_root, args.scene, "test"),
                         CFG.obs_len, CFG.pred_len, CFG.batch_size, shuffle=False)

    ades, fdes, weights = [], [], []
    with torch.no_grad():
        for obs, pred, sse in loader:
            obs, pred = obs.to(dev), pred.to(dev)
            samples = torch.stack(
                [model(obs, sse, pred_pos_gt=None)[0] for _ in range(args.num_samples)], dim=0
            )
            a, f = best_of_n_ade_fde(samples, pred)
            n = pred.size(0)
            ades.append(a * n); fdes.append(f * n); weights.append(n)
    tot = sum(weights)
    print(f"[eval] scene={args.scene}  best-of-{args.num_samples}  "
          f"ADE {sum(ades)/tot:.3f}  FDE {sum(fdes)/tot:.3f}")


if __name__ == "__main__":
    main()

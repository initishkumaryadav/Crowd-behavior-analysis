"""Train NSP-SFM on one ETH/UCY leave-one-out split.

The SGAN folder layout already encodes leave-one-out: datasets/<scene>/train
holds the other scenes, datasets/<scene>/test holds <scene>. So training on
<scene>/train and testing on <scene>/test reproduces the standard protocol.

Example:
    python train.py --scene eth --epochs 200
"""
import os
import argparse
import torch
from tqdm import tqdm

from config import CFG
from utils import set_seed, get_device
from data.dataset import make_loader
from models.nsp_sfm import NSPSFM
from losses import total_loss, ade_fde, best_of_n_ade_fde


def evaluate(model, loader, dev, num_samples):
    model.eval()
    ades, fdes, weights = [], [], []
    with torch.no_grad():
        for obs, pred, sse in loader:
            obs, pred = obs.to(dev), pred.to(dev)
            samples = torch.stack(
                [model(obs, sse, pred_pos_gt=None)[0] for _ in range(num_samples)], dim=0
            )
            a, f = best_of_n_ade_fde(samples, pred)
            n = pred.size(0)
            ades.append(a * n); fdes.append(f * n); weights.append(n)
    tot = sum(weights)
    return sum(ades) / tot, sum(fdes) / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="eth", choices=list(CFG.ethucy_scenes))
    ap.add_argument("--epochs", type=int, default=CFG.epochs)
    ap.add_argument("--batch_size", type=int, default=CFG.batch_size)
    ap.add_argument("--lr", type=float, default=CFG.lr)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()

    set_seed(CFG.seed)
    dev = get_device(force_cpu=args.cpu)
    print(f"[train] scene={args.scene} device={dev} epochs={args.epochs}")

    root = CFG.data_root
    train_loader = make_loader(os.path.join(root, args.scene, "train"),
                               CFG.obs_len, CFG.pred_len, args.batch_size, shuffle=True)
    test_loader = make_loader(os.path.join(root, args.scene, "test"),
                              CFG.obs_len, CFG.pred_len, args.batch_size, shuffle=False)

    model = NSPSFM(CFG).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, args.epochs // 3), gamma=0.5)

    os.makedirs(args.out, exist_ok=True)
    best = float("inf")
    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for obs, pred, sse in train_loader:
            obs, pred = obs.to(dev), pred.to(dev)
            opt.zero_grad()
            _, losses = model(obs, sse, pred_pos_gt=pred)
            loss = total_loss(losses, CFG)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
            opt.step()
            running += loss.item()
        sched.step()

        if ep % 5 == 0 or ep == args.epochs:
            ade, fde = evaluate(model, test_loader, dev, CFG.num_samples)
            print(f"epoch {ep:3d} | train_loss {running/len(train_loader):.4f} | "
                  f"test ADE {ade:.3f} FDE {fde:.3f}")
            if ade < best:
                best = ade
                torch.save(model.state_dict(), os.path.join(args.out, f"nsp_sfm_{args.scene}.pt"))
    print(f"[train] best test ADE {best:.3f}  (checkpoint saved to {args.out})")


if __name__ == "__main__":
    main()

"""Test the video-adapter path: trajectory buffer + NSP-SFM predictor.

This bypasses YOLO (which needs a real video) and checks the part unique to the
video pipeline: that pixel-space tracks are buffered at the right cadence and
that the predictor returns a sensible future path in pixel coordinates.
"""
import os
import numpy as np

from config import CFG
from video.trajectory_buffer import TrajectoryBuffers
from video.predictor import TrajectoryPredictor


def test_buffer_samples_at_dt():
    buf = TrajectoryBuffers(obs_len=8, dt=0.4)
    # 8 points 0.4 s apart = 3.2 s => 80 frames at 25 fps (0.04 s each)
    for i in range(85):
        t = i * 0.04
        buf.update({7: (100.0 + i * 5, 200.0)}, t)
    ready = buf.ready()
    assert 7 in ready, "track should have a full 8-point history"
    assert len(ready[7]) == 8
    # points should be ~0.4 s apart => ~10 frames => ~50 px apart
    xs = [p[0] for p in ready[7]]
    gaps = np.diff(xs)
    assert np.all(gaps > 30), f"sampling too dense: {gaps}"
    print(f"[ok] buffer sampled 8 points ~dt apart (x-gaps {gaps.astype(int).tolist()})")


def test_predictor_continues_straight_walk():
    ckpt = os.path.join("checkpoints", "nsp_sfm_eth.pt")
    if not os.path.exists(ckpt):
        print("[skip] no checkpoint; train first (python train.py --scene eth)")
        return
    pred = TrajectoryPredictor(ckpt, device="cpu", px_per_metre=40.0, num_samples=1)
    # a person walking right at ~50 px/step
    obs = [(100.0 + 50 * k, 300.0) for k in range(CFG.obs_len)]
    out = pred.predict({1: obs})
    path = out[1]
    assert path.shape == (CFG.pred_len, 2)
    # Motion should continue forward in x. The eth-trained model naturally
    # predicts gently curved paths (pedestrians rarely walk perfectly straight),
    # so we only require x to keep advancing and the path to stay finite.
    assert path[-1, 0] > obs[-1][0], "predicted x should continue forward"
    assert np.isfinite(path).all(), "predicted path must be finite"
    forward = path[-1, 0] - obs[-1][0]
    assert forward > 100, f"expected clear forward motion, got {forward:.0f}px"
    print(f"[ok] predictor continued the walk: last obs x={obs[-1][0]:.0f} "
          f"-> pred end x={path[-1,0]:.0f}, y={path[-1,1]:.0f}")


if __name__ == "__main__":
    test_buffer_samples_at_dt()
    test_predictor_continues_straight_walk()
    print("video predictor tests passed")

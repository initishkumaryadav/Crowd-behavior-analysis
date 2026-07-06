"""Wrap a trained NSP-SFM model for one-shot inference on live tracks.

The model was trained on ETH/UCY trajectories measured in **metres** with small
coordinate values, whereas a video gives us **pixel** coordinates in the hundreds
or thousands. Feeding raw pixels would push the network far outside its training
range. So for each observed window we:

  1. shift it to the person's last observed point (translation to ~0), and
  2. divide by `px_per_metre` to convert pixels into approximate metres.

We run the model in that normalised space, then invert the transform to get the
predicted path back in pixels for drawing.

NOTE (accuracy): `px_per_metre` is a rough global scale, not a true camera
calibration. It makes predictions *visually sensible* immediately. For metric
accuracy on a specific camera, replace this with a homography (see
VIDEO_PIPELINE.md, "Calibration").
"""
from typing import Dict, List, Tuple

import numpy as np
import torch

from config import CFG
from models.nsp_sfm import NSPSFM


class TrajectoryPredictor:
    def __init__(self, checkpoint: str, device: str = "cuda", px_per_metre: float = 40.0,
                 num_samples: int = 1):
        """
        Args:
            checkpoint:    path to a trained NSP-SFM .pt file.
            device:        "cuda" or "cpu".
            px_per_metre:  pixels that correspond to ~1 metre in the scene.
            num_samples:   stochastic rollouts per person; we average them for a
                           single smooth predicted path (raise for diversity).
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.px_per_metre = px_per_metre
        self.num_samples = num_samples
        self.model = NSPSFM(CFG).to(self.device)
        self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))
        self.model.eval()

    @torch.no_grad()
    def predict(self, windows: Dict[int, List[Tuple[float, float]]]
                ) -> Dict[int, np.ndarray]:
        """Predict future pixel paths for a batch of people.

        Args:
            windows: {track_id: [obs_len (x, y) pixel points]}.
        Returns:
            {track_id: (pred_len, 2) predicted pixel path}.
        """
        if not windows:
            return {}

        ids = list(windows.keys())
        obs_px = np.stack([np.asarray(windows[i], dtype=np.float32) for i in ids])  # (N, obs, 2)
        ref = obs_px[:, -1:, :]                                   # (N, 1, 2) last obs point
        obs_norm = (obs_px - ref) / self.px_per_metre            # -> metre-ish, centred
        obs_t = torch.from_numpy(obs_norm).to(self.device)

        # Each person is treated as its own scene here (no cross-person repulsion
        # in pixel space by default). See VIDEO_PIPELINE.md to enable social
        # interaction once a common ground plane is calibrated.
        seq_start_end = [(k, k + 1) for k in range(len(ids))]

        samples = torch.stack(
            [self.model(obs_t, seq_start_end, pred_pos_gt=None)[0]
             for _ in range(self.num_samples)], dim=0
        )                                                         # (S, N, pred, 2)
        pred_norm = samples.mean(dim=0).cpu().numpy()             # (N, pred, 2)
        pred_px = pred_norm * self.px_per_metre + ref             # back to pixels

        return {tid: pred_px[k] for k, tid in enumerate(ids)}

"""Reusable crowd-prediction pipeline core.

`CrowdPipeline` ties the four stages together (track -> buffer -> predict -> draw)
and exposes them as a simple generator of annotated frames plus a live `stats`
dict. Both the command-line tool (`stream_predict.py`) and the web app import
this, so there is exactly one copy of the core logic.

Usage:
    pipe = CrowdPipeline(cfg)
    for annotated_bgr in pipe.frames():
        ...            # show / save / stream the frame
    print(pipe.stats) # totals after the stream ends
"""
from typing import Iterator

import numpy as np

from video.tracker import PersonTracker
from video.trajectory_buffer import TrajectoryBuffers
from video.predictor import TrajectoryPredictor
from video.visualize import draw_overlays


class CrowdPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tracker = PersonTracker(cfg)
        self.buffers = TrajectoryBuffers(obs_len=cfg.obs_len, dt=cfg.dt)
        self.predictor = TrajectoryPredictor(
            checkpoint=cfg.checkpoint, device=cfg.device,
            px_per_metre=cfg.px_per_metre,
        )
        # live-updating counters (safe to read from another thread for a status API)
        self.stats = {
            "frames": 0,
            "unique_people": 0,
            "people_now": 0,
            "max_people": 0,
            "frames_with_predictions": 0,
            "total_predictions": 0,
            "done": False,
        }
        self._seen_ids = set()

    def frames(self) -> Iterator[np.ndarray]:
        """Yield annotated BGR frames one at a time, updating `self.stats`."""
        cfg = self.cfg
        for ft in self.tracker.stream(cfg.source):
            if cfg.frame_stride > 1 and (ft.frame_index % cfg.frame_stride):
                continue

            self.buffers.update(ft.points, ft.timestamp)
            windows = self.buffers.ready()
            if len(windows) > cfg.max_tracks:
                windows = dict(list(windows.items())[:cfg.max_tracks])
            predictions = self.predictor.predict(windows)

            histories = {tid: self.buffers.history(tid) for tid in ft.points}
            annotated = draw_overlays(ft.frame, histories, predictions)

            # ---- stats ----
            self._seen_ids.update(ft.points.keys())
            s = self.stats
            s["frames"] += 1
            s["people_now"] = len(ft.points)
            s["max_people"] = max(s["max_people"], len(ft.points))
            s["unique_people"] = len(self._seen_ids)
            if predictions:
                s["frames_with_predictions"] += 1
                s["total_predictions"] += len(predictions)

            yield annotated

        self.stats["done"] = True

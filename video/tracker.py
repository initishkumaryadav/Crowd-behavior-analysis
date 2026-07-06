"""YOLO + ByteTrack person tracker.

Wraps Ultralytics YOLO so the rest of the pipeline never touches detection
details. Given a video source, it yields, frame by frame, a dictionary mapping
each tracked person's ID to their **foot point** (bottom-center of the bounding
box) — the pixel location we treat as "where the person stands on the ground".

Why the foot point? A person's feet are (approximately) on the ground plane, so
they are the most sensible 2-D position for trajectory modelling. The box center
would sit around the torso and drift with posture/height.
"""
from dataclasses import dataclass
from typing import Iterator, Dict, Tuple

import numpy as np


@dataclass
class FrameTracks:
    """Everything the loop needs about one processed frame."""
    frame: np.ndarray                       # BGR image (H, W, 3)
    points: Dict[int, Tuple[float, float]]  # track_id -> (x, y) foot point in pixels
    frame_index: int
    timestamp: float                        # seconds (wall-clock for live, else frame/fps)


class PersonTracker:
    def __init__(self, cfg):
        # Imported lazily so the trajectory-prediction code can be used without
        # the (heavy) detection dependencies installed.
        from ultralytics import YOLO
        self.cfg = cfg
        self.model = YOLO(cfg.yolo_model)

    def _probe_fps(self, source) -> float:
        """Read the source frame rate (for file timestamps). Falls back to 25."""
        import cv2
        cap = cv2.VideoCapture(source if not str(source).isdigit() else int(source))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps if fps and fps > 1e-3 else 25.0

    def stream(self, source) -> Iterator[FrameTracks]:
        """Yield FrameTracks for each frame of `source`.

        Uses Ultralytics streaming mode with `persist=True` so ByteTrack keeps
        IDs stable across frames — essential for building per-person paths.
        Timestamps are always in **seconds**: frame_index / fps for a recorded
        file, or true wall-clock time for a live camera (frames may drop live).
        """
        import time

        live = self.cfg.use_wallclock_sampling
        fps = None if live else self._probe_fps(source)

        results = self.model.track(
            source=source,
            stream=True,            # yield frames one at a time (real-time friendly)
            persist=True,           # keep track IDs alive across frames
            classes=[self.cfg.person_class_id],
            conf=self.cfg.conf_thresh,
            tracker=self.cfg.tracker_cfg,
            device=self.cfg.device,
            verbose=False,
        )

        start = time.time()
        for idx, res in enumerate(results):
            frame = res.orig_img
            points: Dict[int, Tuple[float, float]] = {}
            boxes = res.boxes
            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                ids = boxes.id.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), tid in zip(xyxy, ids):
                    foot_x = 0.5 * (x1 + x2)     # bottom-center x
                    foot_y = float(y2)            # bottom edge = feet on ground
                    points[int(tid)] = (float(foot_x), foot_y)

            ts = (time.time() - start) if live else (idx / fps)
            yield FrameTracks(frame=frame, points=points, frame_index=idx, timestamp=ts)

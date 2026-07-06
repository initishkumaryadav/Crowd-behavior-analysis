"""Draw tracking + prediction overlays onto a frame.

Colour convention:
  * green  : the person's observed (past) path and current foot point
  * red    : the model's predicted future path
  * yellow : track ID label

Everything is drawn on a copy so the original frame is left untouched.
"""
from typing import Dict, List, Tuple

import cv2
import numpy as np

GREEN = (0, 200, 0)
RED = (0, 0, 255)
YELLOW = (0, 220, 220)


def draw_overlays(frame: np.ndarray,
                  histories: Dict[int, List[Tuple[float, float]]],
                  predictions: Dict[int, np.ndarray]) -> np.ndarray:
    """Return an annotated copy of `frame`.

    Args:
        frame:       BGR image.
        histories:   {id: [past (x, y) points]} — observed path (green).
        predictions: {id: (pred_len, 2) array} — predicted path (red).
    """
    out = frame.copy()

    # observed paths + current position + ID
    for tid, hist in histories.items():
        pts = [(int(x), int(y)) for x, y in hist]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(out, a, b, GREEN, 2)
        if pts:
            cv2.circle(out, pts[-1], 4, GREEN, -1)
            cv2.putText(out, f"ID {tid}", (pts[-1][0] + 6, pts[-1][1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1, cv2.LINE_AA)

    # predicted future paths (dashed-ish via segments), starting from last obs point
    for tid, pred in predictions.items():
        hist = histories.get(tid, [])
        start = (int(hist[-1][0]), int(hist[-1][1])) if hist else None
        pts = [(int(x), int(y)) for x, y in pred]
        chain = ([start] + pts) if start else pts
        for a, b in zip(chain[:-1], chain[1:]):
            cv2.line(out, a, b, RED, 2)
        if pts:
            cv2.circle(out, pts[-1], 4, RED, -1)   # predicted end point

    # small legend
    cv2.putText(out, "green=observed  red=predicted", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out

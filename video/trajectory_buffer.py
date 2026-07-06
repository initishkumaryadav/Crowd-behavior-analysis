"""Per-person rolling trajectory buffers.

The tracker reports a foot point for every person on *every* frame (e.g. 25/sec),
but NSP-SFM wants points spaced `dt` seconds apart (0.4 s -> 2.5/sec). This module
bridges the two: for each tracked person it keeps the most recent `obs_len`
*time-sampled* positions, adding a new one only once `dt` seconds have elapsed
since that person's last stored point.

It also forgets people who have left the scene, so memory stays bounded on long
or live streams.
"""
from collections import deque
from typing import Dict, List, Tuple


class TrajectoryBuffers:
    def __init__(self, obs_len: int, dt: float, forget_after: float = 2.0):
        """
        Args:
            obs_len:      how many past points a person needs before we predict.
            dt:           desired spacing between stored points (seconds).
            forget_after: drop a track if unseen for this many seconds.
        """
        self.obs_len = obs_len
        self.dt = dt
        self.forget_after = forget_after
        self._buf: Dict[int, deque] = {}          # id -> deque[(x, y)]
        self._last_t: Dict[int, float] = {}       # id -> timestamp of last stored point
        self._seen_t: Dict[int, float] = {}       # id -> timestamp last seen at all

    def update(self, points: Dict[int, Tuple[float, float]], t: float):
        """Feed one frame's detections; store time-sampled points, forget stale IDs."""
        for tid, xy in points.items():
            self._seen_t[tid] = t
            last = self._last_t.get(tid)
            if last is None or (t - last) >= self.dt:
                dq = self._buf.setdefault(tid, deque(maxlen=self.obs_len))
                dq.append(xy)
                self._last_t[tid] = t

        # forget tracks not seen recently
        stale = [tid for tid, seen in self._seen_t.items() if (t - seen) > self.forget_after]
        for tid in stale:
            self._buf.pop(tid, None)
            self._last_t.pop(tid, None)
            self._seen_t.pop(tid, None)

    def ready(self) -> Dict[int, List[Tuple[float, float]]]:
        """Return {id: [obs_len points]} only for people with a full history."""
        return {tid: list(dq) for tid, dq in self._buf.items() if len(dq) == self.obs_len}

    def history(self, tid: int) -> List[Tuple[float, float]]:
        """The stored (possibly partial) history of one track — used for drawing."""
        return list(self._buf.get(tid, []))

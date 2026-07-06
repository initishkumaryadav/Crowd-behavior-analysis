"""Generate a tiny synthetic crowd video for smoke-testing the video pipeline.

Draws a few coloured rectangles ("people") walking across a frame. YOLO will not
reliably detect these abstract shapes — this clip's purpose is to exercise the
file-reading / buffering / drawing / saving plumbing end to end without needing a
real crowd video. For an actual detection demo, use a real pedestrian clip.
"""
import numpy as np
import cv2

from video.config import fourcc_code

OUT = "sample_data/synthetic_crowd.mp4"


def main():
    import os
    os.makedirs("sample_data", exist_ok=True)
    W, H, fps, n_frames = 640, 480, 25, 100
    writer = cv2.VideoWriter(OUT, fourcc_code("mp4v"), fps, (W, H))
    people = [dict(x=50, y=200, vx=4, vy=1, c=(0, 0, 255)),
              dict(x=600, y=260, vx=-3, vy=-1, c=(255, 0, 0)),
              dict(x=100, y=400, vx=3, vy=-2, c=(0, 255, 0))]
    for _ in range(n_frames):
        frame = np.full((H, W, 3), 40, np.uint8)
        for p in people:
            p["x"] += p["vx"]; p["y"] += p["vy"]
            x, y = int(p["x"]), int(p["y"])
            cv2.rectangle(frame, (x - 12, y - 30), (x + 12, y + 30), p["c"], -1)
        writer.write(frame)
    writer.release()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

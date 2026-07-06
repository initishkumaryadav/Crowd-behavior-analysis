"""Real-time crowd trajectory prediction from video.

Ties the pipeline together:
    source (file | webcam | RTSP)
      -> PersonTracker      (YOLO + ByteTrack, per-frame foot points)
      -> TrajectoryBuffers  (per-person history sampled every dt seconds)
      -> TrajectoryPredictor(NSP-SFM: 8 past -> 12 future)
      -> draw_overlays      (green=observed, red=predicted)
      -> show window and/or save annotated video

The SAME code serves a recorded file (for trials) and a live camera. Only the
`--source` argument changes:
    python -m video.stream_predict --source path/to/video.mp4        # recorded
    python -m video.stream_predict --source 0                         # webcam
    python -m video.stream_predict --source rtsp://user:pass@ip/stream  # IP cam

Press 'q' in the preview window to quit.
"""
import argparse
import sys

import cv2

from video.config import VideoConfig, resolve_device, fourcc_code
from video.pipeline import CrowdPipeline


def parse_args() -> VideoConfig:
    cfg = VideoConfig()
    ap = argparse.ArgumentParser(description="Crowd trajectory prediction from video.")
    ap.add_argument("--source", default=cfg.source,
                    help="video file path, webcam index (e.g. 0), or RTSP URL")
    ap.add_argument("--checkpoint", default=cfg.checkpoint, help="trained NSP-SFM .pt")
    ap.add_argument("--yolo", default=cfg.yolo_model, help="YOLO weights (yolo11n/s/m.pt)")
    ap.add_argument("--device", default=cfg.device, choices=["auto", "cuda", "cpu"])
    ap.add_argument("--dt", type=float, default=cfg.dt, help="seconds between sampled points")
    ap.add_argument("--px-per-metre", type=float, default=40.0,
                    help="rough pixels-per-metre scale (see VIDEO_PIPELINE.md)")
    ap.add_argument("--save", default="", help="path to write annotated video (optional)")
    ap.add_argument("--no-show", action="store_true", help="disable the preview window")
    ap.add_argument("--frame-stride", type=int, default=cfg.frame_stride,
                    help="process every Nth frame (raise to go faster)")
    args = ap.parse_args()

    cfg.source = args.source
    cfg.checkpoint = args.checkpoint
    cfg.yolo_model = args.yolo
    cfg.device = resolve_device(args.device)   # 'auto'/'cuda'/'cpu' -> concrete
    cfg.dt = args.dt
    cfg.save_path = args.save
    cfg.show = not args.no_show
    cfg.frame_stride = args.frame_stride
    cfg.px_per_metre = args.px_per_metre
    # Live sources (webcam index / RTSP) -> sample by wall-clock time.
    cfg.use_wallclock_sampling = str(args.source).isdigit() or "://" in str(args.source)
    return cfg


def main():
    cfg = parse_args()
    print(f"[stream] source={cfg.source} device={cfg.device} "
          f"live={cfg.use_wallclock_sampling}")

    pipe = CrowdPipeline(cfg)                            # shared core (video/pipeline.py)

    writer = None
    for annotated in pipe.frames():
        if cfg.save_path:
            if writer is None:
                h, w = annotated.shape[:2]
                writer = cv2.VideoWriter(cfg.save_path, fourcc_code("mp4v"), 25, (w, h))
            writer.write(annotated)

        if cfg.show:
            cv2.imshow("Crowd Trajectory Prediction", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if writer is not None:
        writer.release()
        print(f"[stream] saved annotated video -> {cfg.save_path}")
    cv2.destroyAllWindows()
    print(f"[stream] {pipe.stats['unique_people']} people tracked, "
          f"{pipe.stats['total_predictions']} predictions over "
          f"{pipe.stats['frames']} frames")


if __name__ == "__main__":
    sys.exit(main())

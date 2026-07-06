"""Configuration for the video → trajectory-prediction pipeline.

All tunable knobs live here so the streaming loop, tracker, and predictor stay
readable. Values are chosen to work out-of-the-box on a mid-range GPU with a
recorded file; the same settings work for a live camera (see `source`).
"""
from dataclasses import dataclass


def fourcc_code(codec: str = "mp4v") -> int:
    """FourCC integer for cv2.VideoWriter, valid on both OpenCV 4.x and 5.x.

    OpenCV 5.0 moved the helper to `cv2.VideoWriter.fourcc`; older versions expose
    `cv2.VideoWriter_fourcc`. We try the new location first and fall back.
    """
    import cv2
    fn = getattr(cv2.VideoWriter, "fourcc", None) or getattr(cv2, "VideoWriter_fourcc")
    return fn(*codec)


def resolve_device(pref: str = "auto") -> str:
    """Turn a device preference into a concrete 'cuda' or 'cpu'.

    'auto' -> 'cuda' when a GPU is available, else 'cpu'. An explicit 'cuda'
    request also degrades to 'cpu' (with a warning) if no GPU is present, so the
    pipeline never crashes just because CUDA is unavailable.
    """
    import torch
    if pref == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if pref == "cuda":
        print("[device] CUDA requested but unavailable -> falling back to CPU.")
    return "cpu"


@dataclass
class VideoConfig:
    # ---- input source (unified) --------------------------------------------
    # Accepts anything OpenCV's VideoCapture understands:
    #   "path/to/file.mp4"  -> recorded video (trial)
    #   "0"                  -> default webcam (live)
    #   "rtsp://user:pass@ip:554/stream" -> IP / CCTV camera (live)
    source: str = "0"

    # ---- detection / tracking ----------------------------------------------
    yolo_model: str = "yolo11n.pt"   # nano = fastest; use yolo11s/m for accuracy
    person_class_id: int = 0          # COCO class 0 == "person"
    conf_thresh: float = 0.3          # min detection confidence
    tracker_cfg: str = "bytetrack.yaml"  # Ultralytics built-in tracker config
    device: str = "auto"              # "auto" (GPU if available) | "cuda" | "cpu"

    # ---- trajectory sampling -----------------------------------------------
    # NSP-SFM expects points spaced `dt` seconds apart (0.4 s in the paper).
    dt: float = 0.4
    obs_len: int = 8                  # history points needed before predicting
    pred_len: int = 12                # future points produced

    # For recorded files we sample every N frames (N = round(dt * fps)).
    # For live cameras we sample by wall-clock time instead (frames may drop).
    use_wallclock_sampling: bool = False  # auto-set True for live sources

    # Rough pixels-per-metre scale for the pixel<->model-space conversion in the
    # predictor. Not a true calibration; see VIDEO_PIPELINE.md, "Calibration".
    px_per_metre: float = 40.0

    # ---- model checkpoint ---------------------------------------------------
    # Trained NSP-SFM weights. Any scene checkpoint works as a starting point;
    # see VIDEO_PIPELINE.md on calibration/fine-tuning for metric accuracy.
    checkpoint: str = "checkpoints/nsp_sfm_eth.pt"

    # ---- performance safety valves -----------------------------------------
    max_tracks: int = 60              # cap simultaneously predicted people
    frame_stride: int = 1             # process every Nth frame (raise to speed up)

    # ---- output -------------------------------------------------------------
    show: bool = True                 # live preview window (press 'q' to quit)
    save_path: str = ""               # if set, write annotated video here

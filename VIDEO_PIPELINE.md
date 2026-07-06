# Video Pipeline — Crowd Trajectory Prediction from Video

This document explains the **video → trajectory-prediction** system: how it turns
a crowd video (recorded file *or* live camera) into per-person future-path
predictions drawn back onto the frames. It is written for developers who will
run, extend, or deploy it.

> The core predictor is **NSP-SFM** (see `PROJECT_OVERVIEW.md`). That model works
> on **trajectories**, not pixels. This pipeline is the bridge: it extracts
> trajectories from video, feeds them to NSP-SFM, and visualises the result.

---

## 1. What it does (in one picture)

```
 --source (file.mp4 | 0 webcam | rtsp://cam)          SAME code for trial & live
        │
        ▼
 ┌──────────────┐   per frame   ┌──────────────────┐   every 0.4s   ┌───────────────┐
 │ PersonTracker│ ───────────▶  │ TrajectoryBuffers│ ────────────▶  │ TrajectoryPred│
 │ YOLO+ByteTrk │  {id: foot xy}│ rolling 8-pt hist│  {id: 8 pts}   │ NSP-SFM 8→12  │
 └──────────────┘               └──────────────────┘                └──────┬────────┘
        │                                                                    │ {id: 12 pts}
        │                          ┌──────────────┐                          │
        └────────── frame ───────▶ │ draw_overlays│ ◀────────────────────────┘
                                   │ green=past   │
                                   │ red=future   │──▶  live window and/or saved .mp4
                                   └──────────────┘
```

---

## 2. Module responsibilities

| File | Responsibility |
|---|---|
| `video/config.py` | All settings + `resolve_device()` (auto GPU/CPU). |
| `video/tracker.py` | `PersonTracker`: YOLO + ByteTrack → per-frame `{id: foot point}` with a seconds timestamp. |
| `video/trajectory_buffer.py` | `TrajectoryBuffers`: keep each person's last 8 points **sampled every `dt`=0.4 s**; forget people who leave. |
| `video/predictor.py` | `TrajectoryPredictor`: normalise pixels → run NSP-SFM → future path back in pixels. |
| `video/visualize.py` | `draw_overlays`: green observed path, red predicted path, ID labels. |
| `video/stream_predict.py` | Entry point: the real-time loop wiring everything together. |
| `video/make_sample_video.py` | Generates a synthetic clip for smoke-testing the plumbing. |

---

## 3. Why it works for BOTH recorded and live video

The single design rule: **one input abstraction, time in seconds.**

- `--source` is passed straight to OpenCV/Ultralytics, which accept a file path,
  a webcam index (`0`), or an RTSP/HTTP URL — no code change to switch.
- Every frame carries a **timestamp in seconds**. For a file it is
  `frame_index / fps`; for a live camera it is true wall-clock time (live feeds
  drop frames, so counting frames would distort the 0.4 s spacing).
- The buffer samples purely on that timestamp, so a person's 8-point history is
  always ~0.4 s apart whether the source is a 25 fps file or a stuttering camera.

Switching from your trial file to the live camera is literally:
```bash
# trial
python -m video.stream_predict --source my_clip.mp4
# live (webcam)
python -m video.stream_predict --source 0
# live (CCTV / IP camera)
python -m video.stream_predict --source "rtsp://user:pass@192.168.1.50:554/stream"
```

---

## 4. Setup

```bash
# 1. CUDA torch (must match torchvision!) — see the version note below
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
# 2. video deps
pip install -r requirements-video.txt
```

**⚠️ Version-pin gotcha (already hit once in this project):**
`pip install ultralytics` may pull a **CPU-only torch** that silently overwrites
your CUDA build, and a mismatched `torchvision` then fails with
`operator torchvision::nms does not exist`. If you see that, reinstall the
matching CUDA pair (`torch==2.11.0` ↔ `torchvision==0.26.0`) from the cu128 index.

YOLO weights (`yolo11n.pt`) download automatically on first run.

---

## 4b. Web interface (upload & watch in the browser)

Prefer a UI over the command line? A Flask app lets you upload a video and watch
the predictions stream live, with running statistics and a download button.

```bash
pip install flask          # or: pip install -r requirements-video.txt
python -m webapp.app        # then open http://127.0.0.1:5000
```

Flow: **choose/drag a video → "Run prediction" → live annotated stream + stats →
download the annotated .mp4**.

How it's built (all under `webapp/`):
- `app.py` — Flask server. Routes: `/` (upload page), `/upload` (save + make a
  job), `/view/<id>` (results page), `/stream/<id>` (MJPEG of annotated frames),
  `/status/<id>` (JSON stats, polled by the page), `/download/<id>`.
- `templates/index.html` — drag-and-drop upload page.
- `templates/view.html` — live `<img>` MJPEG viewer + auto-updating stats panel.
- `static/style.css` — styling.

**Why MJPEG?** Browsers can't reliably play OpenCV's raw mp4 inline, and
re-encoding needs ffmpeg. Streaming annotated JPEG frames into an `<img>` plays
everywhere *and* shows predictions as they are computed — the same mechanism will
later display a live camera feed. The pipeline itself is the shared
`video/pipeline.py:CrowdPipeline`, identical to the CLI.

Notes: it processes one job as the browser consumes the stream (Flask runs with
`threaded=True` so `/status` polling works concurrently). For a production/
multi-user deployment, put it behind a real WSGI server and a task queue.

---

## 5. How to run (command line)

```bash
# Recorded file, show live preview window (press q to quit)
python -m video.stream_predict --source path/to/crowd.mp4

# Recorded file, no window, save annotated result
python -m video.stream_predict --source path/to/crowd.mp4 --no-show --save out.mp4

# Live webcam
python -m video.stream_predict --source 0

# Bigger/more accurate detector, force CPU, faster (skip frames)
python -m video.stream_predict --source 0 --yolo yolo11s.pt --device cpu --frame-stride 2
```

Key flags: `--source`, `--checkpoint`, `--yolo`, `--device {auto,cuda,cpu}`,
`--dt`, `--px-per-metre`, `--save`, `--no-show`, `--frame-stride`.

---

## 6. How to test (no real video needed)

```bash
# a) plumbing: make a synthetic clip and run the whole pipeline headless
python -m video.make_sample_video
python -m video.stream_predict --source sample_data/synthetic_crowd.mp4 --no-show --save sample_data/annotated_out.mp4

# b) the video-specific logic: buffer cadence + predictor output (fast, deterministic)
python -m tests.test_video_predictor
```
Expected from (b):
```
[ok] buffer sampled 8 points ~dt apart ...
[ok] predictor continued the walk: last obs x=450 -> pred end x=745, y=447
```
(For a real detection demo, use an actual pedestrian video — the synthetic clip
draws abstract rectangles that YOLO won't reliably detect as people; it exists
only to exercise the file/buffer/draw/save path.)

---

## 7. Accuracy & calibration (important for real deployment)

The model was trained on **ETH/UCY in metres** (bird's-eye view). A camera gives
**pixels at an angle**, so out of the box the pipeline:

- **works and visualises predictions**, but
- distances are **not metric** — we use a rough `--px-per-metre` scale (default
  40) plus per-track centring so predictions look sensible immediately.

To make predictions genuinely accurate on *your* camera, do one of:

1. **Homography calibration (recommended).** Measure/click 4 ground points whose
   real-world spacing you know; compute a pixel↔ground-plane homography; feed the
   model ground-plane metres instead of raw pixels. This corrects perspective and
   units. (Hook: replace the normalise/denormalise in `predictor.py`.)
2. **Fine-tune NSP-SFM** on trajectories extracted from your own footage (retrain
   `train.py` on tracks exported from this pipeline).

Also note: by default each person is predicted independently in pixel space
(inter-agent repulsion `F_col` is off in the video adapter, because it needs a
common ground plane to be meaningful). Once homography is in place, re-enable
social forces by grouping co-present people into one `seq_start_end` scene in
`predictor.py`.

---

## 8. Performance notes

- `yolo11n` (nano) is the default for real-time; step up to `yolo11s`/`yolo11m`
  for accuracy at the cost of speed.
- NSP-SFM inference is sub-millisecond; the detector is the bottleneck.
- If a dense crowd can't keep up: raise `--frame-stride`, lower the YOLO model
  size, or cap `max_tracks` in `video/config.py`.
- GPU is used automatically when available (`--device auto`).

---

## 9. Known limitations

- Not metric without calibration (Section 7).
- Social interaction disabled in pixel space by default (Section 7).
- Very long occlusions can cause ByteTrack to assign a new ID (history restarts).
- The synthetic sample clip is for plumbing tests only, not detection accuracy.

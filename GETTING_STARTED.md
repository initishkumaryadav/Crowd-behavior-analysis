# Crowd Behaviour Analysis — Developer Guide

Clone this repo, follow these instructions, and you'll have the full system
running on your machine in under 10 minutes. No prior knowledge of the codebase
is required.

---

## What you need

- **Python 3.10 – 3.13** (tested on 3.13)
- **NVIDIA GPU** with CUDA (recommended) — the code runs on CPU too but ~10×
  slower on the video pipeline. Any GTX 1060+ or better works.
- **~3 GB disk** for dependencies + YOLO weights (auto-downloaded).
- **A crowd video** to test on (`.mp4`, `.avi`, `.mov`, etc.)

---

## 1. Clone

```bash
git clone https://github.com/<your-org>/<repo-name>.git
cd <repo-name>
```

---

## 2. Install dependencies

**Step A — PyTorch (CUDA build).**
This is the only step that can trip up. Use *exactly* these two packages together
so YOLO's `torchvision::nms` operator works:

```bash
# For NVIDIA GPUs (recommended — use the matching CUDA version for your system)
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128

# If you have no GPU at all, fall back to:
# pip install torch torchvision
```

Verify it worked:
```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
```
You should see `cuda True` if you installed the GPU build.

**Step B — everything else:**

```bash
pip install -r requirements.txt
pip install -r requirements-video.txt
```

---

## 3. Download the dataset

```bash
python -m data.download
```

This fetches the standard ETH/UCY pedestrian trajectory benchmark (~3 MB) into
`./datasets/`. It is only needed if you want to **train or evaluate** the model;
the video pipeline works without it.

---

## 4. Verify it works

Run three quick tests — no video file needed:

```bash
python -m tests.test_forces           # proves physics is correct (~1s)
python -m tests.test_overfit          # proves the model learns (~30s, needs GPU)
python -m tests.test_video_predictor  # proves the video path works (~5s)
```

All three should print `all ... tests passed` or similar. If `test_overfit`
fails on GPU, fall back to `--cpu` or check your CUDA install.

---

## 5. Run it

### A. Web interface (easiest)

```bash
python -m webapp.app
```

Open **http://127.0.0.1:5000** in your browser. Drag in a crowd video, click
**"Run prediction"**, and watch live. Download the annotated result when done.

### B. Command line — predict on a video

```bash
python -m video.stream_predict --source path/to/your_video.mp4
```

- Preview window opens; press `q` to quit.
- To save instead of preview:
  ```bash
  python -m video.stream_predict --source your_video.mp4 --no-show --save output.mp4
  ```

### C. Live camera

```bash
# Webcam (index 0)
python -m video.stream_predict --source 0

# IP / CCTV camera (RTSP)
python -m video.stream_predict --source "rtsp://user:pass@192.168.1.50:554/stream"
```

### D. Train the model on ETH/UCY

```bash
python train.py --scene eth --epochs 150
python eval.py  --scene eth
```

Scenes: `eth | hotel | univ | zara1 | zara2`.

---

## 6. Project layout

```
.
├── config.py                  Model hyperparameters (obs=8, pred=12, dt=0.4s)
├── utils.py                   Seed, device, neighbour masking
├── losses.py                  Loss function + ADE/FDE metrics
├── train.py                   Train on ETH/UCY (leave-one-out)
├── eval.py                    Evaluate a trained checkpoint
│
├── data/
│   ├── download.py            Fetch ETH/UCY dataset
│   └── dataset.py             Trajectory windowing + batching
│
├── models/
│   ├── forces.py              F_goal, F_col, F_env  (Eqs. 12–14)
│   ├── cvae.py                Goal + Alpha CVAEs    (Fig. 2)
│   └── nsp_sfm.py             Full model + integrator (Eqs. 9–11)
│
├── video/
│   ├── config.py              Video pipeline settings
│   ├── tracker.py             YOLO + ByteTrack wrapper
│   ├── trajectory_buffer.py   Per-person rolling history
│   ├── predictor.py           NSP-SFM adapter for pixel space
│   ├── visualize.py           Green/red path overlays
│   ├── pipeline.py            Reusable core (CrowdPipeline)
│   ├── stream_predict.py      CLI entry point
│   └── make_sample_video.py   Generates a synthetic test clip
│
├── webapp/
│   ├── app.py                 Flask server (upload + stream + stats)
│   ├── templates/             Upload page + results page
│   └── static/                Stylesheet
│
├── tests/
│   ├── test_forces.py         Physics gradient check
│   ├── test_overfit.py        One-batch learning check
│   └── test_video_predictor.py Video adapter check
│
├── sample_data/               Generated clips + test outputs
├── datasets/                  Downloaded ETH/UCY (auto-created)
├── checkpoints/               Trained model weights (auto-created)
│
├── requirements.txt           Core deps (torch, numpy, scipy, tqdm, etc.)
├── requirements-video.txt     Video pipeline deps (ultralytics, opencv, flask)
│
├── README.md                  Equation map + deviations
├── PROJECT_OVERVIEW.md        Plain-language guide for anyone
├── VIDEO_PIPELINE.md          Video + web pipeline docs
└── GETTING_STARTED.md         This file
```

---

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `operator torchvision::nms does not exist` | torch and torchvision versions mismatch. Reinstall the pair from step 2A exactly. |
| `torch.cuda.is_available() == False` | You installed the CPU-only build. Re-run the `pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128` command. |
| Video shows 0 detections | YOLO needs real people (not abstract shapes). Use a pedestrian video. |
| Web page is blank / 404 | Make sure the server started (`python -m webapp.app`) and you're on port 5000. |
| Eval score is high (ADE > 1.0) | Train longer (200+ epochs) or use an easier scene (`zara1`/`zara2` instead of `eth`). |

---

## 8. Key design decisions

- **Time-based sampling, not frame-based:** buffers sample by wall-clock seconds
  (0.4 s apart), so recorded files and live cameras use the same code path.
- **Device-agnostic:** every `.to(device)` call uses the same `get_device()`
  helper. `--cpu` flag forces CPU; auto mode picks GPU if available.
- **No ffmpeg needed:** the web app streams annotated frames as MJPEG (JPEG
  frames in an HTTP stream), which plays in every browser without extra deps.
- **Shared core:** `video/pipeline.py:CrowdPipeline` is used by both the CLI
  (`stream_predict.py`) and the web app (`webapp/app.py`) — one copy of the
  prediction logic, two interfaces.

---


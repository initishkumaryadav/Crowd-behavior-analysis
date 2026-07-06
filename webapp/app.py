"""Flask web interface for the crowd trajectory-prediction pipeline.

Upload a video in the browser, watch the annotated prediction stream live, see
running statistics, and download the annotated result — all on one page.

How the live view works
-----------------------
Browsers can't reliably play OpenCV's raw mp4 output inline, and re-encoding adds
a dependency. Instead we stream **MJPEG** (a sequence of JPEG frames over one HTTP
response) into an <img> tag — this plays in every browser and, as a bonus, shows
the prediction *as it is computed*. The same mechanism will later display a live
camera feed. A downloadable .mp4 is written alongside for offline viewing.

Run:
    python -m webapp.app
    # then open http://127.0.0.1:5000
"""
import os
import uuid
import threading

import cv2
from flask import (Flask, render_template, request, redirect, url_for,
                   Response, jsonify, send_file, abort)

from video.config import VideoConfig, resolve_device, fourcc_code
from video.pipeline import CrowdPipeline

BASE = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE, "uploads")
OUTPUT_DIR = os.path.join(BASE, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB uploads

# job_id -> {"source":..., "output":..., "pipe": CrowdPipeline|None, "started": bool}
JOBS = {}
_DEVICE = resolve_device("auto")


def _make_config(source: str) -> VideoConfig:
    cfg = VideoConfig()
    cfg.source = source
    cfg.device = _DEVICE
    # uploaded files are recorded -> frame-based sampling
    cfg.use_wallclock_sampling = False
    return cfg


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("video")
    if not file or file.filename == "":
        return redirect(url_for("index"))
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED:
        return f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED)}", 400

    job_id = uuid.uuid4().hex[:12]
    src = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")
    file.save(src)
    JOBS[job_id] = {
        "source": src,
        "output": os.path.join(OUTPUT_DIR, f"{job_id}.mp4"),
        "pipe": None,
        "started": False,
    }
    return redirect(url_for("view", job_id=job_id))


@app.route("/view/<job_id>")
def view(job_id):
    if job_id not in JOBS:
        abort(404)
    return render_template("view.html", job_id=job_id)


def _mjpeg(job_id):
    """Run the pipeline and yield annotated frames as an MJPEG stream."""
    job = JOBS[job_id]
    cfg = _make_config(job["source"])
    pipe = CrowdPipeline(cfg)
    job["pipe"] = pipe
    job["started"] = True

    writer = None
    for frame in pipe.frames():
        if writer is None:                       # lazily size the output writer
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(job["output"], fourcc_code("mp4v"), 25, (w, h))
        writer.write(frame)

        ok, jpg = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")

    if writer is not None:
        writer.release()


@app.route("/stream/<job_id>")
def stream(job_id):
    if job_id not in JOBS:
        abort(404)
    job = JOBS[job_id]
    if job["started"]:            # avoid double-processing on refresh
        return Response(status=204)
    return Response(_mjpeg(job_id),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status/<job_id>")
def status(job_id):
    if job_id not in JOBS:
        abort(404)
    pipe = JOBS[job_id]["pipe"]
    if pipe is None:
        return jsonify({"frames": 0, "done": False, "started": False})
    s = dict(pipe.stats)
    s["started"] = True
    return jsonify(s)


@app.route("/download/<job_id>")
def download(job_id):
    if job_id not in JOBS:
        abort(404)
    out = JOBS[job_id]["output"]
    if not os.path.exists(out):
        return "Annotated video not ready yet.", 425
    return send_file(out, as_attachment=True, download_name=f"annotated_{job_id}.mp4")


if __name__ == "__main__":
    print(f"[webapp] device={_DEVICE}  ->  open http://127.0.0.1:5000")
    # threaded=True so the MJPEG stream and the /status polling run concurrently.
    app.run(host="127.0.0.1", port=5000, threaded=True, debug=False)

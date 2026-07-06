"""video: turn a crowd video (recorded file or live camera) into NSP-SFM
trajectory predictions, drawn back onto the frames.

Modules
-------
- config.py            : all tunable settings (source, model, dt, output).
- tracker.py           : YOLO + ByteTrack wrapper -> per-frame {id: foot_point}.
- trajectory_buffer.py : rolling per-person history sampled every dt seconds.
- predictor.py         : wraps a trained NSP-SFM model for single-window inference.
- visualize.py         : draw observed (green) and predicted (red) paths.
- stream_predict.py    : the real-time loop tying it all together (entry point).

The same code path serves a recorded file (trial) and a live camera; only the
`--source` argument changes. See VIDEO_PIPELINE.md for the full walkthrough.
"""

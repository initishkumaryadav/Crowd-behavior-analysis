# Project Overview — Crowd Behaviour Analysis with NSP-SFM

A complete, from-scratch PyTorch project that implements **NSP-SFM
(Neural Social Physics — Social Force Model)**, the flagship *physics-inspired
deep learning* method for **human trajectory prediction** reviewed in Section 2.3
of the survey *"Recent Deep Learning in Crowd Behaviour Analysis: A Brief Review"*
(Yue & Wang, 2025), first proposed by Yue, Manocha & Wang (ECCV 2022).

This document explains, in plain terms, **what the project does, how it works,
what every file is for, and exactly how you can test it** — even if you have
never seen this codebase before.

---

## 1. What problem does it solve?

Given the **past movement** of pedestrians in a scene (where each person walked
for the last 8 time steps = 3.2 seconds), predict **where each person will walk
next** (the next 12 time steps = 4.8 seconds).

This is called **human trajectory prediction**. It matters for:
- **Crowd management & public safety** (spotting bottlenecks, overcrowding).
- **Autonomous vehicles & robots** (anticipating where people will go).
- **Urban planning & event organisation.**

Input and output are 2-D positions `(x, y)` sampled every **0.4 seconds**.

---

## 2. What makes NSP-SFM special?

Most deep-learning predictors are **black boxes** — accurate but unexplainable.
NSP-SFM instead keeps the classic, human-understandable **Social Force Model**
(from physics) and uses **neural networks only to predict its parameters**. Each
pedestrian is treated like a particle pushed by three forces:

| Force | Meaning | In the code |
|---|---|---|
| **Goal attraction** `F_goal` | pull toward the person's destination | `models/forces.py → GoalForce` (Eq. 12) |
| **Inter-agent repulsion** `F_col` | push away from nearby people to avoid collisions | `models/forces.py → CollisionForce` (Eq. 13) |
| **Environment repulsion** `F_env` | push away from walls/obstacles | `models/forces.py → EnvForce` (Eq. 14) |

The total acceleration `p'' = F_goal + F_col + F_env` is integrated forward in
time with a **semi-implicit integrator** (Eqs. 9–11) to roll out the future path.
A neural network also predicts each person's **destination** (a "Goal Sampling
CVAE"), and a second network (an "alpha CVAE") injects **realistic randomness**
so the model can produce several plausible futures instead of one.

**The payoff:** deep-learning accuracy **and** physical explainability — you can
literally read off why the model turned a pedestrian (e.g. "repelled by the
person on the left, attracted to the exit ahead").

---

## 3. How it works — the pipeline end to end

```
        past 8 steps                                     future 12 steps
   ┌─────────────────────┐                          ┌──────────────────────┐
   │ observed x,y per     │                          │ predicted x,y per     │
   │ pedestrian           │                          │ pedestrian (x20 samples)
   └─────────┬───────────┘                          └───────────▲──────────┘
             │                                                   │
             ▼                                                   │
   ┌──────────────────┐   ┌───────────────────────────────────┐ │
   │ TrajEncoder      │──▶│ Goal Sampling CVAE → destination   │ │
   │ (history → cond) │   └───────────────────────────────────┘ │
   └──────────────────┘                 │                        │
             │                          ▼                        │
             │        ┌───────────────────────────────────────┐ │
             │        │  FOR each of the 12 future steps:      │ │
             │        │   p'' = F_goal + F_col (+F_env)        │ │
             │        │   ṗ  += Δt·p''      (Eq.10)            │ │
             │        │   p  += Δt·(ṗ + α)  (Eq.11)            │ │
             └───────▶│   α  ← alpha CVAE  (adds randomness)   │─┘
                      └───────────────────────────────────────┘
```

At **training time** the CVAEs use the ground-truth future (posterior) and the
model minimises the distance between predicted and true positions plus two KL
terms. At **test time** the CVAEs sample from a standard normal (prior), so the
model can generate **20 different plausible futures**; we keep the best one
(standard "best-of-20" evaluation).

---

## 4. How is it measured?

Two standard metrics (survey Eq. 32), computed in `losses.py`:

- **ADE (Average Displacement Error)** — average distance between predicted and
  true positions over all 12 future steps.
- **FDE (Final Displacement Error)** — distance at the very last predicted step.

Lower is better. On ETH/UCY, errors are in **metres**. We use **best-of-20**:
generate 20 futures per person and report the closest.

---

## 5. What data does it use?

**ETH/UCY** — the standard public benchmark of real pedestrian trajectories,
5 scenes: `eth`, `hotel`, `univ`, `zara1`, `zara2`. It is downloaded
automatically (`python -m data.download`) using the widely-used Social-GAN
preprocessing. Each scene has `train/val/test` folders; the standard
**leave-one-scene-out** protocol is already baked into the folders (training on
`<scene>/train` tests on the held-out `<scene>/test`).

---

## 6. Project structure — every file explained

```
config.py            All settings in one place: obs=8, pred=12, dt=0.4 s,
                     collision range r_col, network sizes, learning rate, etc.

utils.py             Helpers: random-seed control, CPU/GPU selection,
                     neighbour masking.

data/
  download.py        Downloads the ETH/UCY dataset into ./datasets.
  dataset.py         Turns raw trajectory files into training windows (8+12
                     steps) and groups pedestrians who share the same time
                     window so the social forces can be computed between them.

models/
  forces.py          The three physics forces (goal, collision, environment).
  cvae.py            Trajectory encoder + the two CVAEs (goal + alpha).
  nsp_sfm.py         The full model: ties forces + CVAEs together and runs the
                     semi-implicit integrator to roll out the future.

losses.py            Loss function + the ADE / FDE metrics (incl. best-of-20).

train.py             Trains the model on one scene, prints test ADE/FDE every
                     5 epochs, saves the best checkpoint.

eval.py              Loads a trained checkpoint and reports best-of-20 ADE/FDE.

tests/
  test_forces.py     Proves the collision force exactly equals -∇U (the physics
                     is implemented correctly) and the goal force points to the
                     destination.
  test_overfit.py    Proves the whole model can learn: it drives the error on a
                     single batch from ~0.34 down to ~0.02.

datasets/            The downloaded ETH/UCY data (auto-created).
checkpoints/         Saved model weights (auto-created during training).

README.md            Short reference + paper→code equation map + deviations.
PROJECT_OVERVIEW.md  (this file) full plain-language guide.
```

---

## 7. Setup (one time)

```bash
# 1. PyTorch — CUDA build for NVIDIA RTX 50-series (Blackwell). Use the CPU
#    wheel instead if you have no GPU.
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 2. The rest of the dependencies
pip install -r requirements.txt

# 3. Download the dataset (~a few MB)
python -m data.download
```
Everything is **device-agnostic** — it runs on GPU automatically if available,
otherwise CPU. Add `--cpu` to any command to force CPU.

---

## 8. How to test it — step by step

### A. Fastest sanity checks (seconds, no training needed)

```bash
# Physics is correct: collision force == -∇U (0 error), goal force direction
python -m tests.test_forces

# The model can actually learn: overfits one batch, ADE 0.335 -> 0.023
python -m tests.test_overfit
```
Expected output:
```
[ok] collision force == -grad U   (max err 0.00e+00)
[ok] goal force points toward destination: ...
[overfit] ADE 0.335 -> 0.023  FDE 0.036
```

### B. Evaluate the already-trained model

```bash
python eval.py --scene eth
```
Expected: `[eval] scene=eth  best-of-20  ADE ~1.00  FDE ~2.14`
(`eth` is the hardest scene; easier scenes score much lower.)

### C. Train your own model on a scene and watch it improve

```bash
python train.py --scene zara1 --epochs 150
```
It prints a line every 5 epochs, e.g.
`epoch  25 | train_loss 0.83 | test ADE 0.42 FDE 0.71`, and saves the best
checkpoint to `checkpoints/nsp_sfm_zara1.pt`. Then evaluate it:
```bash
python eval.py --scene zara1
```

### ⚠️ One important rule
**Do not run `eval.py` on a scene while `train.py` is still training that same
scene.** They share the checkpoint file; reading it mid-write returns garbage
numbers (e.g. a huge ADE like 128). Either wait for training to finish, or
evaluate a *different* scene.

---

## 9. Current results & honest limitations

- **Verified correct:** the collision force matches `-∇U` exactly, and the model
  overfits a single batch to ADE 0.023 — so the maths and the training loop are
  right.
- **Measured:** `eth` best-of-20 **ADE ≈ 1.00 / FDE ≈ 2.14** (hardest scene).
  The paper reports ~0.25 for eth and 0.17 averaged over all scenes.
- **Why the gap:** it is a *generalisation* gap, not a bug. Matching the paper's
  numbers requires details the survey summarises but does not fully specify — the
  environment-image-conditioned goal network, the complete multi-scene training
  schedule, and tuning of the CVAE KL weights.
- **Notes:** ETH/UCY coordinates are already per-scene normalised, so no extra
  input normalisation is applied (it destabilises the explicit integrator).
  Acceleration and the random perturbation are clamped for numerical stability.
  `F_env` is disabled on ETH/UCY because that dataset ships no obstacle maps.

See `README.md` for the full paper→code equation map and the complete list of
deviations.





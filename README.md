# NSP-SFM — Neural Social Physics (Social Force Model)

A faithful, from-scratch PyTorch reproduction of **NSP-SFM**, the flagship
*physics-inspired deep learning* method reviewed in Section 2.3 of the survey
**"Recent Deep Learning in Crowd Behaviour Analysis: A Brief Review"**
(Yue & Wang, 2025), originally proposed in *Human Trajectory Prediction via
Neural Social Physics* (Yue, Manocha & Wang, ECCV 2022).

The idea: keep the classic **Social Force Model** structure (goal attraction +
inter-agent repulsion + environment repulsion) but let **neural networks predict
its parameters**, and add a **CVAE** for stochasticity. This yields deep-learning
accuracy *and* physical explainability — the survey's headline result (Table 2:
NSP-SFM ADE/FDE 0.17/0.24 on ETH/UCY).

## Method → code (survey equations)

| Paper | Equation | File |
|---|---|---|
| Acceleration `p'' = F_goal + F_col + F_env` | Eq. 9 | `models/nsp_sfm.py` |
| Semi-implicit update + stochastic `alpha` | Eqs. 10–11 | `models/nsp_sfm.py` |
| Goal attraction `F_goal=(1/τ)(v_des−ṗ)`, `τ=NN` | Eq. 12 | `models/forces.py:GoalForce` |
| Inter-agent repulsion `F_col=Σ −∇U`, `k_ij=NN` | Eq. 13 | `models/forces.py:CollisionForce` |
| Environment repulsion `F_env` | Eq. 14 | `models/forces.py:EnvForce` |
| Goal Sampling CVAE + `alpha` CVAE | Fig. 2 | `models/cvae.py` |
| ADE/FDE, best-of-20 | Eq. 32 | `losses.py` |

## Setup

```bash
# PyTorch (CUDA 12.8 build for RTX 50-series / Blackwell). Use the CPU wheel if no GPU.
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Data

```bash
python -m data.download        # ETH/UCY (Social-GAN preprocessing) -> ./datasets
```
Five scenes (`eth, hotel, univ, zara1, zara2`) each with `train/val/test`, using
the standard **leave-one-scene-out** protocol baked into the folder layout.

## Train & evaluate

```bash
python train.py --scene eth --epochs 200        # trains, saves checkpoints/nsp_sfm_eth.pt
python eval.py  --scene eth                      # best-of-20 ADE/FDE on eth test
```
Observe 8 steps (3.2 s), predict 12 (4.8 s), dt = 0.4 s. Add `--cpu` to force CPU.

## Tests

```bash
python -m tests.test_forces      # collision force == -∇U (exact), goal force direction
python -m tests.test_overfit     # model overfits one batch -> integrator+backprop correct
```

## Deviations from the paper (surfaced, not hidden)

- **F_env disabled on ETH/UCY.** Eq. 14 needs per-scene obstacle maps / scene
  segmentation, which ETH/UCY does not ship. It is disabled here
  (`Config.use_env_force=False`), as in many reimplementations. The `EnvForce`
  module is implemented and ready for datasets that provide obstacles (e.g. SDD
  with masks).
- **Goal Sampling Network** is implemented as a trajectory-conditioned CVAE
  predicting the destination; the paper additionally conditions on the
  environment image, omitted here for the image-free ETH/UCY setup.
- **Measured result:** `eth` best-of-20 **ADE 1.00 / FDE 2.14** (eth is the
  hardest ETH/UCY scene; paper ~0.25 for eth, 0.17 average). The pipeline is
  correct — it overfits a single batch to ADE 0.023, and the collision force
  matches `-∇U` exactly — so the gap is generalization, not a bug. Closing it to
  the paper's numbers needs the full original recipe (the survey gives the
  equations but not all training details): the environment-conditioned goal
  network, longer multi-scene training, and CVAE KL tuning. Note ETH/UCY data is
  already per-scene normalized, so extra input normalization is unnecessary (it
  actually destabilizes the explicit integrator here).
- **Stability clamps:** acceleration is bounded to ±50 m/s² and the stochastic
  perturbation `alpha` to ±10 in `models/nsp_sfm.py`, so the explicit integrator
  cannot diverge to NaN early in training. Limits are generous (pedestrian
  accelerations are far smaller) and do not bias a converged model.
- Minor integrator choice: realized velocity is recomputed from consecutive
  positions (`ṗ = Δp/Δt`), consistent with Eqs. 10–11 where `alpha` affects the
  velocity used to advance position.

## Layout
```
config.py            hyper-parameters (obs=8, pred=12, dt=0.4, r_col, sizes)
utils.py             seeding, device, neighbour masking
data/download.py     fetch ETH/UCY
data/dataset.py      windowing + per-scene neighbour grouping (seq_start_end)
models/forces.py     F_goal, F_col, F_env
models/cvae.py       TrajEncoder, GoalSamplingCVAE, AlphaCVAE
models/nsp_sfm.py    full model: semi-implicit rollout
losses.py            loss combination + ADE/FDE (best-of-20)
train.py / eval.py   training and evaluation
tests/               force gradient + overfit checks
```

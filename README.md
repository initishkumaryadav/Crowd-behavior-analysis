

To run on your Local Machine read Getting_started.md file.
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

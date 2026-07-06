"""Central configuration for the NSP-SFM reproduction.

Values follow the standard trajectory-prediction protocol used by the survey
(Yue & Wang 2025, Sec. 2.3) and the original Neural Social Physics paper
(Yue et al., ECCV 2022): observe 8 steps, predict 12, dt = 0.4 s.
"""
from dataclasses import dataclass, field


@dataclass
class Config:
    # ---- trajectory protocol (paper Sec. 2.4 / Eq. 32) ----
    obs_len: int = 8            # t_p : observed history length
    pred_len: int = 12          # t_f : future length to predict
    dt: float = 0.4             # seconds per timestep

    # ---- Social Force Model hyper-parameters (Eqs. 12-14) ----
    r_col: float = 0.4          # collision potential range (metres), hyperparam in Eq. 13
    neighbour_thresh: float = 4.0  # distance (m) defining the neighbour set Omega_i
    use_env_force: bool = False # F_env needs obstacle maps; off for ETH/UCY (see README)

    # ---- network sizes ----
    hidden: int = 128           # MLP hidden width for force-parameter networks
    goal_latent: int = 32       # CVAE latent dim for the Goal Sampling Network
    alpha_latent: int = 16      # CVAE latent dim for the stochastic-perturbation net

    # ---- training ----
    lr: float = 1e-3
    epochs: int = 200
    batch_size: int = 64
    kl_weight_goal: float = 1.0
    kl_weight_alpha: float = 1.0
    grad_clip: float = 1.0
    seed: int = 42

    # ---- evaluation ----
    num_samples: int = 20       # best-of-20 stochastic evaluation (Table 2)

    # ---- data ----
    data_root: str = "datasets"
    dataset: str = "eth"        # one of the 5 ETH/UCY leave-one-out scenes
    # ETH/UCY scenes for leave-one-out cross validation
    ethucy_scenes: tuple = ("eth", "hotel", "univ", "zara1", "zara2")


CFG = Config()

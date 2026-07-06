"""ETH/UCY trajectory dataset in the standard Social-GAN windowing.

A "window" (scene) is `obs_len + pred_len` consecutive frames. We keep every
pedestrian present in *all* frames of the window, so the whole window shares a
common time axis — which is exactly what the Social Force Model needs to compute
inter-agent repulsion between co-present neighbours.

Because the number of pedestrians varies per window, we batch several windows by
concatenating them along the pedestrian axis and remembering the split points in
`seq_start_end` (same trick as Social-GAN). Force computation then runs per
window using those slices.
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


def _read_file(path: str) -> np.ndarray:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            rows.append([float(p) for p in parts[:4]])
    return np.asarray(rows)  # (M, 4): frame, ped, x, y


class TrajectoryDataset(Dataset):
    def __init__(self, data_dir: str, obs_len: int = 8, pred_len: int = 12, min_ped: int = 1):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len

        files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
        if not files:
            raise FileNotFoundError(f"No .txt trajectory files in {data_dir}")

        seq_list = []           # each: (num_peds, seq_len, 2) absolute positions
        for path in files:
            data = _read_file(path)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = [data[data[:, 0] == fr, :] for fr in frames]
            n_windows = len(frames) - self.seq_len + 1
            for idx in range(0, max(0, n_windows)):
                window = np.concatenate(frame_data[idx: idx + self.seq_len], axis=0)
                peds = np.unique(window[:, 1])
                window_peds = []
                for ped in peds:
                    ptraj = window[window[:, 1] == ped, :]
                    if len(ptraj) != self.seq_len:      # must be present in every frame
                        continue
                    # sort by frame just in case, take (x, y)
                    ptraj = ptraj[ptraj[:, 0].argsort()]
                    window_peds.append(ptraj[:, 2:4])
                if len(window_peds) >= min_ped:
                    seq_list.append(np.stack(window_peds, axis=0))  # (P, seq_len, 2)

        if not seq_list:
            raise RuntimeError(f"No valid windows found in {data_dir}")

        self.seqs = seq_list
        num_peds = [s.shape[0] for s in seq_list]
        cum = np.cumsum([0] + num_peds)
        # global tensor of all peds, plus per-window slices
        allseq = np.concatenate(seq_list, axis=0).astype(np.float32)  # (sumP, seq_len, 2)
        self.traj = torch.from_numpy(allseq)                          # absolute positions
        self.seq_start_end = [(cum[i], cum[i + 1]) for i in range(len(seq_list))]

    def __len__(self):
        return len(self.seq_start_end)

    def __getitem__(self, index):
        s, e = self.seq_start_end[index]
        full = self.traj[s:e]                       # (P, seq_len, 2)
        obs = full[:, : self.obs_len]               # (P, obs_len, 2)
        pred = full[:, self.obs_len:]               # (P, pred_len, 2)
        return obs, pred


def collate(batch):
    """Concatenate windows along the pedestrian axis; keep split points."""
    obs_list, pred_list, sse = [], [], []
    start = 0
    for obs, pred in batch:
        p = obs.shape[0]
        obs_list.append(obs)
        pred_list.append(pred)
        sse.append((start, start + p))
        start += p
    obs = torch.cat(obs_list, 0)                    # (N, obs_len, 2)
    pred = torch.cat(pred_list, 0)                  # (N, pred_len, 2)
    return obs, pred, sse


def make_loader(scene_dir, obs_len, pred_len, batch_size, shuffle):
    from torch.utils.data import DataLoader
    ds = TrajectoryDataset(scene_dir, obs_len, pred_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)

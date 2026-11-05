"""Memory-mapped clip datasets with one layout for VAS and FlashDot.

A dataset directory holds, per split (train / dev / test):
    <split>_mel.npy     float16 (N, 80, 848)    ground-truth log-mel in [0, 1]
    <split>_feats.npy   float16 (N, 212, 2048)  visual features
    <split>_labels.npy  int64 (N,)
    <split>_tokens.npy  int16 (N, 265)          VQGAN codes of the ground truth (encode.py)
    <split>_recon.npy   float16 (N, 80, 848)    VQGAN reconstruction in [0, 1] (encode.py)
FlashDot additionally has <split>_act.npy (uint8 (N, 848)): ground-truth tone activity.
Everything is read lazily, so a split never has to fit in RAM.
"""

import json
from pathlib import Path

import numpy as np
import torch


class ClipDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, keys=('feats', 'tokens'), limit=None):
        self.root, self.split = Path(root), split
        self.arrays = {k: np.load(self.root / f'{split}_{k}.npy', mmap_mode='r') for k in keys}
        self.n = len(next(iter(self.arrays.values())))
        if limit is not None:
            self.n = min(self.n, limit)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        out = {'index': i}
        for k, a in self.arrays.items():
            x = np.asarray(a[i])
            if k == 'feats':
                x = x.T  # (2048, 212)
            out[k] = torch.from_numpy(x.astype(np.int64 if k in ('tokens', 'labels') else np.float32))
        return out


def standardise_flashdot(src='work/flashdot', out='work/flashdot_std'):
    """Crop FlashDot to the SpecVQGAN geometry (mel 6:854, video 1:213) and split it."""
    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    splits = json.loads((src / 'splits.json').read_text())
    mel = np.load(src / 'mel.npy', mmap_mode='r')
    feats = np.load(src / 'rgb_resnet50.npy', mmap_mode='r')
    act = np.load(src / 'activity.npz')['mel_activity']
    for split, idx in splits.items():
        idx = np.asarray(idx)
        m = np.lib.format.open_memmap(out / f'{split}_mel.npy', 'w+', np.float16, (len(idx), 80, 848))
        f = np.lib.format.open_memmap(out / f'{split}_feats.npy', 'w+', np.float16, (len(idx), 212, 2048))
        for n, i in enumerate(idx):
            m[n] = mel[i][:, 6:854]
            f[n] = feats[i][1:213]
        m.flush(); f.flush()
        np.save(out / f'{split}_act.npy', act[idx][:, 6:854].astype(np.uint8))
        np.save(out / f'{split}_labels.npy', np.zeros(len(idx), np.int64))
        np.save(out / f'{split}_src_index.npy', idx)
        print(split, len(idx))


if __name__ == '__main__':
    standardise_flashdot()

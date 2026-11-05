"""Encode ground-truth spectrograms to VQGAN codes and store their reconstructions.

Usage: python -m perceptalign.encode --data work/vas --ckpt <vas codebook .ckpt>
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from perceptalign.models import VQGAN, load_state, torch_load


def load_vqgan(ckpt, device):
    sd = torch_load(ckpt)
    sd = sd.get('state_dict', sd)
    prefix = 'first_stage_model.' if any(k.startswith('first_stage_model.') for k in sd) else ''
    v = VQGAN()
    missing, _ = load_state(v, sd, prefix)
    missing = [k for k in missing if not k.startswith('loss')]
    assert not missing, missing[:10]
    return v.to(device).eval()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--splits', default='train,dev,test')
    ap.add_argument('--bs', type=int, default=8)
    args = ap.parse_args()
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    v = load_vqgan(args.ckpt, dev)
    root = Path(args.data)
    for split in args.splits.split(','):
        mel = np.load(root / f'{split}_mel.npy', mmap_mode='r')
        N = len(mel)
        tok = np.zeros((N, 265), np.int16)
        rec = np.lib.format.open_memmap(root / f'{split}_recon.npy', 'w+', np.float16, (N, 80, 848))
        err = []
        for s in range(0, N, args.bs):
            x = torch.from_numpy(np.asarray(mel[s:s + args.bs], np.float32)).to(dev)
            idx = v.encode_to_tokens(2 * x[:, None] - 1)
            r = ((v.decode_tokens(idx)[:, 0] + 1) / 2).clamp(0, 1)
            tok[s:s + len(idx)] = idx.cpu().numpy()
            rec[s:s + len(idx)] = r.cpu().numpy().astype(np.float16)
            err.append((r - x).abs().mean().item())
            if s % (50 * args.bs) == 0:
                print(split, s, N, f'recon L1 {np.mean(err):.4f}', flush=True)
        rec.flush()
        np.save(root / f'{split}_tokens.npy', tok)
        print(split, 'done, recon L1', float(np.mean(err)), 'codes used', len(np.unique(tok)))


if __name__ == '__main__':
    os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.3')
    os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO', '0.25')
    main()

"""Wrong-video cross-entropy probe: how much does a trained generator use the video?

Development cross-entropy of each run with each clip's own visual features and with another clip's
(features rolled by one clip), under teacher forcing (p_keep = 1) and with a fully corrupted prefix
(p_keep = 0, every input token replaced by a random code). The uniform level is log(128).

  python -m perceptalign.probe_wrong_video --runs work/runs/fd2_ce work/runs/fd2_ce_s1 work/runs/fd2_ce_s2 \
      --data work/flashdot_std --out work/results/probe_wrong_video.json
"""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from perceptalign.data import ClipDataset
from perceptalign.models import GPTFeats, torch_load
from perceptalign.train_critic import batches
from perceptalign.train_gen import corrupt


@torch.no_grad()
def probe(gpt, ds, dev, bs, seed):
    out = {}
    for pkeep in (1.0, 0.0):
        for wrong in (False, True):
            torch.manual_seed(seed)
            tot, n = 0.0, 0
            for b in batches(ds, bs, False, 0):
                tok, feats = b['tokens'].to(dev), b['feats'].to(dev)
                if wrong:
                    feats = feats.roll(1, dims=0)
                lg = gpt(corrupt(tok, pkeep), feats)
                tot += F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), tok.reshape(-1), reduction='sum').item()
                n += tok.numel()
            out[f'pkeep{pkeep:g}_{"wrong" if wrong else "own"}'] = tot / n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--split', default='dev')
    ap.add_argument('--bs', type=int, default=8)  # rolling within a batch: use bs > 1
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    ds = ClipDataset(args.data, args.split, ('feats', 'tokens'))
    res = dict(uniform=math.log(128), n_clips=len(ds), split=args.split, runs={})
    for run in args.runs:
        ck = torch_load(Path(run) / 'last.pt')
        a = ck['args']
        gpt = GPTFeats(n_layer=a['n_layer'], n_head=a['n_head'], n_embd=a['n_embd'])
        gpt.load_state_dict(ck['state_dict'])
        gpt = gpt.to(dev).eval()
        res['runs'][run] = probe(gpt, ds, dev, args.bs, args.seed)
        print(run, json.dumps({k: round(v, 3) for k, v in res['runs'][run].items()}), flush=True)
        del gpt
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

"""Stage 1: fit the PAL sync critic on real audio-video pairs.

The audio input is the VQGAN reconstruction of the ground truth (what the decoder produces
at generation time) or, with probability --p_gt, the ground-truth spectrogram itself.
Negatives are the other temporal offsets of the same clip.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from perceptalign.data import ClipDataset
from perceptalign.pal import build_critic, offset_accuracy, pal_loss


def batches(ds, bs, shuffle, seed):
    order = np.random.default_rng(seed).permutation(len(ds)) if shuffle else np.arange(len(ds))
    for s in range(0, len(ds), bs):
        items = [ds[int(i)] for i in order[s:s + bs]]
        yield {k: torch.stack([it[k] for it in items]) for k in items[0] if k != 'index'}


@torch.no_grad()
def evaluate(critic, ds, args, dev):
    critic.eval()
    losses, acc0, acc1 = [], [], []
    for b in batches(ds, 32, False, 0):
        a = critic.embed_audio(2 * b['recon'].to(dev) - 1)
        v = critic.embed_video(b['feats'].to(dev))
        l, p = pal_loss(a, v, args.K, args.segments, args.tau, args.sigma, args.reduce)
        losses.append(l.mean().item()); acc0.append(offset_accuracy(p, 0)); acc1.append(offset_accuracy(p, 1))
    critic.train()
    return dict(loss=float(np.mean(losses)), acc_tol0=float(np.mean(acc0)), acc_tol1=float(np.mean(acc1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--K', type=int, default=10)
    ap.add_argument('--segments', type=int, default=8)
    ap.add_argument('--tau', type=float, default=0.1)
    ap.add_argument('--sigma', type=float, default=0.0)
    ap.add_argument('--reduce', default='log_then_mean')
    ap.add_argument('--kernel', type=int, default=5)
    ap.add_argument('--n_blocks', type=int, default=2)
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--audio_repr', default='mel', choices=['mel', 'bands'])
    ap.add_argument('--n_bands', type=int, default=8)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--p_gt', type=float, default=0.3)
    ap.add_argument('--train_limit', type=int, default=None)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'args.json').write_text(json.dumps(vars(args), indent=1))

    tr = ClipDataset(args.data, 'train', ('feats', 'recon', 'mel'), limit=args.train_limit)
    dv = ClipDataset(args.data, 'dev', ('feats', 'recon'))
    critic = build_critic(vars(args)).to(dev)
    opt = torch.optim.AdamW(critic.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    rng = np.random.default_rng(args.seed)
    step, best, log, t0 = 0, 1e9, [], time.time()
    while step < args.steps:
        for b in batches(tr, args.bs, True, args.seed + step):
            use_gt = torch.from_numpy(rng.random(len(b['mel'])) < args.p_gt)[:, None, None]
            mel = torch.where(use_gt, b['mel'], b['recon']).to(dev)
            l, p = pal_loss(critic.embed_audio(2 * mel - 1), critic.embed_video(b['feats'].to(dev)),
                            args.K, args.segments, args.tau, args.sigma, args.reduce)
            opt.zero_grad(); l.mean().backward(); opt.step(); sched.step()
            step += 1
            if step % 250 == 0 or step == args.steps:
                ev = evaluate(critic, dv, args, dev)
                ev.update(step=step, train_loss=l.mean().item(), sec=round(time.time() - t0))
                log.append(ev); print(json.dumps(ev), flush=True)
                if ev['loss'] < best:
                    best = ev['loss']
                    torch.save({'state_dict': critic.state_dict(), 'args': vars(args), 'dev': ev}, out / 'critic.pt')
            if step >= args.steps:
                break
    (out / 'log.json').write_text(json.dumps(log, indent=1))


if __name__ == '__main__':
    os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.3')
    os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO', '0.25')
    main()

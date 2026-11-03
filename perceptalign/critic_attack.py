"""Exploitability test for a sync critic.

Free logits (no generator, no cross-entropy) are optimised to minimise PAL through the same
straight-through decoding used in training. If the critic only rewards real synchrony, the
PAL-optimal spectrograms should be aligned with the video; if it can be fooled, PAL drops to
the ground-truth level while the audio stays unaligned. Reports the PAL reached, the PAL of the
ground-truth codes, and timing metrics of the decoded optimum (tone-activity F1 on FlashDot,
envelope lag and onset F1 everywhere) next to random logits and the ground-truth codes.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from perceptalign import timing_metrics as tm
from perceptalign.encode import load_vqgan
from perceptalign.models import torch_load
from perceptalign.pal import build_critic
from perceptalign.sample_eval import calibrate_flashdot, flashdot_activity
from perceptalign.train_gen import pal_term


def decode(vq, logits):
    with torch.no_grad():
        m = torch.cat([vq.decode_tokens(logits.argmax(-1)[j:j + 1]) for j in range(len(logits))])[:, 0]
    return ((m + 1) / 2).clamp(0, 1).cpu().numpy()


def score(mels, gt, act, band_thr):
    rows = [tm.clip_scores(mels[i], gt[i]) for i in range(len(mels))]
    out = {k: v for k, v in tm.aggregate(rows).items() if k in ('onset_f1', 'env_corr', 'abs_lag_ms')}
    if act is not None:
        out['act_f1'] = float(np.mean([tm.activity_scores(flashdot_activity(mels[i], *band_thr), act[i])['act_f1']
                                       for i in range(len(mels))]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--critic', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--codebook', required=True)
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    dev = 'mps'
    torch.manual_seed(0)
    vq = load_vqgan(a.codebook, dev)
    del vq.encoder
    ck = torch_load(a.critic)
    c = build_critic(ck['args']); c.load_state_dict(ck['state_dict']); c = c.to(dev).eval()
    for p in list(vq.parameters()) + list(c.parameters()):
        p.requires_grad = False
    root = Path(a.data)
    feats = torch.from_numpy(np.load(root / 'dev_feats.npy', mmap_mode='r')[:a.n].astype(np.float32)).transpose(1, 2).to(dev)
    tok = torch.from_numpy(np.load(root / 'dev_tokens.npy')[:a.n].astype(np.int64)).to(dev)
    gt = np.load(root / 'dev_mel.npy', mmap_mode='r')[:a.n].astype(np.float32)
    act = np.load(root / 'dev_act.npy')[:a.n] if (root / 'dev_act.npy').exists() else None
    band_thr = calibrate_flashdot(root) if act is not None else None
    args = SimpleNamespace(pal_bs=a.n, crop_cols=12, margin=1, pal_amp=1, decode='st', K=8, segments=4,
                           tau=0.1, sigma=1.0, reduce='mean_then_log', sigma_late=None)
    rng = np.random.default_rng(0)
    gt_logits = torch.nn.functional.one_hot(tok, 128).float() * 20
    with torch.no_grad():
        ref = np.mean([pal_term(vq, c, gt_logits, feats, args, np.random.default_rng(i))[0].item() for i in range(8)])
    logits = torch.randn(a.n, 265, 128, device=dev, requires_grad=True)
    rand = logits.detach().clone()
    opt = torch.optim.Adam([logits], lr=0.1)
    for it in range(a.iters):
        l, _ = pal_term(vq, c, logits, feats, args, rng)
        opt.zero_grad(); l.backward(); opt.step()
    with torch.no_grad():
        final = np.mean([pal_term(vq, c, logits, feats, args, np.random.default_rng(i))[0].item() for i in range(8)])
        start = np.mean([pal_term(vq, c, rand, feats, args, np.random.default_rng(i))[0].item() for i in range(8)])
    res = dict(critic=a.critic, pal_random=start, pal_optimised=final, pal_gt_codes=ref,
               random=score(decode(vq, rand), gt, act, band_thr),
               optimised=score(decode(vq, logits.detach()), gt, act, band_thr),
               gt_codes=score(decode(vq, gt_logits), gt, act, band_thr))
    print(json.dumps(res))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

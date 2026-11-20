"""Paired bootstrap over clips between sample files written by sample_eval.

Per-clip metrics (onset_f1, onset_ap, env_corr, abs_lag_ms, lag_within_1vf; act_f1 on FlashDot;
sync_tol0 / sync_tol1 / sync_abs_offset_ms with --eval_critic) are computed exactly as sample_eval
computes its aggregates, so their means reproduce metrics_*.json. They are cached next to each
sample file as <stem>_perclip.npz. For each pair a:b the difference b - a is bootstrapped over
clips (percentile CI, two-sided p) and also tested with a paired sign-flip permutation test.

  python -m perceptalign.significance --data work/flashdot_std --split dev \
      --eval_critic work/runs/fd_evalcritic/critic.pt \
      --samples base=work/runs/fd2_ce_pk05/samples_dev_s0.npy rr8=work/runs/fd2_ce_pk05/samples_dev_s0_rr8.npy \
      --out work/results/sig_fd_dev_rr8.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from perceptalign import timing_metrics as tm
from perceptalign.pal import pal_loss
from perceptalign.sample_eval import calibrate_flashdot, flashdot_activity, load_critic

METRICS = ('onset_f1', 'onset_ap', 'env_corr', 'abs_lag_ms', 'lag_within_1vf', 'act_f1',
           'sync_tol0', 'sync_tol1', 'sync_abs_offset_ms')
# sample_eval names for the per-clip critic metrics
AGG_NAME = dict(sync_tol0='sync_acc_tol0', sync_tol1='sync_acc_tol1')


@torch.no_grad()
def critic_perclip(critic, mels, feats, dev, bs=32):
    """critic_scores (segments=1, sigma 0) with one value per clip instead of the mean."""
    c, a = critic if isinstance(critic, tuple) else load_critic(critic, dev)
    out = {k: [] for k in ('sync_tol0', 'sync_tol1', 'sync_abs_offset_ms')}
    for s in range(0, len(mels), bs):
        m = torch.from_numpy(np.asarray(mels[s:s + bs], np.float32)).to(dev)
        v = torch.from_numpy(np.asarray(feats[s:s + bs], np.float32)).transpose(1, 2).to(dev)
        _, p = pal_loss(c.embed_audio(2 * m - 1), c.embed_video(v), a['K'], 1, a['tau'], 0.0)
        off = (p.argmax(-1) - a['K']).abs()[:, 0].float().cpu().numpy()
        out['sync_tol0'].append(off <= 0); out['sync_tol1'].append(off <= 1)
        out['sync_abs_offset_ms'].append(off * (1000 / 21.5))
    return {k: np.concatenate(v).astype(np.float64) for k, v in out.items()}


def perclip_metrics(gen, root, split, idx=None, eval_critic=None, dev='cpu', chunk=256):
    """gen: (M, 80, 848) in [0, 1] (memmap fine); row r is scored against GT clip idx[r] (default r).

    Returns {metric: float64 (M,)}.
    """
    root = Path(root)
    gt = np.load(root / f'{split}_mel.npy', mmap_mode='r')
    idx = np.arange(len(gen)) if idx is None else np.asarray(idx)
    rows = [tm.clip_scores(np.asarray(gen[r], np.float32), np.asarray(gt[idx[r]], np.float32))
            for r in range(len(gen))]
    lag = np.array([r['lag_ms'] for r in rows], np.float64)
    res = {k: np.array([r[k] for r in rows], np.float64) for k in ('onset_f1', 'onset_ap', 'env_corr')}
    res['abs_lag_ms'] = np.abs(lag)
    res['lag_within_1vf'] = np.where(np.isnan(lag), np.nan, np.abs(lag) <= 1000 / 21.5 + 1e-6)
    act_path = root / f'{split}_act.npy'
    if act_path.exists():
        band, thr = calibrate_flashdot(root)
        act = np.load(act_path)
        res['act_f1'] = np.array([tm.activity_scores(flashdot_activity(np.asarray(gen[r], np.float32), band, thr),
                                                     act[idx[r]])['act_f1'] for r in range(len(gen))])
    if eval_critic:
        feats = np.load(root / f'{split}_feats.npy', mmap_mode='r')
        critic = load_critic(eval_critic, dev)
        parts = [critic_perclip(critic, gen[s:s + chunk], feats[idx[s:s + chunk]], dev)
                 for s in range(0, len(gen), chunk)]
        res.update({k: np.concatenate([p[k] for p in parts]) for k in parts[0]})
    return res


def cached_perclip(path, root, split, eval_critic=None, dev='cpu'):
    """Per-clip metrics of a sample file, cached as <stem>_perclip.npz next to it."""
    path = Path(path)
    cp = path.with_name(path.stem + '_perclip.npz')
    meta = json.dumps([str(Path(root).resolve()), split, str(eval_critic)])
    if cp.exists() and cp.stat().st_mtime >= path.stat().st_mtime:
        z = np.load(cp)
        if str(z['_meta']) == meta:
            return {k: z[k] for k in z.files if k != '_meta'}
    gen = np.load(path, mmap_mode='r')
    res = perclip_metrics(gen, root, split, eval_critic=eval_critic, dev=dev)
    np.savez(cp, _meta=meta, **res)
    return res


def paired_bootstrap(a, b, n_boot=10000, seed=0):
    """Bootstrap of mean(b - a) over clips; drops clips that are NaN in either array."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    d = b - a
    if len(d) == 0:
        return dict(n=0)
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    p_boot = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
    flips = rng.choice([-1.0, 1.0], (n_boot, len(d)))
    perm = (flips * d).mean(1)
    p_perm = (1 + (np.abs(perm) >= abs(d.mean()) - 1e-12).sum()) / (1 + n_boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(n=int(len(d)), mean_a=float(a.mean()), mean_b=float(b.mean()), diff=float(d.mean()),
                ci95=[float(lo), float(hi)], p_boot=float(p_boot), p_perm=float(p_perm))


def compare(pa, pb, metrics=None, n_boot=10000, seed=0):
    """Paired bootstrap b - a for every metric the two per-clip dicts share."""
    metrics = [m for m in (metrics or METRICS) if m in pa and m in pb]
    return {m: paired_bootstrap(pa[m], pb[m], n_boot, seed) for m in metrics}


def print_table(res):
    print(f"{'pair':24s} {'metric':18s} {'n':>4s} {'a':>9s} {'b':>9s} {'b-a':>9s} {'95% CI':>21s} {'p_boot':>7s} {'p_perm':>7s}")
    for pair, ms in res.items():
        for m, r in ms.items():
            if not r['n']:
                continue
            ci = f'[{r["ci95"][0]:.4g}, {r["ci95"][1]:.4g}]'
            print(f'{pair:24s} {m:18s} {r["n"]:4d} {r["mean_a"]:9.4g} {r["mean_b"]:9.4g} {r["diff"]:9.4g} '
                  f'{ci:>21s} {r["p_boot"]:7.4f} {r["p_perm"]:7.4f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--split', default='dev')
    ap.add_argument('--eval_critic', default=None)
    ap.add_argument('--samples', nargs='+', required=True, help='name=path.npy ...')
    ap.add_argument('--pairs', nargs='*', default=None, help='a:b ... (b - a); default: every entry vs the first')
    ap.add_argument('--n_boot', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    files = dict(s.split('=', 1) for s in args.samples)
    names = list(files)
    pairs = [p.split(':') for p in args.pairs] if args.pairs else [(names[0], n) for n in names[1:]]
    pc = {n: cached_perclip(f, args.data, args.split, args.eval_critic, args.device) for n, f in files.items()}
    for n, d in pc.items():
        print(n, json.dumps({AGG_NAME.get(k, k): round(float(np.nanmean(v)), 5) for k, v in d.items()}))
    res = {f'{a}:{b}': compare(pc[a], pc[b], n_boot=args.n_boot, seed=args.seed) for a, b in pairs}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print_table(res)


if __name__ == '__main__':
    main()

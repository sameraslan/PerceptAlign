"""Sample once, select many ways: candidate-selection rules compared on the SAME candidates.

For every clip n_cand free-running samples are drawn exactly as sample_eval draws its rerank
candidates (same seed, same order), so best{n_cand}_s1 with the sample_eval rerank critic matches
sample_eval --rerank n_cand --rr_sigma 1 up to fp16 rounding of the scored spectrograms. Every
candidate is scored with the rerank critic under every binding window and with every per-clip
metric (significance.perclip_metrics), and selections are made by indexing:

  first            candidate 0 (one plain sample)
  best{n}_{w}      lowest PAL under window w among the first n candidates (n = 2, 4, ..., n_cand)
  oracle_actf1     candidate with the highest true act_f1 (FlashDot upper bound)
  oracle_lag       candidate with the lowest true |lag_ms| (upper bound, mainly for VAS)

Windows: sN is a symmetric Gaussian window of width N (s0 = zero offset only); aE-L is asymmetric
with sigma=E on the audio-early side (offset o > 0) and sigma_late=L on the audio-late side
(o < 0, the side binding_window's sigma_late controls). All use reduce mean_then_log and the
rerank critic's own K / segments / tau.

Writes to --out: cands_<tag>.npy (float16 (N, n_cand, 80, 848) in [0, 1], reused if present),
scores_<tag>.npz, perclip_<tag>.npz (metrics (N, n_cand)), sel_<tag>.npz (per-clip arrays of every
selection, keys '<selection>/<metric>') and rerank_ablation_<tag>.json (per-selection means and
paired bootstraps). With --semantic 1, Melception FID / MKL of each selected set as well.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from perceptalign import significance as sig
from perceptalign.encode import load_vqgan
from perceptalign.pal import pal_loss
from perceptalign.sample_eval import load_critic, load_generator


def parse_window(w):
    """'s1' -> (1.0, None); 'a0.5-2' -> (0.5, 2.0) = (sigma audio-early, sigma_late)."""
    if w[0] == 's':
        return float(w[1:]), None
    e, l = w[1:].split('-')
    return float(e), float(l)


@torch.no_grad()
def sample_candidates(args, feats, N, path, dev):
    gpt, a = load_generator(args.run, dev)
    vq = load_vqgan(args.codebook, dev)
    del vq.encoder
    torch.manual_seed(args.seed)
    tmp = path.with_suffix('.tmp.npy')
    cands = np.lib.format.open_memmap(tmp, 'w+', np.float16, (N, args.n_cand, 80, 848))
    for s in range(0, N, args.bs):
        f = torch.from_numpy(np.asarray(feats[s:s + args.bs], np.float32)).transpose(1, 2).to(dev)
        for k in range(args.n_cand):
            with torch.autocast('mps', dtype=torch.float16, enabled=dev == 'mps' and bool(a.amp or a.lora_rank)):
                tok = gpt.sample(f, temperature=args.temperature, top_k=args.top_k)
                m = torch.cat([vq.decode_tokens(tok[j:j + 2])[:, 0].float() for j in range(0, len(tok), 2)])
            cands[s:s + len(m), k] = ((m + 1) / 2).clamp(0, 1).cpu().numpy().astype(np.float16)
        print('sampled', s + len(m), N, flush=True)
    cands.flush()
    del cands, gpt, vq
    os.replace(tmp, path)
    if dev == 'mps':
        torch.mps.empty_cache()


@torch.no_grad()
def window_scores(critic_path, cands, feats, windows, dev, bs=16):
    """PAL of every candidate under every window -> (n_windows, N, n_cand); lower is better."""
    c, ca = load_critic(critic_path, dev)
    N, C = cands.shape[:2]
    out = np.zeros((len(windows), N, C), np.float32)
    for s in range(0, N, bs):
        v = c.embed_video(torch.from_numpy(np.asarray(feats[s:s + bs], np.float32)).transpose(1, 2).to(dev))
        for k in range(C):
            a = c.embed_audio(2 * torch.from_numpy(np.asarray(cands[s:s + bs, k], np.float32)).to(dev) - 1)
            for i, w in enumerate(windows):
                sigma, late = parse_window(w)
                out[i, s:s + bs, k] = pal_loss(a, v, ca['K'], ca['segments'], ca['tau'], sigma,
                                               'mean_then_log', late)[0].cpu().numpy()
    return out


def melception_chunked(mels, dev, chunk=128):
    from perceptalign import semantic_metrics as sm
    model = sm.load_melception(dev)
    parts = [sm.melception_features(np.asarray(mels[s:s + chunk], np.float32), dev, 8, model)
             for s in range(0, len(mels), chunk)]
    del model
    return {k: torch.cat([p[k] for p in parts]).numpy() for k in ('2048', 'logits')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='run dir or public .ckpt')
    ap.add_argument('--data', required=True)
    ap.add_argument('--codebook', required=True)
    ap.add_argument('--split', default='dev')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--top_k', type=int, default=100)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--n_cand', type=int, default=8)
    ap.add_argument('--rerank_critic', required=True)
    ap.add_argument('--windows', nargs='+', default=['s0', 's0.5', 's1', 's2', 'a0.5-2', 'a2-0.5'])
    ap.add_argument('--eval_critic', default=None)
    ap.add_argument('--semantic', type=int, default=0)
    ap.add_argument('--n_boot', type=int, default=10000)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='mps')
    args = ap.parse_args()
    dev, root, out = args.device, Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gt = np.load(root / f'{args.split}_mel.npy', mmap_mode='r')
    feats = np.load(root / f'{args.split}_feats.npy', mmap_mode='r')
    N = len(gt) if args.limit is None else min(args.limit, len(gt))
    feats, C = feats[:N], args.n_cand
    ck_path = Path(args.run) if str(args.run).endswith('.ckpt') else Path(args.run) / 'last.pt'
    key = json.dumps([str(ck_path), ck_path.stat().st_mtime, args.top_k, args.temperature])
    tag = f'{args.split}_s{args.seed}_n{N}_c{C}_' + hashlib.md5(key.encode()).hexdigest()[:6]

    # (1) candidates
    cp = out / f'cands_{tag}.npy'
    if not cp.exists():
        sample_candidates(args, feats, N, cp, dev)
    cands = np.load(cp, mmap_mode='r')

    # (2) PAL of every candidate under every window
    scores = window_scores(args.rerank_critic, cands, feats, args.windows, dev)
    np.savez(out / f'scores_{tag}.npz', scores=scores, windows=np.array(args.windows), critic=args.rerank_critic)

    # (3) per-candidate metrics, cached (they depend only on the candidates and the eval critic)
    mp = out / f'perclip_{tag}.npz'
    meta = json.dumps([str(args.eval_critic)])
    if mp.exists() and str(np.load(mp)['_meta']) == meta:
        z = np.load(mp)
        M = {k: z[k] for k in z.files if k != '_meta'}
    else:
        flat = sig.perclip_metrics(cands.reshape(N * C, 80, 848), root, args.split, np.repeat(np.arange(N), C),
                                   args.eval_critic, dev)
        M = {k: v.reshape(N, C) for k, v in flat.items()}
        np.savez(mp, _meta=meta, **M)

    # (4) selections: chosen candidate index per clip
    sel = {'first': np.zeros(N, int)}
    ns = sorted({2 ** i for i in range(1, C.bit_length())} | ({C} if C > 1 else set()))
    for i, w in enumerate(args.windows):
        for n in ns:
            sel[f'best{n}_{w}'] = scores[i, :, :n].argmin(1)
    if 'act_f1' in M:
        sel['oracle_actf1'] = np.nan_to_num(M['act_f1'], nan=-np.inf).argmax(1)
    sel['oracle_lag'] = np.nan_to_num(M['abs_lag_ms'], nan=np.inf).argmin(1)
    rows = np.arange(N)
    pc = {s: {m: v[rows, idx] for m, v in M.items()} for s, idx in sel.items()}
    np.savez(out / f'sel_{tag}.npz', **{f'{s}/{m}': v for s, d in pc.items() for m, v in d.items()},
             **{f'{s}/idx': idx for s, idx in sel.items()})
    means = {s: {m: float(np.nanmean(v)) for m, v in d.items()} for s, d in pc.items()}

    if args.semantic:
        fp = out / f'melfeats_{tag}.npz'
        if fp.exists():
            F = dict(np.load(fp))
        else:
            fc = melception_chunked(cands.reshape(N * C, 80, 848), dev)
            fg = melception_chunked(gt[:N], dev)
            F = {'cand_2048': fc['2048'], 'cand_logits': fc['logits'], 'gt_2048': fg['2048'], 'gt_logits': fg['logits']}
            np.savez(fp, **F)
            if dev == 'mps':
                torch.mps.empty_cache()
        from perceptalign import semantic_metrics as sm
        for s, idx in sel.items():
            j = rows * C + idx
            means[s].update(fid=float(sm.fid(F['cand_2048'][j], F['gt_2048'])),
                            mkl=float(sm.mkl(F['cand_logits'][j], F['gt_logits'])))

    # paired bootstraps between selections (key a:b reports b - a, as in significance.py)
    pairs = [('first', f'best{C}_s1'), (f'best{C}_s1', f'best{C}_s0')]
    pairs += [(f'best{C}_s1', f'best{C}_{w}') for w in args.windows if w[0] == 'a']
    boot = {f'{a}:{b}': sig.compare(pc[a], pc[b], n_boot=args.n_boot) for a, b in pairs if a in pc and b in pc}
    res = dict(tag=tag, run=str(args.run), n=N, n_cand=C, windows=args.windows, rerank_critic=args.rerank_critic,
               eval_critic=args.eval_critic, means=means, bootstrap=boot)
    (out / f'rerank_ablation_{tag}.json').write_text(json.dumps(res, indent=1))
    cols = [k for k in ('act_f1', 'onset_f1', 'abs_lag_ms', 'lag_within_1vf', 'sync_tol0', 'fid', 'mkl')
            if k in means['first']]
    print(f"{'selection':18s}" + ''.join(f'{k:>15s}' for k in cols))
    for s, d in means.items():
        print(f'{s:18s}' + ''.join(f'{d[k]:15.4f}' for k in cols))
    sig.print_table(boot)


if __name__ == '__main__':
    os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.3')
    os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO', '0.25')
    main()

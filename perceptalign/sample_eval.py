"""Sample spectrograms from a trained transformer and score them.

Writes <run>/samples_<split>_s<seed>.npy (float16 (N, 80, 848) in [0, 1]) and
<run>/metrics_<split>_s<seed>.json. Timing metrics compare every sample with its
ground truth; FlashDot adds frame-level tone-activity scores; VAS adds Melception FID / MKL.
With --run gt or --run recon the ground truth or its VQGAN reconstruction is scored instead
(reference rows for the tables).
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from perceptalign import timing_metrics as tm
from perceptalign.encode import load_vqgan
from perceptalign.models import torch_load
from perceptalign.pal import build_critic, offset_accuracy, pal_loss
from perceptalign.train_gen import add_lora, build_gpt


class Args(dict):
    __getattr__ = dict.get


def load_generator(run, dev):
    """Load a trained run, or the public SpecVQGAN transformer when `run` is a .ckpt path."""
    if str(run).endswith('.ckpt'):
        a = Args(init=str(run), lora_rank=0, amp=1)
        return build_gpt(a).half().to(dev).eval(), a
    ck = torch_load(Path(run) / 'last.pt')
    a = Args(ck['args'])
    gpt = build_gpt(a)
    if a.init != 'scratch':
        gpt.half()
    if a.lora_rank > 0:
        add_lora(gpt, a.lora_rank, a.lora_alpha)
    missing, unexpected = gpt.load_state_dict(ck['state_dict'], strict=a.lora_rank == 0)
    assert not unexpected, unexpected[:5]
    assert not any(k.endswith('.A') or k.endswith('.B') for k in missing), 'adapter weights missing'
    return gpt.to(dev).eval(), a


def flashdot_activity(mel, band, thr):
    return (mel[band].mean(0) > thr).astype(np.uint8)


def calibrate_flashdot(root):
    """Pick the tone band and threshold on TRAIN VQGAN reconstructions (never on test).

    Every generated spectrogram comes out of the VQGAN decoder, which renders the tone
    slightly quieter than the ground truth, so the detector is calibrated in that domain.
    """
    mel = np.load(root / 'train_recon.npy', mmap_mode='r')[:400].astype(np.float32)
    act = np.load(root / 'train_act.npy')[:400].astype(bool)
    on = np.stack([m[:, a].mean(1) for m, a in zip(mel, act) if a.any()]).mean(0)
    off = np.stack([m[:, ~a].mean(1) for m, a in zip(mel, act)]).mean(0)
    band = np.argsort(on - off)[-3:]
    e = mel[:, band].mean(1)
    best = max(np.linspace(0.05, 0.95, 91), key=lambda t: tm.activity_scores(e > t, act)['act_f1'])
    return band, float(best)


def load_critic(path, dev):
    ck = torch_load(path)
    a = ck['args']
    c = build_critic(a)
    c.load_state_dict(ck['state_dict'])
    return c.to(dev).eval(), a


@torch.no_grad()
def critic_scores(critic_path, mels, feats, dev):
    c, a = load_critic(critic_path, dev)
    acc0, acc1, lag = [], [], []
    for s in range(0, len(mels), 32):
        m = torch.from_numpy(mels[s:s + 32].astype(np.float32)).to(dev)
        v = torch.from_numpy(np.asarray(feats[s:s + 32], np.float32)).transpose(1, 2).to(dev)
        _, p = pal_loss(c.embed_audio(2 * m - 1), c.embed_video(v), a['K'], 1, a['tau'], 0.0)
        acc0.append(offset_accuracy(p, 0) * len(m)); acc1.append(offset_accuracy(p, 1) * len(m))
        lag.append(((p.argmax(-1) - a['K']).abs().float() * (1000 / 21.5)).sum().item())
    n = len(mels)
    return dict(sync_acc_tol0=sum(acc0) / n, sync_acc_tol1=sum(acc1) / n, sync_abs_offset_ms=sum(lag) / n)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='run dir, or "gt" / "recon"')
    ap.add_argument('--data', required=True)
    ap.add_argument('--codebook', required=True)
    ap.add_argument('--split', default='dev')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--top_k', type=int, default=100)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--eval_critic', default=None)
    ap.add_argument('--semantic', type=int, default=0)
    ap.add_argument('--out', default=None)
    ap.add_argument('--rerank', type=int, default=1, help='draw this many samples per clip and keep the best under --rerank_critic')
    ap.add_argument('--rerank_critic', default=None)
    ap.add_argument('--rr_sigma', type=float, default=1.0, help='binding-window width used to score candidates')
    ap.add_argument('--rr_reduce', default='mean_then_log')
    args = ap.parse_args()
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    root = Path(args.data)
    gt = np.load(root / f'{args.split}_mel.npy', mmap_mode='r')
    feats = np.load(root / f'{args.split}_feats.npy', mmap_mode='r')
    N = len(gt) if args.limit is None else min(args.limit, len(gt))
    out_dir = Path(args.out or (args.run if args.run not in ('gt', 'recon') else root / f'ref_{args.run}'))
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f'{args.split}_s{args.seed}' + (f'_n{N}' if args.limit else '') + (f'_rr{args.rerank}' if args.rerank > 1 else '')
    if args.run not in ('gt', 'recon'):
        # samples are cached; the cache key covers everything that changes them
        import hashlib
        ck_path = Path(args.run) if str(args.run).endswith('.ckpt') else Path(args.run) / 'last.pt'
        key = json.dumps([str(ck_path), ck_path.stat().st_mtime, args.top_k, args.temperature,
                          args.rerank_critic, args.rr_sigma, args.rr_reduce])
        tag += '_' + hashlib.md5(key.encode()).hexdigest()[:6]

    if args.run == 'gt':
        gen = np.asarray(gt[:N])
    elif args.run == 'recon':
        gen = np.asarray(np.load(root / f'{args.split}_recon.npy', mmap_mode='r')[:N])
    else:
        sp = out_dir / f'samples_{tag}.npy'
        if sp.exists():
            gen = np.load(sp)
        else:
            gpt, a = load_generator(args.run, dev)
            vq = load_vqgan(args.codebook, dev)
            del vq.encoder
            torch.manual_seed(args.seed)
            gen = np.zeros((N, 80, 848), np.float16)
            rr = load_critic(args.rerank_critic, dev) if args.rerank > 1 else None
            for s in range(0, N, args.bs):
                f = torch.from_numpy(np.asarray(feats[s:s + args.bs], np.float32)).transpose(1, 2).to(dev)
                best, best_score = None, None
                for _ in range(args.rerank):
                    with torch.autocast('mps', dtype=torch.float16, enabled=bool(a.amp or a.lora_rank)):
                        tok = gpt.sample(f, temperature=args.temperature, top_k=args.top_k)
                        m = torch.cat([vq.decode_tokens(tok[j:j + 2])[:, 0].float() for j in range(0, len(tok), 2)])
                    if rr is None:
                        best = m
                        break
                    c, ca = rr
                    with torch.no_grad():
                        sc, _ = pal_loss(c.embed_audio(m.clamp(-1, 1)), c.embed_video(f), ca['K'], ca['segments'],
                                         ca['tau'], args.rr_sigma, args.rr_reduce)
                    if best is None:
                        best, best_score = m, sc
                    else:
                        better = sc < best_score
                        best = torch.where(better[:, None, None], m, best)
                        best_score = torch.minimum(sc, best_score)
                m = best
                gen[s:s + len(m)] = ((m + 1) / 2).clamp(0, 1).cpu().numpy().astype(np.float16)
                print('sampled', s + len(m), N, flush=True)
            np.save(sp, gen)
            del gpt, vq
            torch.mps.empty_cache()

    rows = [tm.clip_scores(gen[i].astype(np.float32), np.asarray(gt[i], np.float32)) for i in range(N)]
    res = dict(n=N, split=args.split, seed=args.seed, **tm.aggregate(rows))
    act_path = root / f'{args.split}_act.npy'
    if act_path.exists():
        band, thr = calibrate_flashdot(root)
        act = np.load(act_path)[:N]
        fr = [tm.activity_scores(flashdot_activity(gen[i].astype(np.float32), band, thr), act[i]) for i in range(N)]
        res.update({k: float(np.mean([r[k] for r in fr])) for k in fr[0]})
    if args.eval_critic:
        res.update(critic_scores(args.eval_critic, gen, feats[:N], dev))
    if args.semantic:
        from perceptalign import semantic_metrics as sm
        fg = sm.melception_features(gen.astype(np.float32), dev, 4)
        fr_ = sm.melception_features(np.asarray(gt[:N], np.float32), dev, 4)
        res.update(fid=float(sm.fid(fg['2048'], fr_['2048'])), mkl=float(sm.mkl(fg['logits'], fr_['logits'])),
                   isc=float(sm.inception_score(fg['logits_unbiased'])[0]))
    (out_dir / f'metrics_{tag}.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == '__main__':
    os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.3')
    os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO', '0.25')
    main()

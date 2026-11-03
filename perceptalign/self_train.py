"""PAL-filtered self-training.

Each round: for a random subset of training clips, draw --n_cand free-running samples from the
current generator, score every candidate with the frozen PAL critic, keep one candidate per
clip (the lowest PAL with --select pal, the first candidate with --select random), then
fine-tune the generator with cross-entropy on a mix of kept samples and ground truth.

Because the candidates are the generator's own samples, the critic only ranks realistic
spectrograms and is never optimised against directly. The random-selection control uses the
same samples, steps and data mix, so any difference comes from the critic's choice.

--init is either the public SpecVQGAN .ckpt (with --lora_rank > 0: frozen fp16 base plus
LoRA adapters) or a run directory produced by train_gen.py (full fine-tuning).
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from perceptalign.encode import load_vqgan
from perceptalign.models import torch_load
from perceptalign.pal import build_critic, pal_loss
from perceptalign.sample_eval import Args, load_generator
from perceptalign.train_gen import add_lora, build_gpt, forward_logits


def load_init(args, dev):
    if args.init.endswith('.ckpt'):
        gpt = build_gpt(Args(init=args.init)).half()
        for p in gpt.parameters():
            p.requires_grad = False
        add_lora(gpt, args.lora_rank, args.lora_alpha)
        base_args = dict(init=args.init, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, amp=1)
        return gpt.to(dev), base_args
    gpt, a = load_generator(args.init, dev)
    return gpt, dict(a)


@torch.no_grad()
def score_candidates(gpt, vq, critic, ca, feats, args, dev):
    """Sample n_cand candidates for a batch of clips; return tokens (n_cand, B, 265) and PAL (n_cand, B)."""
    toks, scores = [], []
    for _ in range(args.n_cand):
        with torch.autocast('mps', dtype=torch.float16, enabled=bool(args.amp)):
            t = gpt.sample(feats, top_k=args.top_k)
            mel = torch.cat([vq.decode_tokens(t[j:j + 2])[:, 0].float() for j in range(0, len(t), 2)])
        l, _ = pal_loss(critic.embed_audio(mel.clamp(-1, 1)), critic.embed_video(feats.float()),
                        args.K, args.segments, args.tau, args.sigma, args.reduce, args.sigma_late)
        toks.append(t); scores.append(l)
    return torch.stack(toks), torch.stack(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--codebook', required=True)
    ap.add_argument('--init', required=True)
    ap.add_argument('--critic', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--select', default='pal', choices=['pal', 'random'])
    ap.add_argument('--lora_rank', type=int, default=16)
    ap.add_argument('--lora_alpha', type=float, default=16)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--clips_per_round', type=int, default=512)
    ap.add_argument('--n_cand', type=int, default=4)
    ap.add_argument('--top_k', type=int, default=100)
    ap.add_argument('--sample_bs', type=int, default=16)
    ap.add_argument('--steps_per_round', type=int, default=200)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--gt_frac', type=float, default=0.5, help='fraction of each batch drawn from ground truth')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--amp', type=int, default=1)
    ap.add_argument('--grad_ckpt', type=int, default=1)
    ap.add_argument('--K', type=int, default=10)
    ap.add_argument('--segments', type=int, default=8)
    ap.add_argument('--tau', type=float, default=0.1)
    ap.add_argument('--sigma', type=float, default=1.0)
    ap.add_argument('--sigma_late', type=float, default=None)
    ap.add_argument('--reduce', default='mean_then_log')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    dev = 'mps'
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'args.json').write_text(json.dumps(vars(args), indent=1))

    vq = load_vqgan(args.codebook, dev)
    del vq.encoder
    ck = torch_load(args.critic)
    critic = build_critic(ck['args']); critic.load_state_dict(ck['state_dict']); critic = critic.to(dev).eval()
    for p in list(vq.parameters()) + list(critic.parameters()):
        p.requires_grad = False
    gpt, base_args = load_init(args, dev)
    params = [p for p in gpt.parameters() if p.requires_grad]
    print('trainable params (M):', sum(p.numel() for p in params) / 1e6, flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    root = Path(args.data)
    feats_all = np.load(root / 'train_feats.npy', mmap_mode='r')
    tok_all = np.load(root / 'train_tokens.npy')
    N = len(tok_all)
    log, t0 = [], time.time()
    for r in range(args.rounds):
        # 1. sample, score, select
        gpt.eval()
        idx = np.sort(rng.choice(N, args.clips_per_round, replace=False))
        kept = np.zeros((len(idx), 265), np.int64)
        pal_kept, pal_first, pal_mean = [], [], []
        for s in range(0, len(idx), args.sample_bs):
            ii = idx[s:s + args.sample_bs]
            f = torch.from_numpy(np.asarray(feats_all[ii], np.float32)).transpose(1, 2).to(dev)
            toks, sc = score_candidates(gpt, vq, critic, ck['args'], f, args, dev)
            pick = sc.argmin(0) if args.select == 'pal' else torch.zeros(sc.shape[1], dtype=torch.long, device=dev)
            ar = torch.arange(sc.shape[1], device=dev)
            kept[s:s + len(ii)] = toks[pick, ar].cpu().numpy()
            pal_kept += sc[pick, ar].tolist(); pal_first += sc[0].tolist(); pal_mean += sc.mean(0).tolist()
            torch.mps.empty_cache()
        np.save(out / f'round{r}_kept_tokens.npy', kept.astype(np.int16))
        np.save(out / f'round{r}_clip_index.npy', idx)
        # 2. fine-tune on kept samples + ground truth
        gpt.train()
        n_self = int(round(args.bs * (1 - args.gt_frac)))
        ce_hist = []
        for step in range(args.steps_per_round):
            a_i = rng.choice(len(idx), n_self, replace=False)
            g_i = rng.choice(N, args.bs - n_self, replace=False)
            ii = np.concatenate([idx[a_i], g_i])
            tk = torch.from_numpy(np.concatenate([kept[a_i], tok_all[g_i].astype(np.int64)])).to(dev)
            f = torch.from_numpy(np.asarray(feats_all[np.sort(ii)], np.float32))
            order = np.argsort(np.argsort(ii))  # memmap fancy indexing wants sorted indices
            f = f[torch.from_numpy(order)].transpose(1, 2).to(dev)
            with torch.autocast('mps', dtype=torch.float16, enabled=bool(args.amp)):
                lg = forward_logits(gpt, tk, f, args.grad_ckpt)
            ce = F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), tk.reshape(-1))
            opt.zero_grad(set_to_none=True); ce.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            ce_hist.append(ce.item())
        ev = dict(round=r, pal_kept=float(np.mean(pal_kept)), pal_first=float(np.mean(pal_first)),
                  pal_mean=float(np.mean(pal_mean)), train_ce=float(np.mean(ce_hist)), sec=round(time.time() - t0),
                  mps_gb=round(torch.mps.driver_allocated_memory() / 1e9, 2))
        log.append(ev); print(json.dumps(ev), flush=True)
        state = {k: v for k, v in gpt.state_dict().items()
                 if base_args.get('lora_rank', 0) == 0 or k.endswith('.A') or k.endswith('.B')}
        torch.save({'state_dict': state, 'args': {**base_args, 'self_train': vars(args)}, 'round': r}, out / 'last.pt')
    (out / 'log.json').write_text(json.dumps(log, indent=1))


if __name__ == '__main__':
    os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.3')
    os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO', '0.25')
    main()

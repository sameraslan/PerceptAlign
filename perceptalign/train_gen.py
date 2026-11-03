"""Stage 2: train or fine-tune the transformer with cross-entropy (CE) plus PAL.

    L = CE + beta * PAL( critic_a(decode(logits)), critic_v(video) )

The critic and the VQGAN decoder are frozen. Logits reach the decoder through
straight-through (or soft) codebook decoding. To bound memory, PAL is computed on a random
window of --crop_cols code columns (16 mel frames, 4 video frames each); --margin columns
on each side are decoded for context and then discarded.

--init scratch trains a small GPT from scratch (FlashDot). --init <ckpt> loads the public
SpecVQGAN VAS transformer; with --lora_rank > 0 its weights stay frozen (fp16) and only
low-rank adapters on every linear layer of the transformer blocks are trained.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from perceptalign.data import ClipDataset
from perceptalign.encode import load_vqgan
from perceptalign.models import Z_H, GPTFeats, load_state, torch_load
from perceptalign.pal import build_critic, pal_loss
from perceptalign.train_critic import batches


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.randn(rank, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + (x.to(self.A.dtype) @ self.A.t() @ self.B.t()).to(x.dtype) * self.scale


def add_lora(gpt, rank, alpha):
    for blk in gpt.blocks:
        for name in ['key', 'query', 'value', 'proj']:
            setattr(blk.attn, name, LoRALinear(getattr(blk.attn, name), rank, alpha))
        blk.mlp[0] = LoRALinear(blk.mlp[0], rank, alpha)
        blk.mlp[2] = LoRALinear(blk.mlp[2], rank, alpha)


def build_gpt(args):
    if args.init == 'scratch':
        return GPTFeats(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)
    gpt = GPTFeats()
    sd = torch_load(args.init)
    sd = sd.get('state_dict', sd)
    missing, unexpected = load_state(gpt, sd, 'transformer.')
    assert not missing, missing[:10]
    return gpt


def checkpointed_decode(dec, z):
    """VQGAN Decoder.forward with activation checkpointing around every ResNet/attention block."""
    ck = lambda f, *x: checkpoint(f, *x, use_reentrant=False)
    h = dec.conv_in(z)
    h = ck(lambda x: dec.mid.block_2(dec.mid.attn_1(dec.mid.block_1(x, None)), None), h)
    for i_level in reversed(range(dec.num_resolutions)):
        up = dec.up[i_level]
        for i_block in range(dec.num_res_blocks + 1):
            if len(up.attn) > 0:
                h = ck(lambda x, b=up.block[i_block], a=up.attn[i_block]: a(b(x, None)), h)
            else:
                h = ck(lambda x, b=up.block[i_block]: b(x, None), h)
        if i_level != 0:
            h = ck(up.upsample, h)
    return ck(lambda x: dec.conv_out(torch.nn.functional.silu(dec.norm_out(x))), h)


def decode_crop(vq, logits, c0, cols, mode):
    """Decode code columns [c0, c0 + cols) from logits (B, 265, V) -> mel (B, 80, 16 * cols)."""
    lg = logits[:, c0 * Z_H:(c0 + cols) * Z_H]
    p = F.softmax(lg.float(), dim=-1)
    if mode == 'st':
        w = F.one_hot(p.argmax(-1), p.shape[-1]).float() + p - p.detach()
    else:
        w = p
    emb = (w @ vq.codebook).view(lg.shape[0], cols, Z_H, -1).permute(0, 3, 2, 1)  # (B, C, 5, cols)
    return checkpointed_decode(vq.decoder, vq.post_quant_conv(emb))[:, 0]


def pal_term(vq, critic, logits, feats, args, rng):
    logits, feats = logits[:args.pal_bs], feats[:args.pal_bs]
    total = args.crop_cols + 2 * args.margin
    c0 = int(rng.integers(0, 53 - total + 1))
    with torch.autocast('mps', dtype=torch.float16, enabled=bool(args.pal_amp)):
        mel = decode_crop(vq, logits, c0, total, args.decode)
        m = args.margin * 16
        mel = mel[:, :, m:mel.shape[-1] - m].float()
        v0 = (c0 + args.margin) * 4
        a = critic.embed_audio(mel)
        v = critic.embed_video(feats[:, :, v0:v0 + args.crop_cols * 4])
    l, p = pal_loss(a.float(), v.float(), args.K, args.segments, args.tau, args.sigma, args.reduce, args.sigma_late)
    return l.mean(), p


def corrupt(tokens, pkeep):
    """Replace each input token by a uniformly random code with probability 1 - pkeep."""
    if pkeep >= 1:
        return tokens
    keep = torch.rand(tokens.shape, device=tokens.device) < pkeep
    return torch.where(keep, tokens, torch.randint_like(tokens, 128))


def forward_logits(gpt, tokens, feats, grad_ckpt):
    if not grad_ckpt:
        return gpt(tokens, feats)
    c = gpt.embedder(feats).transpose(1, 2)
    x = torch.cat([c, gpt.tok_emb(tokens[:, :-1])], dim=1)
    x = x + gpt.pos_emb[:, :x.shape[1]]
    for blk in gpt.blocks:
        x = checkpoint(blk, x, use_reentrant=False)
    return gpt.head(gpt.ln_f(x))[:, c.shape[1] - 1:]


@torch.no_grad()
def dev_ce(gpt, ds, dev, args):
    gpt.eval()
    tot, n = 0.0, 0
    for b in batches(ds, args.bs, False, 0):
        with torch.autocast('mps', dtype=torch.float16, enabled=bool(args.amp)):
            lg = gpt(b['tokens'].to(dev), b['feats'].to(dev))
        tot += F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), b['tokens'].to(dev).reshape(-1), reduction='sum').item()
        n += b['tokens'].numel()
    gpt.train()
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--codebook', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--init', default='scratch')
    ap.add_argument('--n_layer', type=int, default=8)
    ap.add_argument('--n_head', type=int, default=8)
    ap.add_argument('--n_embd', type=int, default=512)
    ap.add_argument('--lora_rank', type=int, default=0)
    ap.add_argument('--lora_alpha', type=float, default=16)
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--warmup', type=int, default=200)
    ap.add_argument('--pal_weight', type=float, default=0.0)
    ap.add_argument('--critic', default=None)
    ap.add_argument('--pkeep', type=float, default=1.0, help='keep-probability of input tokens for the CE pass')
    ap.add_argument('--pal_pkeep', type=float, default=1.0, help='keep-probability of input tokens for the PAL pass')
    ap.add_argument('--pal_start', type=int, default=0, help='step at which PAL switches on')
    ap.add_argument('--pal_ramp', type=int, default=1, help='steps over which beta ramps linearly to --pal_weight')
    ap.add_argument('--K', type=int, default=8)
    ap.add_argument('--segments', type=int, default=4)
    ap.add_argument('--tau', type=float, default=0.1)
    ap.add_argument('--sigma', type=float, default=1.0)
    ap.add_argument('--sigma_late', type=float, default=None)
    ap.add_argument('--reduce', default='mean_then_log')
    ap.add_argument('--decode', default='st')
    ap.add_argument('--crop_cols', type=int, default=12)
    ap.add_argument('--margin', type=int, default=1)
    ap.add_argument('--amp', type=int, default=0, help='fp16 autocast for the transformer')
    ap.add_argument('--pal_amp', type=int, default=1, help='fp16 autocast for the frozen decoder and critic')
    ap.add_argument('--pal_bs', type=int, default=4, help='examples per batch that receive PAL')
    ap.add_argument('--grad_ckpt', type=int, default=0)
    ap.add_argument('--train_limit', type=int, default=None)
    ap.add_argument('--eval_every', type=int, default=500)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'args.json').write_text(json.dumps(vars(args), indent=1))

    vq = load_vqgan(args.codebook, dev)
    del vq.encoder
    for p in vq.parameters():
        p.requires_grad = False
    gpt = build_gpt(args)
    if args.lora_rank > 0:
        for p in gpt.parameters():
            p.requires_grad = False
        gpt.half()
        add_lora(gpt, args.lora_rank, args.lora_alpha)
    gpt = gpt.to(dev).train()
    critic = None
    if args.pal_weight > 0:
        ck = torch_load(args.critic)
        critic = build_critic(ck['args'])
        critic.load_state_dict(ck['state_dict'])
        critic = critic.to(dev).eval()
        for p in critic.parameters():
            p.requires_grad = False

    params = [p for p in gpt.parameters() if p.requires_grad]
    print('trainable params (M):', sum(p.numel() for p in params) / 1e6, flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))

    tr = ClipDataset(args.data, 'train', ('feats', 'tokens'), limit=args.train_limit)
    dv = ClipDataset(args.data, 'dev', ('feats', 'tokens'), limit=64)
    step, log, t0, best = 0, [], time.time(), 1e9
    run = dict(ce=0.0, pal=0.0, n=0)
    while step < args.steps:
        for b in batches(tr, args.bs, True, args.seed * 100003 + step):
            tokens, feats = b['tokens'].to(dev), b['feats'].to(dev)
            with torch.autocast('mps', dtype=torch.float16, enabled=bool(args.amp)):
                logits = forward_logits(gpt, corrupt(tokens, args.pkeep), feats, args.grad_ckpt)
            ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), tokens.reshape(-1))
            loss = ce
            beta = args.pal_weight * min(1.0, max(0.0, (step - args.pal_start) / max(1, args.pal_ramp)))
            if critic is not None and beta > 0:
                if args.pal_pkeep < 1:
                    # PAL on predictions made from a corrupted prefix, so the model cannot look
                    # aligned by copying the (aligned) ground-truth context.
                    with torch.autocast('mps', dtype=torch.float16, enabled=bool(args.amp)):
                        pal_logits = forward_logits(gpt, corrupt(tokens[:args.pal_bs], args.pal_pkeep),
                                                    feats[:args.pal_bs], args.grad_ckpt)
                else:
                    pal_logits = logits
                pl, _ = pal_term(vq, critic, pal_logits, feats, args, rng)
                loss = loss + beta * pl
                run['pal'] += pl.item()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
            run['ce'] += ce.item(); run['n'] += 1
            step += 1
            if step % args.eval_every == 0 or step == args.steps:
                ev = dict(step=step, train_ce=run['ce'] / run['n'], train_pal=run['pal'] / run['n'],
                          dev_ce=dev_ce(gpt, dv, dev, args), sec=round(time.time() - t0),
                          mps_gb=round(torch.mps.driver_allocated_memory() / 1e9, 2) if dev == 'mps' else 0)
                run = dict(ce=0.0, pal=0.0, n=0)
                log.append(ev); print(json.dumps(ev), flush=True)
                state = {k: v for k, v in gpt.state_dict().items()
                         if args.lora_rank == 0 or k.endswith('.A') or k.endswith('.B')}
                torch.save({'state_dict': state, 'args': vars(args), 'step': step}, out / 'last.pt')
            if step >= args.steps:
                break
    (out / 'log.json').write_text(json.dumps(log, indent=1))


if __name__ == '__main__':
    os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.3')
    os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO', '0.25')
    main()

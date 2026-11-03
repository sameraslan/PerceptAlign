"""SpecVQGAN-compatible models: spectrogram VQGAN and the feature-conditioned GPT.

Parameter names match the public SpecVQGAN checkpoints (Iashin & Rahtu, 2021) so the
pretrained VAS codebook and transformer load directly. Two additions over the original
code: a key/value cache for fast autoregressive sampling, and differentiable decoding of
transformer logits (soft or straight-through) so that losses on the decoded spectrogram
can reach the transformer.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from specvqgan.modules.diffusionmodules.model import Decoder, Encoder

# Spectrogram / token geometry used by SpecVQGAN for 10 s VAS clips.
MEL_BINS, MEL_FRAMES = 80, 848
Z_H, Z_W = 5, 53
N_TOKENS = Z_H * Z_W
FEAT_FRAMES = 212

VQGAN_DDCONFIG = dict(double_z=False, z_channels=256, resolution=848, in_channels=1, out_ch=1,
                      ch=128, ch_mult=[1, 1, 2, 2, 4], num_res_blocks=2, attn_resolutions=[53],
                      dropout=0.0)


def column_major_idx(h=Z_H, w=Z_W):
    """Token order used by SpecVQGAN's ColumnMajor permuter: time-major, 5 codes per column."""
    return torch.tensor(np.arange(h * w).reshape(h, w).T.ravel())


class VQGAN(nn.Module):
    def __init__(self, n_embed=128, embed_dim=256, ddconfig=VQGAN_DDCONFIG):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = nn.Module()
        self.quantize.embedding = nn.Embedding(n_embed, embed_dim)
        self.quant_conv = nn.Conv2d(ddconfig['z_channels'], embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig['z_channels'], 1)
        perm = column_major_idx()
        self.register_buffer('perm', perm, persistent=False)
        self.register_buffer('inv_perm', torch.argsort(perm), persistent=False)

    @property
    def codebook(self):
        return self.quantize.embedding.weight

    @torch.no_grad()
    def encode_to_tokens(self, x):
        """x: (B, 1, 80, 848) in [-1, 1] -> token ids (B, 265) in transformer order."""
        h = self.quant_conv(self.encoder(x))  # (B, 256, 5, 53)
        B, C, H, W = h.shape
        z = h.permute(0, 2, 3, 1).reshape(-1, C)
        d = (z.pow(2).sum(1, keepdim=True) + self.codebook.pow(2).sum(1)
             - 2 * z @ self.codebook.t())
        idx = d.argmin(1).view(B, H * W)
        return idx[:, self.perm]

    def decode_embeddings(self, emb):
        """emb: (B, 265, 256) code vectors in transformer order -> mel (B, 1, 80, 848)."""
        B = emb.shape[0]
        emb = emb[:, self.inv_perm].view(B, Z_H, Z_W, -1).permute(0, 3, 1, 2).contiguous()
        return self.decoder(self.post_quant_conv(emb))

    def decode_tokens(self, idx):
        return self.decode_embeddings(F.embedding(idx, self.codebook))

    def decode_logits(self, logits, mode='st', temperature=1.0):
        """Differentiable decoding of transformer logits (B, 265, V).

        mode='soft': expected code vector under softmax(logits / temperature).
        mode='st':   straight-through: forward uses the argmax code, backward uses the
                     softmax relaxation, so the decoder only ever sees real codebook entries.
        """
        p = F.softmax(logits / temperature, dim=-1)
        if mode == 'soft':
            w = p
        elif mode == 'st':
            hard = F.one_hot(p.argmax(-1), p.shape[-1]).to(p.dtype)
            w = hard + p - p.detach()
        else:
            raise ValueError(mode)
        return self.decode_embeddings(w @ self.codebook)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, cache=None):
        B, T, C = x.shape
        hs = C // self.n_head
        q = self.query(x).view(B, T, self.n_head, hs).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, hs).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, hs).transpose(1, 2)
        if cache is not None:
            if 'k' in cache:
                k = torch.cat([cache['k'], k], dim=2)
                v = torch.cat([cache['v'], v], dim=2)
            cache['k'], cache['v'] = k, v
        if T == 1:
            y = F.scaled_dot_product_attention(q, k, v)
        else:
            assert k.shape[2] == T, 'multi-token steps are only supported without a prefix'
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.mlp = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
                                 nn.Linear(4 * n_embd, n_embd), nn.Dropout(0.0))

    def forward(self, x, cache=None):
        x = x + self.attn(self.ln1(x), cache)
        return x + self.mlp(self.ln2(x))


class GPTFeats(nn.Module):
    """GPT that prefixes projected visual features to the code sequence (SpecVQGAN GPTFeats)."""

    def __init__(self, vocab_size=128, block_size=477, n_layer=24, n_head=16, n_embd=1024,
                 feat_depth=2048):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.embedder = nn.Conv1d(feat_depth, n_embd, kernel_size=1)
        self.block_size = block_size
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            m.weight.data.normal_(0.0, 0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            m.bias.data.zero_()
            m.weight.data.fill_(1.0)

    def _run(self, x, pos0, caches=None):
        x = x + self.pos_emb[:, pos0:pos0 + x.shape[1]]
        for i, blk in enumerate(self.blocks):
            x = blk(x, None if caches is None else caches[i])
        return self.head(self.ln_f(x))

    def forward(self, idx, feats):
        """Teacher-forced logits for all 265 tokens.

        idx: (B, 265) ground-truth tokens, feats: (B, 2048, 212).
        Returns logits (B, 265, V); output i is p(z_i | z_<i, feats).
        """
        c = self.embedder(feats).transpose(1, 2)
        x = torch.cat([c, self.tok_emb(idx[:, :-1])], dim=1)
        logits = self._run(x, 0)
        return logits[:, c.shape[1] - 1:]

    @torch.no_grad()
    def sample(self, feats, steps=N_TOKENS, temperature=1.0, top_k=None, generator=None):
        c = self.embedder(feats).transpose(1, 2)
        caches = [dict() for _ in self.blocks]
        logits = self._run(c, 0, caches)[:, -1]
        pos = c.shape[1]
        out = []
        for t in range(steps):
            logits = logits.float() / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits = logits.masked_fill(logits < v[:, [-1]], float('-inf'))
            probs = F.softmax(logits, dim=-1)
            if generator is not None:
                nxt = torch.multinomial(probs.cpu(), 1, generator=generator).to(probs.device)
            else:
                nxt = torch.multinomial(probs, 1)
            out.append(nxt)
            if t == steps - 1:
                break
            logits = self._run(self.tok_emb(nxt), pos, caches)[:, -1]
            pos += 1
        return torch.cat(out, dim=1)


def load_state(module, state, prefix):
    sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    missing, unexpected = module.load_state_dict(sub, strict=False)
    return missing, unexpected


class _Stub:
    """Stand-in for non-tensor objects pickled into Lightning checkpoints (callbacks, configs)."""

    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        pass


def _restricted_pickle_module():
    import pickle
    import types

    allowed_prefixes = ('torch', 'collections', 'numpy')
    allowed_builtins = {'dict', 'list', 'tuple', 'set', 'int', 'float', 'str', 'bool', 'slice', 'frozenset', 'bytes'}

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.split('.')[0] in allowed_prefixes:
                return super().find_class(module, name)
            if module == 'builtins' and name in allowed_builtins:
                return super().find_class(module, name)
            if module == '_codecs' and name == 'encode':
                return super().find_class(module, name)
            return _Stub

    m = types.ModuleType('restricted_pickle')
    m.Unpickler = Unpickler
    m.load = lambda f, **kw: Unpickler(f, **kw).load()
    m.__name__ = 'pickle'
    return m


def torch_load(path):
    """Load a checkpoint without executing arbitrary pickled code.

    Plain state dicts load with weights_only=True. Old Lightning checkpoints also pickle
    callbacks and configs; those load through an unpickler that only builds torch / numpy /
    container objects and replaces everything else with an inert stub.
    """
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except Exception:
        return torch.load(path, map_location='cpu', weights_only=False,
                          pickle_module=_restricted_pickle_module())

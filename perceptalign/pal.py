"""Perceptual Alignment Loss (PAL).

PAL scores how well an audio stream and a video stream are synchronised. Both streams are
embedded frame by frame by a small sync critic. For every segment of the clip we compute
the audio-visual cross-correlation at each temporal offset o in [-K, K], turn the offsets
into a distribution with a temperature-scaled softmax, and read off the probability mass
that falls inside a Gaussian temporal binding window around zero offset:

    C[s, o]  = mean_{i in segment s} <a_i, v_{i+o}>
    p[s, o]  = softmax_o(C[s, o] / tau)
    WS[s]    = sum_o w_o p[s, o],     w_o = exp(-o^2 / (2 sigma^2))   (w_0 = 1)
    PAL      = -log( mean_s WS[s] )

sigma = 0 reduces the window to the zero-offset probability alone (a pure contrastive
objective over offsets). Cross-correlations are computed only over the central frames
i in [K, T - K), so every offset compares the same number of real (unpadded) frames and
the critic cannot win by exploiting zero padding.

Training happens in two stages. The critic is first fitted on real audio-video pairs by
minimising PAL (the other offsets of the same clip act as negatives). It is then frozen
and PAL is applied to spectrograms decoded from the generator's logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(c_in, c_out, k):
    return nn.Sequential(nn.Conv1d(c_in, c_out, k, padding=k // 2), nn.GroupNorm(8, c_out), nn.GELU())


class StreamEncoder(nn.Module):
    """Local temporal encoder: Conv1D-GN-GELU blocks, a linear projection, L2 normalisation."""

    def __init__(self, c_in, hidden=256, out=128, kernel=5, n_blocks=2):
        super().__init__()
        layers = [conv_block(c_in, hidden, kernel)]
        layers += [conv_block(hidden, hidden, kernel) for _ in range(n_blocks - 1)]
        self.net = nn.Sequential(*layers)
        self.proj = nn.Conv1d(hidden, out, 1)

    def forward(self, x):
        return F.normalize(self.proj(self.net(x)), dim=1)


class SyncCritic(nn.Module):
    """Embeds audio and visual features at the video frame rate.

    audio_repr='mel' gives the audio encoder all 80 mel bins. audio_repr='bands' gives it only
    n_bands coarse band energies (means over equal groups of mel bins), so the critic judges
    when energy changes rather than what the spectrum contains, which leaves far less room for
    a generator to satisfy it with spectrograms that are not actually synchronised.
    """

    def __init__(self, mel_bins=80, feat_depth=2048, mel_pool=4, hidden=256, out=128,
                 kernel=5, n_blocks=2, audio_repr='mel', n_bands=8):
        super().__init__()
        self.mel_pool, self.audio_repr, self.n_bands = mel_pool, audio_repr, n_bands
        a_in = mel_bins if audio_repr == 'mel' else n_bands
        self.audio = StreamEncoder(a_in, hidden, out, kernel, n_blocks)
        self.video = StreamEncoder(feat_depth, hidden, out, kernel, n_blocks)

    def audio_input(self, mel):
        if mel.dim() == 4:
            mel = mel[:, 0]
        if self.audio_repr == 'bands':
            B, M, T = mel.shape
            mel = mel.view(B, self.n_bands, M // self.n_bands, T).mean(2)
        return F.avg_pool1d(mel, self.mel_pool)

    def embed_audio(self, mel):
        """mel: (B, 80, 848) or (B, 1, 80, 848) in [-1, 1] -> (B, D, 212)."""
        return self.audio(self.audio_input(mel))

    def embed_video(self, feats):
        """feats: (B, 2048, 212) -> (B, D, 212)."""
        return self.video(feats)


CRITIC_KEYS = ('kernel', 'n_blocks', 'hidden', 'audio_repr', 'n_bands')


def build_critic(args):
    """Rebuild a critic from the args dict saved with its checkpoint."""
    return SyncCritic(**{k: args[k] for k in CRITIC_KEYS if k in args})


def offset_logits(a, v, K, segments):
    """Cross-correlation of audio (B, D, T) and video (B, D, T) embeddings.

    Returns (B, S, 2K+1); entry [b, s, K + o] is the mean similarity between audio frame i and
    video frame i + o over the frames i of segment s (central region only).
    """
    B, D, T = a.shape
    L = T - 2 * K
    seg_len = L // segments
    L = seg_len * segments
    a_c = a[:, :, K:K + L]
    out = []
    for o in range(-K, K + 1):
        v_o = v[:, :, K + o:K + o + L]
        sim = (a_c * v_o).sum(1)  # (B, L)
        out.append(sim.view(B, segments, seg_len).mean(-1))
    return torch.stack(out, dim=-1)


def binding_window(K, sigma, device=None, sigma_late=None):
    """Window weights over offsets -K..K (w_0 = 1).

    Offset o < 0 means the audio event comes after the visual one (audio late). Human
    observers tolerate audio lag more than audio lead, so sigma_late (if given) sets the width
    on the audio-late side and sigma on the audio-early side.
    """
    o = torch.arange(-K, K + 1, dtype=torch.float32, device=device)
    if sigma <= 0 and not sigma_late:
        return (o == 0).float()
    s = torch.full_like(o, float(sigma))
    if sigma_late is not None:
        s = torch.where(o < 0, torch.full_like(o, float(sigma_late)), s)
    w = torch.exp(-o ** 2 / (2 * s.clamp_min(1e-6) ** 2))
    w[o == 0] = 1.0
    return w


def pal_loss(a, v, K=10, segments=8, tau=0.1, sigma=1.0, reduce='mean_then_log', sigma_late=None):
    """PAL for audio/video embeddings (B, D, T). Returns per-example loss (B,) and p (B, S, 2K+1)."""
    p = F.softmax(offset_logits(a, v, K, segments) / tau, dim=-1)
    ws = (p * binding_window(K, sigma, p.device, sigma_late)).sum(-1)  # (B, S)
    if reduce == 'mean_then_log':
        loss = -torch.log(ws.mean(-1) + 1e-8)
    elif reduce == 'log_then_mean':
        loss = -torch.log(ws + 1e-8).mean(-1)
    else:
        raise ValueError(reduce)
    return loss, p


@torch.no_grad()
def offset_accuracy(p, tol=0):
    """Fraction of segments whose most likely offset is within +-tol frames of zero."""
    K = (p.shape[-1] - 1) // 2
    return ((p.argmax(-1) - K).abs() <= tol).float().mean().item()

"""FlashDot v2: an exactly-labelled synthetic video-to-audio timing dataset.

Each clip is 10 s long. A red filled dot (radius 12 px, random position per
flash) appears on a white 224x224 background during "on" intervals, and a
1 kHz sine (amplitude 0.5, 5 ms linear fade in/out) sounds during exactly the
same intervals. Everything is generated in memory; no mp4 round trip.

Clip sampling (per clip, RNG = np.random.default_rng([global_seed, clip_idx])):
  * n_flashes ~ U{1..6}
  * durations ~ U[0.15, 1.5] s each (redrawn until they fit with the gaps)
  * gaps between flashes >= 0.25 s; all flashes inside [0.3, 9.7] s
  * spare time is split over the n+1 slots with a flat Dirichlet
  * dot centre ~ U{14..209}^2 (integer pixels) per flash

Timing conventions (the ground truth is defined by these, exactly):
  * audio sample n (time n/22050) is "on" iff onset <= n/22050 < offset
  * video frame f has timestamp f/21.5 s and shows the dot iff
    onset <= f/21.5 < offset (frame-start / pts convention)
  * mel frame t has centre time t*256/22050 s (librosa center=True) and is
    "on" iff onset <= t*256/22050 < offset

Mel spectrogram: an exact reproduction of the SpecVQGAN VAS pipeline in
``dataset_prep/extract_mel_spectrogram.py`` (TRANSFORMS):
  1. waveform zero-padded / cut to 220500 samples, float64
  2. |librosa.stft(y, n_fft=1024, hop_length=256, win_length=1024,
     window='hann', center=True, pad_mode='reflect')| ** 1   (magnitude)
     pad_mode='reflect' is the librosa 0.8.0 default that the repo env pins;
     librosa >= 0.10 changed the default to 'constant', so it is set here
     explicitly (only the first/last 2 frames depend on it).
  3. mel_basis = librosa.filters.mel(sr=22050, n_fft=1024, n_mels=80,
     fmin=125, fmax=7600)  (Slaney mel scale, Slaney area norm)
  4. x = max(mel_basis @ spec, 1e-5)
  5. x = (20*log10(x) - 20 + 100) / 100, clipped to [0, 1]
  6. trimmed to the first 860 frames (librosa gives 862 for 220500 samples)
  Result: float in [0, 1], shape [80, 860]; stored as float16.

Visual features: torchvision ResNet-50 with ResNet50_Weights.IMAGENET1K_V2,
fc replaced by Identity, i.e. the 2048-d global-average-pooled output of
layer4, one vector per video frame. Input is the 224x224 RGB frame scaled to
[0, 1] and normalised with ImageNet mean/std (no resize or crop is needed).
Runs on MPS when available. Because a frame is a pure function of whether the
dot is on and where it is, only the unique images are pushed through the
network (one blank frame plus one image per flash) and their features are
copied to every frame that shows them; ``--no-dedupe`` runs every frame.

Outputs (in --out, default work/flashdot/):
  mel.npy             float16 [N, 80, 860]
  rgb_resnet50.npy    float16 [N, 215, 2048]
  activity.npz        mel_activity uint8 [N, 860], video_activity uint8 [N, 215],
                      mel_frame_times [860], video_frame_times [215]
  intervals.json      meta + per clip: split, intervals [(onset_s, offset_s)],
                      dot centres
  splits.json         {"train": [...], "dev": [...], "test": [...]} (indices)
  checks.json         results of the built-in checks
  examples/           3 mp4s with audio + one PNG figure

Usage:
  python perceptalign/flashdot.py --n 2000 --seed 0 --out work/flashdot
  python perceptalign/flashdot.py --n 20 --out /tmp/fd_bench --benchmark
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field

import numpy as np

# Keep MPS memory small on a 16 GB laptop (must be set before torch is imported).
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.15")

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
SR = 22050
DURATION_S = 10.0
N_SAMPLES = 220500
FPS = 21.5
N_VFRAMES = 215
IMG = 224
DOT_RADIUS = 12
DOT_RGB = (255, 0, 0)

TONE_HZ = 1000.0
TONE_AMP = 0.5
FADE_S = 0.005
NOISE_DBFS = -60.0  # RMS of the white noise floor

N_FLASH_MIN, N_FLASH_MAX = 1, 6
DUR_MIN, DUR_MAX = 0.15, 1.5
GAP_MIN = 0.25
T_MIN, T_MAX = 0.3, 9.7

# SpecVQGAN mel parameters (dataset_prep/extract_mel_spectrogram.py)
N_FFT = 1024
HOP = 256
N_MELS = 80
FMIN = 125
FMAX = 7600
MEL_FLOOR = 1e-5
N_MEL_FRAMES = 860

SPLITS = {"train": 1600, "dev": 200, "test": 200}

MEL_FRAME_TIMES = np.arange(N_MEL_FRAMES) * HOP / SR
VIDEO_FRAME_TIMES = np.arange(N_VFRAMES) / FPS


# ----------------------------------------------------------------------------
# Clip specification
# ----------------------------------------------------------------------------
@dataclass
class ClipSpec:
    idx: int
    intervals: list = field(default_factory=list)  # [(onset_s, offset_s)]
    centers: list = field(default_factory=list)    # [(x, y)] per flash
    noise_seed: int = 0


def sample_clip(idx: int, global_seed: int) -> ClipSpec:
    rng = np.random.default_rng([global_seed, idx])
    n = int(rng.integers(N_FLASH_MIN, N_FLASH_MAX + 1))
    span = T_MAX - T_MIN
    while True:
        d = rng.uniform(DUR_MIN, DUR_MAX, n)
        slack = span - d.sum() - (n - 1) * GAP_MIN
        if slack >= 0:
            break
    w = rng.dirichlet(np.ones(n + 1)) * slack
    t = T_MIN + w[0]
    intervals = []
    for i in range(n):
        on, off = t, t + d[i]
        intervals.append((float(on), float(off)))
        t = off + GAP_MIN + w[i + 1]
    lo, hi = DOT_RADIUS + 2, IMG - DOT_RADIUS - 2
    centers = [tuple(int(v) for v in rng.integers(lo, hi, 2)) for _ in range(n)]
    noise_seed = int(rng.integers(0, 2**31 - 1))
    return ClipSpec(idx=idx, intervals=intervals, centers=centers, noise_seed=noise_seed)


def activity(spec: ClipSpec, times: np.ndarray) -> np.ndarray:
    """Binary activity at the given time stamps (onset <= t < offset)."""
    a = np.zeros(len(times), dtype=np.uint8)
    for on, off in spec.intervals:
        a[(times >= on) & (times < off)] = 1
    return a


def flash_index(spec: ClipSpec, times: np.ndarray) -> np.ndarray:
    """Index of the active flash at each time stamp, -1 if none."""
    k = np.full(len(times), -1, dtype=np.int64)
    for i, (on, off) in enumerate(spec.intervals):
        k[(times >= on) & (times < off)] = i
    return k


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------
_YY, _XX = np.mgrid[0:IMG, 0:IMG]


def dot_image(center) -> np.ndarray:
    img = np.full((IMG, IMG, 3), 255, dtype=np.uint8)
    if center is not None:
        cx, cy = center
        mask = (_XX - cx) ** 2 + (_YY - cy) ** 2 <= DOT_RADIUS ** 2
        img[mask] = DOT_RGB
    return img


def render_video(spec: ClipSpec) -> np.ndarray:
    """uint8 [215, 224, 224, 3] RGB frames."""
    k = flash_index(spec, VIDEO_FRAME_TIMES)
    imgs = [dot_image(None)] + [dot_image(c) for c in spec.centers]
    return np.stack([imgs[i + 1] for i in k])


def render_audio(spec: ClipSpec) -> np.ndarray:
    """float32 [220500] mono waveform at 22050 Hz."""
    t = np.arange(N_SAMPLES) / SR
    env = np.zeros(N_SAMPLES)
    for on, off in spec.intervals:
        m = (t >= on) & (t < off)
        env[m] = np.minimum(1.0, np.minimum((t[m] - on) / FADE_S, (off - t[m]) / FADE_S))
    y = TONE_AMP * env * np.sin(2 * np.pi * TONE_HZ * t)
    noise_std = 10 ** (NOISE_DBFS / 20)
    y = y + np.random.default_rng(spec.noise_seed).normal(0.0, noise_std, N_SAMPLES)
    return y.astype(np.float32)


# ----------------------------------------------------------------------------
# Mel (SpecVQGAN VAS pipeline)
# ----------------------------------------------------------------------------
_MEL_BASIS = None


def mel_basis() -> np.ndarray:
    global _MEL_BASIS
    if _MEL_BASIS is None:
        import librosa
        _MEL_BASIS = librosa.filters.mel(sr=SR, n_fft=N_FFT, fmin=FMIN, fmax=FMAX, n_mels=N_MELS)
    return _MEL_BASIS


def mel_spectrogram(wav: np.ndarray) -> np.ndarray:
    """float64 [80, 860] in [0, 1]; identical to the repo's TRANSFORMS."""
    import librosa
    y = np.zeros(N_SAMPLES)
    n = min(len(wav), N_SAMPLES)
    y[:n] = wav[:n]
    spec = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP, pad_mode="reflect"))
    x = mel_basis() @ spec
    x = np.maximum(MEL_FLOOR, x)
    x = (20 * np.log10(x) - 20 + 100) / 100
    x = np.clip(x, 0, 1.0)
    return x[:, :N_MEL_FRAMES]


# ----------------------------------------------------------------------------
# ResNet-50 features
# ----------------------------------------------------------------------------
class FeatureExtractor:
    WEIGHTS = "IMAGENET1K_V2"

    def __init__(self, device=None, batch_size=32):
        import torch
        from torchvision.models import ResNet50_Weights, resnet50
        self.torch = torch
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        m.fc = torch.nn.Identity()
        self.model = m.eval().to(self.device)
        self.batch_size = batch_size
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def __call__(self, imgs: np.ndarray) -> np.ndarray:
        """uint8 [B, 224, 224, 3] -> float32 [B, 2048]."""
        torch = self.torch
        out = []
        with torch.inference_mode():
            for s in range(0, len(imgs), self.batch_size):
                x = torch.from_numpy(np.ascontiguousarray(imgs[s:s + self.batch_size])).to(self.device)
                x = x.permute(0, 3, 1, 2).float() / 255.0
                x = (x - self.mean) / self.std
                out.append(self.model(x).float().cpu().numpy())
        return np.concatenate(out)


def clip_features(specs, fx: FeatureExtractor, dedupe=True, blank_feat=None):
    """float32 [len(specs), 215, 2048]."""
    if not dedupe:
        return np.stack([fx(render_video(s)) for s in specs])
    if blank_feat is None:
        blank_feat = fx(dot_image(None)[None])[0]
    centers = [c for s in specs for c in s.centers]
    flash_feats = fx(np.stack([dot_image(c) for c in centers]))
    out = np.empty((len(specs), N_VFRAMES, 2048), dtype=np.float32)
    j = 0
    for i, s in enumerate(specs):
        table = np.concatenate([blank_feat[None], flash_feats[j:j + len(s.centers)]])
        out[i] = table[flash_index(s, VIDEO_FRAME_TIMES) + 1]
        j += len(s.centers)
    return out


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def tone_mel_bin() -> int:
    import librosa
    freqs = librosa.mel_frequencies(n_mels=N_MELS + 2, fmin=FMIN, fmax=FMAX)[1:-1]
    return int(np.argmin(np.abs(freqs - TONE_HZ)))


def steady_tone_level(k: int) -> float:
    t = np.arange(N_SAMPLES) / SR
    return float(np.median(mel_spectrogram(TONE_AMP * np.sin(2 * np.pi * TONE_HZ * t))[k, 10:-10]))


def f1(pred: np.ndarray, gt: np.ndarray) -> float:
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def red_pixels(frames: np.ndarray) -> np.ndarray:
    r, g, b = frames[..., 0].astype(int), frames[..., 1].astype(int), frames[..., 2].astype(int)
    return ((r > 200) & (g < 60) & (b < 60)).sum(axis=(1, 2))


def repo_mel(wav: np.ndarray, pad_mode: str | None) -> np.ndarray:
    """Run the repo's own TRANSFORMS (dataset_prep/extract_mel_spectrogram.py)."""
    import importlib.util
    import librosa
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "dataset_prep", "extract_mel_spectrogram.py")
    sp = importlib.util.spec_from_file_location("repo_mel_extractor", path)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    y = np.zeros(N_SAMPLES)
    y[:len(wav)] = wav
    if pad_mode is None:
        return mod.TRANSFORMS(y)
    orig = librosa.stft
    mod.librosa.stft = lambda *a, **kw: orig(*a, pad_mode=pad_mode, **kw)
    try:
        return mod.TRANSFORMS(y)
    finally:
        mod.librosa.stft = orig


# ----------------------------------------------------------------------------
# Examples
# ----------------------------------------------------------------------------
def write_mp4(spec: ClipSpec, path: str) -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    from scipy.io import wavfile
    frames = render_video(spec)
    wav = render_audio(spec)
    with tempfile.TemporaryDirectory() as td:
        wpath = os.path.join(td, "a.wav")
        wavfile.write(wpath, SR, (np.clip(wav, -1, 1) * 32767).astype(np.int16))
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{IMG}x{IMG}", "-r", "43/2", "-i", "-",
               "-i", wpath, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
               "-c:a", "aac", "-b:a", "128k", "-t", str(DURATION_S), path]
        subprocess.run(cmd, input=frames.tobytes(), check=True)
    return True


def write_figure(spec: ClipSpec, mel: np.ndarray, path: str, k: int, thr: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "serif"
    frames = render_video(spec)
    va = activity(spec, VIDEO_FRAME_TIMES)
    ma = activity(spec, MEL_FRAME_TIMES)
    on_idx = np.flatnonzero(va)
    pick = [0]
    for on, off in spec.intervals[:3]:
        f_on = int(np.ceil(on * FPS))
        pick += [max(f_on - 1, 0), f_on]
    pick = sorted(set(pick))[:7]
    fig = plt.figure(figsize=(12, 7.5))
    gs = fig.add_gridspec(3, len(pick), height_ratios=[1.1, 1.4, 0.9], hspace=0.45)
    for j, f in enumerate(pick):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(frames[f])
        ax.set_title(f"frame {f}\nt={f / FPS:.3f}s  on={va[f]}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp_ in ax.spines.values():
            sp_.set_color("0.6")
    ax = fig.add_subplot(gs[1, :])
    ax.imshow(mel, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1,
              extent=[-0.5 * HOP / SR, (N_MEL_FRAMES - 0.5) * HOP / SR, -0.5, N_MELS - 0.5])
    ax.axhline(k, color="cyan", lw=0.6, ls="--")
    ax.set_ylabel("mel bin")
    ax.set_title(f"clip {spec.idx}: mel spectrogram [80 x 860] (dashed: 1 kHz bin {k})", fontsize=9)
    ax.set_xlim(0, DURATION_S)
    ax = fig.add_subplot(gs[2, :])
    ax.plot(MEL_FRAME_TIMES, mel[k], color="0.3", lw=0.8, label="mel @ 1 kHz bin")
    ax.axhline(thr, color="0.6", lw=0.6, ls=":", label="threshold")
    ax.step(MEL_FRAME_TIMES, ma * 1.05, where="mid", color="tab:red", lw=1, label="GT mel-frame activity")
    ax.step(VIDEO_FRAME_TIMES, va * 1.12, where="post", color="tab:blue", lw=1, label="GT video-frame activity")
    for on, off in spec.intervals:
        ax.axvspan(on, off, color="tab:red", alpha=0.08)
    ax.set_xlim(0, DURATION_S)
    ax.set_ylim(-0.05, 1.25)
    ax.set_xlabel("time (s)")
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------------
def split_of(i: int, n: int) -> str:
    n_train = round(n * SPLITS["train"] / 2000)
    n_dev = round(n * SPLITS["dev"] / 2000)
    return "train" if i < n_train else ("dev" if i < n_train + n_dev else "test")


def build(n: int, seed: int, out: str, chunk: int = 50, dedupe: bool = True,
          device: str | None = None, batch_size: int = 32, examples: bool = True,
          compare_full: bool = False):
    t0 = time.time()
    os.makedirs(out, exist_ok=True)
    specs = [sample_clip(i, seed) for i in range(n)]
    fx = FeatureExtractor(device=device, batch_size=batch_size)
    blank_feat = fx(dot_image(None)[None])[0]

    mel_mm = np.lib.format.open_memmap(os.path.join(out, "mel.npy"), mode="w+",
                                       dtype=np.float16, shape=(n, N_MELS, N_MEL_FRAMES))
    vis_mm = np.lib.format.open_memmap(os.path.join(out, "rgb_resnet50.npy"), mode="w+",
                                       dtype=np.float16, shape=(n, N_VFRAMES, 2048))
    mel_act = np.stack([activity(s, MEL_FRAME_TIMES) for s in specs])
    vid_act = np.stack([activity(s, VIDEO_FRAME_TIMES) for s in specs])

    k = tone_mel_bin()
    tone_level = steady_tone_level(k)
    thr = tone_level + 20 * np.log10(0.5) / 100  # half-magnitude (-6 dB) point
    pred_act = np.zeros_like(mel_act)
    red_ok = np.zeros(n, dtype=bool)
    feat_ok = np.zeros(n, dtype=bool)
    mel_min, mel_max = np.inf, -np.inf
    t_mel = t_vis = t_chk = 0.0

    for s0 in range(0, n, chunk):
        cs = specs[s0:s0 + chunk]
        ta = time.time()
        for j, s in enumerate(cs):
            m = mel_spectrogram(render_audio(s))
            mel_min, mel_max = min(mel_min, m.min()), max(mel_max, m.max())
            mel_mm[s0 + j] = m.astype(np.float16)
            pred_act[s0 + j] = (m[k] > thr).astype(np.uint8)
        tb = time.time()
        feats = clip_features(cs, fx, dedupe=dedupe, blank_feat=blank_feat)
        vis_mm[s0:s0 + len(cs)] = feats.astype(np.float16)
        tc = time.time()
        for j, s in enumerate(cs):
            i = s0 + j
            drawn = (red_pixels(render_video(s)) > 0).astype(np.uint8)
            red_ok[i] = np.array_equal(drawn, vid_act[i])
            is_blank = np.all(feats[j] == blank_feat[None], axis=1)
            feat_ok[i] = np.array_equal(~is_blank, vid_act[i].astype(bool))
        td = time.time()
        t_mel += tb - ta
        t_vis += tc - tb
        t_chk += td - tc
        print(f"[{s0 + len(cs)}/{n}] mel {t_mel:.1f}s  resnet {t_vis:.1f}s  checks {t_chk:.1f}s", flush=True)
    mel_mm.flush()
    vis_mm.flush()

    np.savez(os.path.join(out, "activity.npz"), mel_activity=mel_act, video_activity=vid_act,
             mel_frame_times=MEL_FRAME_TIMES, video_frame_times=VIDEO_FRAME_TIMES)
    splits = {sp: [i for i in range(n) if split_of(i, n) == sp] for sp in SPLITS}
    with open(os.path.join(out, "splits.json"), "w") as f:
        json.dump(splits, f)
    meta = dict(
        version="flashdot-v2", n=n, global_seed=seed, sr=SR, duration_s=DURATION_S,
        n_samples=N_SAMPLES, fps=FPS, n_video_frames=N_VFRAMES, img=IMG, dot_radius=DOT_RADIUS,
        dot_rgb=DOT_RGB, tone_hz=TONE_HZ, tone_amp=TONE_AMP, fade_s=FADE_S, noise_dbfs_rms=NOISE_DBFS,
        n_flashes=[N_FLASH_MIN, N_FLASH_MAX], flash_dur_s=[DUR_MIN, DUR_MAX], min_gap_s=GAP_MIN,
        flash_window_s=[T_MIN, T_MAX],
        mel=dict(n_fft=N_FFT, hop=HOP, win_length=N_FFT, window="hann", center=True, pad_mode="reflect",
                 power=1, n_mels=N_MELS, fmin=FMIN, fmax=FMAX, mel_scale="slaney", mel_norm="slaney",
                 floor=MEL_FLOOR, norm="(20*log10(max(x,1e-5)) - 20 + 100)/100, clip [0,1]",
                 n_frames=N_MEL_FRAMES, frame_center_s="t*256/22050"),
        video_frame_time_s="f/21.5 (frame start / pts)",
        activity_rule="onset <= t < offset",
        visual=dict(model="torchvision resnet50", weights="ResNet50_Weights.IMAGENET1K_V2",
                    layer="global avg pool (fc=Identity)", dim=2048,
                    normalize=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    dedupe=dedupe, device=str(fx.device)),
        clip_rng="np.random.default_rng([global_seed, clip_idx])",
    )
    clips = [dict(idx=s.idx, split=split_of(s.idx, n), intervals=[[round(a, 6), round(b, 6)] for a, b in s.intervals],
                  intervals_exact=s.intervals, dot_centers_xy=s.centers) for s in specs]
    with open(os.path.join(out, "intervals.json"), "w") as f:
        json.dump(dict(meta=meta, clips=clips), f, indent=1)

    # --- checks --------------------------------------------------------------
    per_clip_f1 = np.array([f1(pred_act[i], mel_act[i]) for i in range(n)])
    wav0 = render_audio(specs[0])
    mine = mel_spectrogram(wav0)
    repo_reflect = repo_mel(wav0, "reflect")
    repo_default = repo_mel(wav0, None)
    checks = dict(
        shapes=dict(mel=list(mel_mm.shape), rgb_resnet50=list(vis_mm.shape),
                    mel_activity=list(mel_act.shape), video_activity=list(vid_act.shape)),
        dtypes=dict(mel=str(mel_mm.dtype), rgb_resnet50=str(vis_mm.dtype),
                    mel_activity=str(mel_act.dtype), video_activity=str(vid_act.dtype)),
        mel_range=[float(mel_min), float(mel_max)],
        mel_vs_repo_transforms_reflect_maxabs=float(np.abs(mine - repo_reflect).max()),
        mel_vs_repo_transforms_librosa_default_pad=dict(
            maxabs_all=float(np.abs(mine - repo_default).max()),
            maxabs_frames_2_to_857=float(np.abs(mine - repo_default)[:, 2:-2].max())),
        tone_mel_bin=k, steady_tone_level=tone_level, threshold=float(thr),
        mel_activity_f1_micro=f1(pred_act.ravel(), mel_act.ravel()),
        mel_activity_f1_per_clip_min=float(per_clip_f1.min()),
        mel_activity_f1_per_clip_mean=float(per_clip_f1.mean()),
        mel_activity_frame_errors=int((pred_act != mel_act).sum()),
        video_red_pixels_match_gt_all=bool(red_ok.all()), video_red_pixels_mismatch_clips=int((~red_ok).sum()),
        resnet_feature_on_off_matches_gt_all=bool(feat_ok.all()),
        split_sizes={k_: len(v) for k_, v in splits.items()},
        n_flashes_hist=np.bincount([len(s.intervals) for s in specs], minlength=7)[1:].tolist(),
        min_gap_s=float(min([b2[0] - a2[1] for s in specs for a2, b2 in zip(s.intervals, s.intervals[1:])] or [np.nan])),
        flash_bounds_s=[float(min(s.intervals[0][0] for s in specs)), float(max(s.intervals[-1][1] for s in specs))],
        time_s=dict(mel=t_mel, resnet=t_vis, checks=t_chk),
    )
    if compare_full:
        m = min(n, 5)
        tf = time.time()
        full = clip_features(specs[:m], fx, dedupe=False)
        checks["full_resnet_frames_per_s"] = m * N_VFRAMES / (time.time() - tf)
        dd = clip_features(specs[:m], fx, dedupe=True, blank_feat=blank_feat)
        checks["dedupe_vs_full_resnet_maxabs"] = float(np.abs(full - dd).max())
        checks["dedupe_vs_full_resnet_max_rel"] = float(np.abs(full - dd).max() / np.abs(full).max())

    if examples:
        ex = os.path.join(out, "examples")
        os.makedirs(ex, exist_ok=True)
        ex_ids = splits["test"][:3] if len(splits["test"]) >= 3 else list(range(min(3, n)))
        checks["example_mp4s"] = [f"clip_{i:04d}.mp4" for i in ex_ids if write_mp4(specs[i], os.path.join(ex, f"clip_{i:04d}.mp4"))]
        write_figure(specs[ex_ids[0]], mel_mm[ex_ids[0]].astype(np.float32), os.path.join(ex, "flashdot_example.png"), k, thr)
    checks["total_time_s"] = time.time() - t0
    with open(os.path.join(out, "checks.json"), "w") as f:
        json.dump(checks, f, indent=1)
    print(json.dumps(checks, indent=1))
    return checks


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="work/flashdot")
    p.add_argument("--chunk", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default=None)
    p.add_argument("--no-dedupe", action="store_true", help="run ResNet on every frame")
    p.add_argument("--no-examples", action="store_true")
    p.add_argument("--benchmark", action="store_true", help="also compare dedupe vs full ResNet pass")
    a = p.parse_args()
    build(a.n, a.seed, a.out, chunk=a.chunk, dedupe=not a.no_dedupe, device=a.device,
          batch_size=a.batch_size, examples=not a.no_examples, compare_full=a.benchmark)


if __name__ == "__main__":
    main()

"""Semantic-quality metrics for generated VAS spectrograms (FID, mean KL, ISc) with Melception.

A standalone port of the SpecVQGAN evaluation (evaluate.py + evaluation/metrics/{fid,kl,isc}.py +
evaluation/feature_extractors/melception.py), with no dependency on the specvqgan/evaluation code.

Input convention (what `melception_features` expects):
    mels: array [N, 80, T] of log-mel spectrograms in the on-disk SpecVQGAN convention, i.e. the
    values stored in the VAS `*_mel.npy` files (roughly [0, 1]); generated samples must be mapped
    from the model's [-1, 1] output with (x + 1) / 2, exactly as evaluation/generate_samples.py
    does before saving (no clamping). T = 848 is used as is; T > 848 (e.g. the raw 860-frame VAS
    files) is centre-cropped to 848 the way the VAS data module does (frames 6:854).
    Before Melception each clip is standardised per mel bin with the VGGSound train statistics
    (StandardNormalizeAudio): x = (mel - mean[f]) / std[f], computed in float64, cast to float32.

Memory: on a 16 GB M1, keep batch_size at 8 to 16 and set PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.15 and
PYTORCH_MPS_LOW_WATERMARK_RATIO=0.1 when running on MPS.

Melception is an Inception-v3 trained on VGGSound (309 classes), not on VAS, so there is no
VAS class accuracy: its logits index the sorted VGGSound label set (`vggsound_classes()`).

Layers returned by `melception_features`:
    '2048'            pool features   -> fid()
    'logits'          fc(x)           -> mkl()   (SpecVQGAN kl_cfg.feat_layer_name = 'logits')
    'logits_unbiased' x @ fc.weight.T -> inception_score()  (SpecVQGAN isc_cfg uses the
                                         bias-free logits, as torch-fidelity does)
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import scipy.linalg
import torch
import torch.nn.functional as F
from torchvision.models.inception import BasicConv2d, Inception3

REPO_ROOT = Path(__file__).resolve().parents[1]
CKPT_PATH = Path(os.environ.get(
    'MELCEPTION_CKPT',
    REPO_ROOT / 'work/eval_models/logs/21-05-10T09-28-40/melception-21-05-10T09-28-40.pt'))
CKPT_MD5 = 'a71a41041e945b457c7d3d814bbcf72d'
CKPT_URL = ('https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/'
            'melception-21-05-10T09-28-40.pt')
# Per-mel-bin mean/std over the VGGSound train set (two columns, 80 rows).
STATS_PATH = REPO_ROOT / 'specvqgan/modules/losses/vggishish/data/train_means_stds_melspec_10s_22050hz.txt'
VGGSOUND_CSV = REPO_ROOT / 'specvqgan/modules/losses/vggishish/data/vggsound.csv'

N_MELS, N_FRAMES, N_CLASSES = 80, 848, 309
KL_EPS = 1e-6


class Melception(Inception3):
    """Inception-v3 with a 1-channel stem and no max pools, as in SpecVQGAN."""

    def __init__(self):
        super().__init__(num_classes=N_CLASSES, init_weights=False)
        self.Conv2d_1a_3x3 = BasicConv2d(1, 32, kernel_size=3, stride=2)
        self.maxpool1 = torch.nn.Identity()
        self.maxpool2 = torch.nn.Identity()

    def forward(self, x):
        x = x.unsqueeze(1)  # B x 1 x 80 x 848
        for name in ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3', 'maxpool1', 'Conv2d_3b_1x1',
                     'Conv2d_4a_3x3', 'maxpool2', 'Mixed_5b', 'Mixed_5c', 'Mixed_5d', 'Mixed_6a',
                     'Mixed_6b', 'Mixed_6c', 'Mixed_6d', 'Mixed_6e', 'Mixed_7a', 'Mixed_7b', 'Mixed_7c',
                     'avgpool', 'dropout']:
            x = getattr(self, name)(x)
        pool = torch.flatten(x, 1)
        # same arithmetic as the original: unbiased logits first, then add the bias
        unbiased = pool.mm(self.fc.weight.T)
        logits = unbiased + self.fc.bias.unsqueeze(0)
        return {'2048': pool, 'logits': logits, 'logits_unbiased': unbiased}


class _Stub:
    """Inert stand-in for the training config objects (omegaconf) pickled next to the weights."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _SafePickle:
    """pickle_module for torch.load that only builds tensors, dicts and numpy scalars; config
    objects stored in the checkpoint become inert _Stub instances, anything else is refused."""
    load, loads, dump, dumps = pickle.load, pickle.loads, pickle.dump, pickle.dumps

    class Unpickler(pickle.Unpickler):
        ALLOWED = {('collections', 'OrderedDict'), ('collections', 'defaultdict'), ('torch._utils', '_rebuild_tensor_v2'),
                   ('torch._utils', '_rebuild_parameter'), ('numpy.core.multiarray', 'scalar'),
                   ('numpy._core.multiarray', 'scalar'), ('numpy', 'dtype'), ('_codecs', 'encode')}
        BUILTINS = {'int', 'long', 'float', 'str', 'bool', 'list', 'dict', 'tuple', 'set', 'object', 'bytes'}

        def find_class(self, module, name):
            if (module, name) in self.ALLOWED or (module == 'torch' and name.endswith('Storage')) \
                    or (module in ('builtins', '__builtin__') and name in self.BUILTINS):
                return super().find_class(module, name)
            if module.split('.')[0] in ('omegaconf', 'typing', 'enum') or (module, name) == ('copyreg', '_reconstructor'):
                return _Stub
            raise pickle.UnpicklingError(f'blocked global {module}.{name}')


def load_melception(device='cpu', ckpt_path=None):
    ckpt_path = Path(ckpt_path or CKPT_PATH)
    if not ckpt_path.exists():
        raise FileNotFoundError(f'{ckpt_path} missing; extract it from eval-logs.tar.gz or get {CKPT_URL}')
    try:
        state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    except pickle.UnpicklingError:
        # the checkpoint also stores numpy scalars (training metadata) under the numpy 1.x name
        # `numpy.core.multiarray.scalar`, which weights_only=True rejects; use an allowlist instead
        state = torch.load(ckpt_path, map_location='cpu', weights_only=False, pickle_module=_SafePickle)
    model = Melception()
    model.load_state_dict(state['model'])
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval().to(device)


def _norm_stats():
    means, stds = np.loadtxt(STATS_PATH).T
    return means.reshape(-1, 1), stds.reshape(-1, 1)


def preprocess(mels):
    """[N, 80, T] on-disk mels -> [N, 80, 848] float32 Melception input."""
    mels = np.asarray(mels)
    if mels.ndim == 2:
        mels = mels[None]
    assert mels.ndim == 3 and mels.shape[1] == N_MELS, f'expected [N, 80, T], got {mels.shape}'
    T = mels.shape[2]
    assert T >= N_FRAMES, f'need at least {N_FRAMES} frames, got {T}'
    start = (T - N_FRAMES) // 2  # albumentations.CenterCrop offset
    mels = mels[:, :, start:start + N_FRAMES].astype(np.float32)
    means, stds = _norm_stats()
    return ((mels.astype(np.float64) - means) / stds).astype(np.float32)


@torch.no_grad()
def melception_features(mels, device='cpu', batch_size=16, model=None):
    """Returns {'2048': [N, 2048], 'logits': [N, 309], 'logits_unbiased': [N, 309]} as float32 CPU tensors."""
    model = model or load_melception(device)
    x = preprocess(mels)
    out = {'2048': [], 'logits': [], 'logits_unbiased': []}
    for i in range(0, len(x), batch_size):
        feats = model(torch.from_numpy(x[i:i + batch_size]).to(device))
        for k, v in feats.items():
            out[k].append(v.float().cpu())
    return {k: torch.cat(v) for k, v in out.items()}


def _np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def fid(feats_fake, feats_real, eps=1e-6):
    """Frechet distance between Gaussians fit to two [N, 2048] feature sets (evaluation/metrics/fid.py)."""
    f1, f2 = _np(feats_fake), _np(feats_real)
    mu1, mu2 = f1.mean(axis=0), f2.mean(axis=0)
    s1, s2 = np.atleast_2d(np.cov(f1, rowvar=False)), np.atleast_2d(np.cov(f2, rowvar=False))
    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(s1.dot(s2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(s1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((s1 + offset).dot(s2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f'imaginary component {np.max(np.abs(covmean.imag))}')
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def mkl(logits_fake, logits_real, fake_to_real=None):
    """Mean over paired clips of KL(p_real || p_fake), from 'logits' (evaluation/metrics/kl.py).

    Row i of logits_fake is paired with row fake_to_real[i] of logits_real (default: row i), so
    several samples per real clip are each compared with that clip, as in the original.
    """
    lf = torch.as_tensor(_np(logits_fake), dtype=torch.float32)
    lr = torch.as_tensor(_np(logits_real), dtype=torch.float32)
    if fake_to_real is None:
        assert len(lf) == len(lr), 'pass fake_to_real when fake and real counts differ'
    else:
        lr = lr[torch.as_tensor(np.asarray(fake_to_real), dtype=torch.long)]
    p_fake, p_real = lf.softmax(dim=1), lr.softmax(dim=1)
    return float(F.kl_div((p_fake + KL_EPS).log(), p_real, reduction='sum') / len(p_fake))


def inception_score(logits_fake, splits=10, shuffle=True, rng_seed=2020):
    """Inception score from 'logits_unbiased' (evaluation/metrics/isc.py). Returns (mean, std) over splits.

    The seeded shuffle permutes the rows as given, so the split assignment (and the score) depends
    on input order; the original saw fakes in FakesFolder order (sorted cls_*/ folder, then file name).
    """
    x = torch.as_tensor(_np(logits_fake))
    N = x.shape[0]
    if shuffle:
        x = x[np.random.RandomState(rng_seed).permutation(N), :]
    x = x.double()
    p, log_p = x.softmax(dim=1), x.log_softmax(dim=1)
    scores = []
    for i in range(splits):
        sl = slice(i * N // splits, (i + 1) * N // splits)
        q = p[sl].mean(dim=0, keepdim=True)
        scores.append((p[sl] * (log_p[sl] - q.log())).sum(dim=1).mean().exp().item())
    return float(np.mean(scores)), float(np.std(scores))


def vggsound_classes():
    """Melception's class order: sorted unique VGGSound labels (index = logit column)."""
    import csv
    with open(VGGSOUND_CSV) as f:
        return sorted({row[2] for row in csv.reader(f)})


def evaluate(mels_fake, mels_real, device='cpu', batch_size=16, fake_to_real=None):
    model = load_melception(device)
    ff = melception_features(mels_fake, device, batch_size, model)
    fr = melception_features(mels_real, device, batch_size, model)
    isc_mean, isc_std = inception_score(ff['logits_unbiased'])
    return {'fid': fid(ff['2048'], fr['2048']),
            'mkl': mkl(ff['logits'], fr['logits'], fake_to_real),
            'isc_mean': isc_mean, 'isc_std': isc_std,
            'n_fake': len(ff['2048']), 'n_real': len(fr['2048'])}


def _default_device():
    if torch.cuda.is_available():
        return 'cuda'
    return 'mps' if torch.backends.mps.is_available() else 'cpu'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--fake', required=True, help='[N, 80, T] .npy of generated mels in [0, 1] convention')
    ap.add_argument('--real', required=True, help='[M, 80, T] .npy of real mels; row i pairs with fake row i')
    ap.add_argument('--fake-to-real', help='optional .npy of int indices pairing each fake row with a real row')
    ap.add_argument('--device', default=_default_device())
    ap.add_argument('--batch-size', type=int, default=16)
    args = ap.parse_args()
    pairing = np.load(args.fake_to_real) if args.fake_to_real else None
    res = evaluate(np.load(args.fake, mmap_mode='r'), np.load(args.real, mmap_mode='r'),
                   args.device, args.batch_size, pairing)
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()

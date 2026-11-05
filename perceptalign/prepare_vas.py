"""Consolidate the public SpecVQGAN VAS features into float16 arrays.

Input: the per-class tarballs from the SpecVQGAN release, extracted under
work/downloads/vas/<class>_<kind>/ (mel spectrograms as *_mel.npy, BN-Inception RGB and
flow features as *.pkl). Output, per split, in work/vas/:
    <split>_mel.npy    float16 (N, 80, 848)    log-mel in [0, 1], centre crop of 860 frames
    <split>_feats.npy  float16 (N, 212, 2048)  RGB ++ flow, padded to 215, centre crop 212
    <split>_labels.npy int64 (N,)              class index (sorted class names)
    <split>_ids.txt                            "<class>/<video id>" per row

Preprocessing reproduces SpecVQGAN's VAS data module (CenterCrop to 848 / 212 frames).
Splits: SpecVQGAN's vas_valid list is our test set; 5% of vas_train (stratified, seeded)
is held out as a dev set for model selection; the rest is train.
"""

import argparse
import io
import pickle
import urllib.request
from pathlib import Path

import numpy as np

CLASSES = sorted(['dog', 'fireworks', 'drum', 'baby', 'gun', 'sneeze', 'cough', 'hammer'])
SPLIT_URL = 'https://raw.githubusercontent.com/v-iashin/SpecVQGAN/main/data/vas_{}.txt'


class NumpyOnlyUnpickler(pickle.Unpickler):
    """Refuses to build anything except numpy arrays, so a tampered pickle cannot run code."""
    ALLOWED = {('numpy.core.multiarray', '_reconstruct'), ('numpy._core.multiarray', '_reconstruct'),
               ('numpy', 'ndarray'), ('numpy', 'dtype'), ('numpy.core.multiarray', 'scalar'),
               ('numpy._core.multiarray', 'scalar')}

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f'blocked {module}.{name}')


def load_pkl(path):
    with open(path, 'rb') as f:
        return NumpyOnlyUnpickler(io.BytesIO(f.read()), encoding='bytes').load()


def find_one(root, cls, kind, vid, suffix):
    hits = list((root / f'{cls}_{kind}').rglob(f'{vid}{suffix}'))
    if len(hits) != 1:
        raise FileNotFoundError(f'{cls}/{vid}{suffix}: {len(hits)} matches')
    return hits[0]


def index_dir(root, cls, kind, suffix):
    return {p.name[:-len(suffix)]: p for p in (root / f'{cls}_{kind}').rglob(f'*{suffix}')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='work/downloads/vas')
    ap.add_argument('--out', default='work/vas')
    ap.add_argument('--dev_frac', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lists = {s: urllib.request.urlopen(SPLIT_URL.format(s)).read().decode().split() for s in ['train', 'valid']}
    rng = np.random.default_rng(args.seed)
    train, dev = [], []
    for c in CLASSES:
        rows = [r for r in lists['train'] if r.split('/')[0] == c]
        perm = rng.permutation(len(rows))
        n_dev = int(round(args.dev_frac * len(rows)))
        dev += [rows[i] for i in sorted(perm[:n_dev])]
        train += [rows[i] for i in sorted(perm[n_dev:])]
    splits = {'train': train, 'dev': dev, 'test': lists['valid']}

    idx = {}
    for c in CLASSES:
        idx[c] = dict(mel=index_dir(src, c, 'melspec_10s_22050hz', '_mel.npy'),
                      rgb=index_dir(src, c, 'feature_rgb_bninception_dim1024_21.5fps', '.pkl'),
                      flow=index_dir(src, c, 'feature_flow_bninception_dim1024_21.5fps', '.pkl'))
        print(c, {k: len(v) for k, v in idx[c].items()})

    for split, rows in splits.items():
        N = len(rows)
        mel = np.lib.format.open_memmap(out / f'{split}_mel.npy', 'w+', np.float16, (N, 80, 848))
        feats = np.lib.format.open_memmap(out / f'{split}_feats.npy', 'w+', np.float16, (N, 212, 2048))
        labels = np.zeros(N, np.int64)
        for n, row in enumerate(rows):
            c, vid = row.split('/')
            m = np.load(idx[c]['mel'][vid], allow_pickle=False)
            assert m.shape == (80, 860), (row, m.shape)
            mel[n] = m[:, 6:6 + 848]
            f = np.concatenate([load_pkl(idx[c]['rgb'][vid]), load_pkl(idx[c]['flow'][vid])], axis=1)
            pad = np.zeros((215, 2048), np.float32)
            pad[:min(215, len(f))] = f[:215]
            feats[n] = pad[1:1 + 212]
            labels[n] = CLASSES.index(c)
            if n % 1000 == 0:
                print(split, n, N, flush=True)
        mel.flush(); feats.flush()
        np.save(out / f'{split}_labels.npy', labels)
        (out / f'{split}_ids.txt').write_text('\n'.join(rows) + '\n')
        print(split, N, 'done')


if __name__ == '__main__':
    main()

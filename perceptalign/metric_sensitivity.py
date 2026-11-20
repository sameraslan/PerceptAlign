"""How do candidate metrics respond to a known timing error?

Each ground-truth spectrogram is compared with (a) a copy of itself delayed by a known
offset and (b) a different clip of the same class (right kind of sound, unrelated timing).
A timing metric should degrade monotonically with the offset and should rank the unrelated
clip as badly aligned. Output: work/results/metric_sensitivity_<dataset>.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from perceptalign import timing_metrics as tm

SHIFTS_MS = [0, 23, 46, 93, 186, 372, 743]


def delay(mel, frames):
    """Delay by `frames` mel frames, filling the start with the clip's quietest column."""
    if frames == 0:
        return mel.copy()
    fill = mel[:, [int(np.argmin(mel.mean(0)))]]
    return np.concatenate([np.repeat(fill, frames, 1), mel[:, :-frames]], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--name', required=True)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    root = Path(args.data)
    mel = np.load(root / f'{args.split}_mel.npy', mmap_mode='r')
    labels = np.load(root / f'{args.split}_labels.npy')
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(len(mel), min(args.n, len(mel)), replace=False))
    res = {}
    conds = [(f'delay_{ms}ms', ms) for ms in SHIFTS_MS] + [('other_clip_same_class', None)]
    for name, ms in conds:
        rows = []
        for i in idx:
            gt = np.asarray(mel[i], np.float32)
            if ms is None:
                same = np.flatnonzero((labels == labels[i]) & (np.arange(len(labels)) != i))
                cand = np.asarray(mel[rng.choice(same)], np.float32)
            else:
                cand = delay(gt, int(round(ms / 1000 * tm.FPS_MEL)))
            rows.append(tm.clip_scores(cand, gt, with_dtw=True))
        res[name] = tm.aggregate(rows)
        print(name, json.dumps({k: round(v, 4) for k, v in res[name].items()}), flush=True)
    out = Path('work/results'); out.mkdir(parents=True, exist_ok=True)
    (out / f'metric_sensitivity_{args.name}.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

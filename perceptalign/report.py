"""Collect metrics_*.json files into one table (mean +- std over training seeds).

Usage: python -m perceptalign.report --runs 'work/runs/fd_*' --split dev --out work/results/fd_dev.csv
Runs that share a name up to the trailing _s<seed> are grouped.
"""

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

KEYS = ['act_f1', 'onset_f1', 'onset_ap', 'env_corr', 'abs_lag_ms', 'lag_within_1vf',
        'sync_acc_tol0', 'sync_acc_tol1', 'sync_abs_offset_ms', 'mel_l1', 'fid', 'mkl', 'isc']


def collect(pattern, split, tag=''):
    groups = defaultdict(list)
    for d in sorted(glob.glob(pattern)):
        for f in glob.glob(f'{d}/metrics_{split}_s0{tag}.json'):
            name = re.sub(r'_s\d+$', '', Path(d).name)
            groups[name].append(json.loads(Path(f).read_text()))
    return groups


def summarise(groups):
    rows = []
    for name, ms in groups.items():
        row = {'run': name, 'n_seeds': len(ms)}
        for k in KEYS:
            v = [m[k] for m in ms if k in m]
            if v:
                row[k] = float(np.mean(v))
                row[k + '_std'] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--split', default='dev')
    ap.add_argument('--tag', default='')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    rows = summarise(collect(args.runs, args.split, args.tag))
    cols = [k for k in KEYS if any(k in r for r in rows)]
    print('run'.ljust(28), 'n', *[c[:10].rjust(11) for c in cols])
    for r in rows:
        print(r['run'][:28].ljust(28), r['n_seeds'],
              *[(f"{r[c]:.3f}±{r[c + '_std']:.3f}" if c in r else '-').rjust(11) for c in cols])
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()

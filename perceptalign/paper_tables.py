"""Generate the paper's result tables straight from the result json files.

Every number in paper/generated/*.tex is read from a metrics_*.json, a rerank_ablation_*.json or a
significance json; paper/generated/sources.md lists the file(s) and key behind every cell. Missing
values render as \\placeholder{}, so the script can be rerun as results land.

  PYTHONPATH=. python -m perceptalign.paper_tables            # writes paper/generated/
  PYTHONPATH=. python -m perceptalign.paper_tables --out /tmp/x --sig_vas work/results/sig_vas_test.json

Outputs: tab_flashdot.tex (tab:flashdot), tab_vas.tex (tab:vas), tab_window.tex (tab:window),
bestofn.json (best-of-N curves for a figure) and sources.md (audit trail).

Conventions
  metrics files   <run>/metrics_<split>_s<sampleseed>[_n<N>][_rr<k>][_<6-hex hash>].json; only sample
                  seed 0 and full splits (no _n) are used. Several files for the same (split, rr) that
                  differ only by hash: the newest by mtime wins and a warning lists the others.
  training seeds  run dirs X (seed 0), X_s1, X_s2 (the teacher-forced PAL run: fd_pal_s0/_s1/_s2).
                  Cells show mean +- sample std (ddof 1) over the seeds found; a dagger marks a single
                  seed and a double dagger two seeds.
  rerank jsons    seeds are identified by the json's 'run' field (the --run path); older jsons
                  without it fall back to the tag hash, with a warning.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

PH = r'\placeholder{}'
METRIC_RE = re.compile(r'^metrics_(?P<split>[a-z]+)_s(?P<ss>\d+)(?:_n(?P<n>\d+))?(?:_rr(?P<rr>\d+))?'
                       r'(?:_(?P<hash>[0-9a-f]{6}))?\.json$')
MARK = {1: r'$^{\dagger}$', 2: r'$^{\ddagger}$'}
MARK_TEXT = {1: r'$^{\dagger}$\,single training seed', 2: r'$^{\ddagger}$\,two training seeds'}
STD_MACRO = r'\providecommand{\std}[1]{\,{\scriptsize$\pm$#1}}'
# tab:vas is too wide for inline stds: the std goes on a second line of the cell
STACK_MACRO = r'\providecommand{\stackstd}[2]{\begin{tabular}[t]{@{}c@{}}#1\\[-2pt]{\scriptsize #2}\end{tabular}}'
# significance.py / rerank_ablation.py per-clip names for the metrics_*.json keys
PERCLIP = dict(sync_acc_tol0='sync_tol0', sync_acc_tol1='sync_tol1')

WARNINGS = []


def warn(msg):
    WARNINGS.append(msg)
    print('WARNING:', msg, file=sys.stderr)


def rel(p):
    try:
        return str(Path(p).resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(p)


# ----------------------------------------------------------------------------- loading


def find_metrics(d, split, rr=None):
    """Newest metrics_<split>_s0[_rr<rr>][_hash].json in run dir d (no _n), or None."""
    d = Path(d)
    if not d.is_dir():
        return None
    hits = []
    for f in d.glob(f'metrics_{split}_s0*.json'):
        m = METRIC_RE.match(f.name)
        if not m or m['split'] != split or m['ss'] != '0' or m['n'] is not None:
            continue
        if (int(m['rr']) if m['rr'] else None) != rr:
            continue
        hits.append(f)
    if not hits:
        return None
    hits.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    if len(hits) > 1:
        warn(f'{rel(d)}: {len(hits)} files for split={split} rr={rr}; using {hits[0].name} (newest), '
             f'ignoring {", ".join(h.name for h in hits[1:])}')
    return hits[0]


def seed_dirs(base, runs, seed0_suffix=False, seeds=(0, 1, 2)):
    """[(seed, dir)]: X, X_s1, X_s2 (or X_s0, X_s1, X_s2 with seed0_suffix)."""
    out = []
    for s in seeds:
        name = f'{base}_s{s}' if (s or seed0_suffix) else base
        out.append((s, Path(runs) / name))
    return out


class Cell:
    """Mean +- std of one key over the files that have it."""

    def __init__(self, key, files):
        self.key = key
        self.vals, self.srcs = [], []
        for f in files:
            if f is None:
                continue
            m = json.loads(Path(f).read_text())
            v = m.get(key)
            if v is not None and np.isfinite(v):
                self.vals.append(float(v))
                self.srcs.append(rel(f))
        self.n = len(self.vals)
        self.mean = float(np.mean(self.vals)) if self.n else None
        self.std = float(np.std(self.vals, ddof=1)) if self.n > 1 else None


def row_files(spec, runs):
    """Metric files of one table row: list of paths (None where missing) and expected patterns."""
    kind = spec['kind']
    if kind == 'fixed':
        dirs = [Path(spec['dir'])]
    elif kind == 'seed0':
        dirs = [Path(runs) / spec['base']]
    else:
        dirs = [d for _, d in seed_dirs(spec['base'], runs, spec.get('seed0_suffix', False))]
    files = [find_metrics(d, spec['split'], spec.get('rr')) for d in dirs]
    rr = f'_rr{spec["rr"]}' if spec.get('rr') else ''
    patterns = [f'{rel(d)}/metrics_{spec["split"]}_s0{rr}[_<hash>].json' for d in dirs]
    return files, patterns


# ----------------------------------------------------------------------------- formatting


def fmt_num(x, kind, std=False):
    if kind == 'f3':
        return f'{x:.3f}'
    if kind == 'ms':  # integers, one decimal below 10 ms (reference rows)
        return f'{x:.1f}' if (abs(x) < 10 and not std) or (std and abs(x) < 1) else f'{x:.0f}'
    if kind == 'pct':  # one decimal, but 100% and 0% as integers
        v = f'{100 * x:.1f}'
        v = v[:-2] if v in ('100.0', '0.0') and not std else v
        return v + ('' if std else r'\%')
    if kind == 'f1':
        return f'{x:.1f}'
    if kind == 'f2':
        return f'{x:.2f}'
    raise ValueError(kind)


def fmt_parts(c, kind, row_n=None, star=False):
    """(mean string with markers, std string or None)."""
    if c is None or c.mean is None:
        return PH, None
    s = fmt_num(c.mean, kind)
    if star:
        s += r'$^{*}$'
    if row_n is not None and c.n < row_n and c.n in MARK:
        s += MARK[c.n]
    return s, (fmt_num(c.std, kind, std=True) if c.std is not None else None)


def fmt_cell(c, kind, row_n=None, star=False):
    """Inline 'mean \\std{sd}' (FlashDot and window tables)."""
    m, sd = fmt_parts(c, kind, row_n, star)
    return m if sd is None else m + r'\std{' + sd + '}'


class Sources:
    def __init__(self):
        self.lines = []

    def table(self, title):
        self.lines += ['', f'## {title}', '', '| Row | Column | Value | Source file(s) | Key |', '|---|---|---|---|---|']

    def add(self, row, col, value, files, key, note=''):
        row, col, value = (x.replace('|', r'\|') for x in (row, col, value))
        f = '<br>'.join(files) if files else '(missing)'
        self.lines.append(f'| {row} | {col} | `{value}` | {f} | `{key}`{(" " + note) if note else ""} |')


def latex_label(s):
    """Row label for sources.md (strip the most common LaTeX)."""
    s = re.sub(r'\\quad\s*', '  ', s)
    s = s.replace(r'$p_{\mathrm{keep}} = 0.5$', 'p_keep = 0.5').replace(r'\uparrow', '(higher better)')
    s = s.replace(r'\downarrow', '(lower better)').replace(r'\mathrm{late}', 'late').replace('{', '').replace('}', '')
    s = s.replace('$', '')
    return s.replace('\\', '')


# ----------------------------------------------------------------------------- tab:flashdot

FD_COLS = [('Tone F1 $\\uparrow$', 'act_f1', 'f3'),
           ('$|$Lag$|$ (ms) $\\downarrow$', 'abs_lag_ms', 'ms'),
           ('Within 1 frame $\\uparrow$', 'lag_within_1vf', 'f3'),
           ('Sync acc.\\ $\\uparrow$', 'sync_acc_tol1', 'f3')]


def fd_rows(work):
    ref = work / 'flashdot_std' / 'ref_recon'
    return [
        [dict(label='VQGAN reconstruction', split='dev', kind='fixed', dir=ref),
         dict(label='VQGAN reconstruction', split='test', kind='fixed', dir=ref)],
        [dict(label='CE (teacher forcing)', split='dev', kind='seeds', base='fd2_ce'),
         dict(label='CE, corrupted prefix ($p_{\\mathrm{keep}} = 0.5$)', split='dev', kind='seeds', base='fd2_ce_pk05'),
         dict(label='CE + PAL, teacher-forced (3000 steps)', split='dev', kind='seeds', base='fd_pal', seed0_suffix=True),
         dict(label='CE + PAL, half-corrupted PAL prefix', split='dev', kind='seeds', base='fd2_pal_pp05'),
         dict(label='CE + PAL, fully corrupted PAL prefix', split='dev', kind='seed0', base='fd2_pal_pp0')],
        [dict(label='CE, corrupted prefix, best of 8', split='dev', kind='seed0', base='fd2_ce_pk05', rr=8)],
        [dict(label='CE, corrupted prefix', split='test', kind='seeds', base='fd2_ce_pk05'),
         dict(label='\\quad best of 8', split='test', kind='seeds', base='fd2_ce_pk05', rr=8),
         dict(label='\\quad self-training, random selection', split='test', kind='seeds', base='fd_st_rand'),
         dict(label='\\quad self-training, PAL selection', split='test', kind='seeds', base='fd_st_pal')],
    ]


def seed_mark(spec, files):
    """Row marker for rows trained with seeds: number of seed files found, if fewer than 3."""
    if spec['kind'] == 'fixed':
        return '', None
    n = sum(f is not None for f in files)
    return (MARK.get(n, '') if n else ''), (n or None)


def n_seeds_note(used, max_n):
    parts = []
    if max_n and max_n > 1:
        word = {1: 'one', 2: 'two', 3: 'three'}.get(max_n, str(max_n))
        parts.append(f'Mean $\\pm$ std over {word} training seed{"s" if max_n > 1 else ""}')
    parts += [MARK_TEXT[k] for k in sorted(used)]
    return '; '.join(parts) + '.' if parts else ''


def tab_flashdot(work, runs, src):
    src.table('tab:flashdot (tab_flashdot.tex)')
    body, used, max_n = [], set(), 0
    for bi, block in enumerate(fd_rows(work)):
        if bi:
            body.append('    \\midrule')
        for spec in block:
            files, patterns = row_files(spec, runs)
            mark, row_n = seed_mark(spec, files)
            if spec['kind'] == 'seeds' and row_n:
                max_n = max(max_n, row_n)
            if mark:
                used.add(row_n)
            cells = []
            for col, key, kind in FD_COLS:
                c = Cell(key, files)
                if c.n and c.n < (row_n or 0):
                    used.add(c.n)
                v = fmt_cell(c, kind, row_n)
                cells.append(v)
                src.add(f'{latex_label(spec["label"])} ({spec["split"]})', latex_label(col), v,
                        c.srcs or [f'missing: {p}' for p in patterns], key,
                        f'mean of {c.n} seeds' if c.n > 1 else '')
            label = spec['label'] + mark
            body.append(f'    {label:<52s} & {spec["split"]:<4s} & ' + ' & '.join(cells) + ' \\\\')
    note = n_seeds_note(used, max_n)
    caption = ('FlashDot. Top: references. Middle: PAL as a loss and the CE baselines, trained from scratch '
               '(development split, 2000 steps unless stated). Bottom: PAL as a reward. Sync acc.\\ is the held-out '
               'critic at tolerance 1 (chance $\\approx 0.14$); within 1 frame is the share of clips with $|$lag$|$ '
               'within one video frame. ' + (note + ' ' if note else '') + 'Best of 8 on the development split uses training seed 0.')
    return '\n'.join([
        '% Generated by perceptalign/paper_tables.py; do not edit by hand (sources: paper/generated/sources.md).',
        STD_MACRO,
        '\\begin{table}[t]', '  \\centering', f'  \\caption{{{caption}}}', '  \\label{tab:flashdot}', '  \\small',
        '  \\begin{tabular}{llcccc}', '    \\toprule',
        '    Model & Split & ' + ' & '.join(c[0] for c in FD_COLS) + ' \\\\', '    \\midrule',
        *body, '    \\bottomrule', '  \\end{tabular}', '\\end{table}', ''])


# ----------------------------------------------------------------------------- tab:vas

VAS_COLS = [('On.\\ F1 $\\uparrow$', ['onset_f1'], 'f3'),
            ('On.\\ AP $\\uparrow$', ['onset_ap'], 'f3'),
            ('Env.\\ $\\uparrow$', ['env_corr'], 'f3'),
            ('$|$Lag$|$ $\\downarrow$', ['abs_lag_ms'], 'ms'),
            ('1 fr.\\ $\\uparrow$', ['lag_within_1vf'], 'pct'),
            ('Sync acc.\\ $\\uparrow$', ['sync_acc_tol0', 'sync_acc_tol1'], 'f3'),
            ('FID $\\downarrow$', ['fid'], 'f1'),
            ('MKL $\\downarrow$', ['mkl'], 'f2')]


def load_sig(path, a='rand', b='pal'):
    """{perclip metric: (p_boot, diff of pal - rand)} from a significance.py json, or None."""
    path = Path(path)
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    if f'{a}:{b}' in d:
        return {m: (r.get('p_boot'), r.get('diff')) for m, r in d[f'{a}:{b}'].items() if r.get('n')}
    if f'{b}:{a}' in d:
        return {m: (r.get('p_boot'), -r['diff'] if r.get('diff') is not None else None)
                for m, r in d[f'{b}:{a}'].items() if r.get('n')}
    warn(f'{rel(path)} has no "{a}:{b}" or "{b}:{a}" pair (has {", ".join(d)}); no significance marks')
    return None


def tab_vas(work, runs, src, sig_path, alpha):
    src.table('tab:vas (tab_vas.tex)')
    sig = load_sig(sig_path)
    vas = work / 'vas'
    blocks = [
        [dict(label='Ground truth', split='test', kind='fixed', dir=vas / 'ref_gt', only=['sync_acc_tol0', 'sync_acc_tol1']),
         dict(label='VQGAN reconstruction', split='test', kind='fixed', dir=vas / 'ref_recon')],
        [dict(label='Pretrained SpecVQGAN', split='test', kind='fixed', dir=runs / 'vas_base'),
         dict(label='\\quad best of 4', split='test', kind='fixed', dir=runs / 'vas_base', rr=4),
         dict(label='\\quad GT-only fine-tune', split='test', kind='seeds', base='vas_st_gt'),
         dict(label='\\quad self-train, random', split='test', kind='seeds', base='vas_st_rand'),
         dict(label='\\quad self-train, PAL', split='test', kind='seeds', base='vas_st_pal', sig=True)],
    ]
    body, used, max_n, starred = [], set(), 0, False
    for bi, block in enumerate(blocks):
        if bi:
            body.append('    \\midrule')
        for spec in block:
            files, patterns = row_files(spec, runs)
            mark, row_n = seed_mark(spec, files)
            if spec['kind'] == 'seeds' and row_n:
                max_n = max(max_n, row_n)
            if mark:
                used.add(row_n)
            if max_n <= 1:
                mark = ''  # all VAS rows are single runs; the caption says so
            cells = []
            for col, keys, kind in VAS_COLS:
                parts, sds = [], []
                for key in keys:
                    if 'only' in spec and key not in spec['only']:
                        parts.append('--')
                        sds.append(None)
                        src.add(latex_label(spec['label']), f'{latex_label(col)} [{key}]', '--', [], key, '(not applicable)')
                        continue
                    c = Cell(key, files)
                    if c.n and c.n < (row_n or 0):
                        used.add(c.n)
                    star, note = False, f'mean of {c.n} seeds' if c.n > 1 else ''
                    if spec.get('sig') and sig is not None:
                        pk = PERCLIP.get(key, key)
                        if pk in sig and sig[pk][0] is not None:
                            p, diff = sig[pk]
                            star = p < alpha
                            starred |= star
                            note += f'; sig {rel(sig_path)} :: rand:pal/{pk}/p_boot = {p:.4g} (diff {diff:+.4g})'
                    m, sd = fmt_parts(c, kind, row_n, star)
                    parts.append(m)
                    sds.append(sd)
                    v = m if sd is None else f'{m} +- {sd}'
                    src.add(latex_label(spec['label']), f'{latex_label(col)} [{key}]', v,
                            c.srcs or [f'missing: {p}' for p in patterns], key, note.lstrip('; '))
                if parts and all(p == '--' for p in parts):
                    cells.append('--')
                else:
                    cell = ' / '.join(parts)
                    if any(sd is not None for sd in sds):
                        cell = (r'\stackstd{' + cell + '}{' +
                                ' / '.join(r'$\pm$' + sd if sd else '' for sd in sds) + '}')
                    cells.append(cell)
            body.append(f'    {spec["label"] + mark:<28s} & ' + ' & '.join(cells) + ' \\\\')
    note = n_seeds_note(used, max_n) if max_n > 1 else 'Fine-tuned rows are single training runs (seed 0).'
    sig_note = (f' $^{{*}}$\\,PAL selection differs from random selection with $p < {alpha:g}$ (paired bootstrap over clips).'
                if starred else '')
    caption = ('VAS test results (768 clips of the SpecVQGAN \\texttt{vas\\_valid} list), one sample per clip unless '
               'stated. Sync acc.\\ is the held-out critic at tolerance 0 / 1 (chance $\\approx 0.05$ / $0.14$). '
               'On.: onset; Env.: envelope correlation; $|$Lag$|$ in ms; 1 fr.: share of clips with $|$lag$|$ within one '
               'video frame. Fine-tuned rows train LoRA adapters with the same budget. ' + note + sig_note).strip()
    return '\n'.join([
        '% Generated by perceptalign/paper_tables.py; do not edit by hand (sources: paper/generated/sources.md).',
        STACK_MACRO,
        '\\begin{table}[t]', '  \\centering', f'  \\caption{{{caption}}}', '  \\label{tab:vas}', '  \\small',
        '  \\setlength{\\tabcolsep}{3.2pt}', '  \\begin{tabular}{lcccccccc}', '    \\toprule',
        '    Model & ' + ' & '.join(c[0] for c in VAS_COLS) + ' \\\\', '    \\midrule',
        *body, '    \\bottomrule', '  \\end{tabular}', '\\end{table}', ''])


# ----------------------------------------------------------------------------- rerank ablation


def load_ablation(d, dataset):
    """{generator key: (path, json)} for rerank_ablation_test_*.json in d, one per generator run."""
    d = Path(d)
    files = sorted(d.glob('rerank_ablation_test_*.json'), key=lambda f: f.stat().st_mtime, reverse=True)
    groups = {}
    for f in files:
        j = json.loads(f.read_text())
        if j.get('run'):
            k = Path(j['run']).name or j['run']
        else:
            k = 'hash_' + j.get('tag', f.stem).rsplit('_', 1)[-1]
            warn(f'{rel(f)} has no "run" field; identifying its generator by the tag hash ({k})')
        groups.setdefault(k, []).append((f, j))
    out = {}
    for k, lst in groups.items():
        if len(lst) > 1:
            warn(f'{dataset} ablation: {len(lst)} jsons for generator {k}; using {lst[0][0].name} (newest), '
                 f'ignoring {", ".join(f.name for f, _ in lst[1:])}')
        out[k] = lst[0]
    ns = {j['n'] for _, j in out.values()}
    if len(ns) > 1:
        warn(f'{dataset} ablation: generators were evaluated on different clip counts {sorted(ns)}')
    return out


def abl_cell(abl, sel, key):
    """Cell-like mean +- std over generators of means[sel][key]."""
    c = Cell.__new__(Cell)
    c.key, c.vals, c.srcs = key, [], []
    for g, (f, j) in sorted(abl.items()):
        v = j.get('means', {}).get(sel, {}).get(key)
        if v is not None and np.isfinite(v):
            c.vals.append(float(v))
            c.srcs.append(f'{rel(f)} [{g}]')
    c.n = len(c.vals)
    c.mean = float(np.mean(c.vals)) if c.n else None
    c.std = float(np.std(c.vals, ddof=1)) if c.n > 1 else None
    return c


WINDOWS = [('s0', '$\\sigma = 0$'), ('s0.5', '$\\sigma = 0.5$'), ('s1', '$\\sigma = 1$'), ('s2', '$\\sigma = 2$'),
           ('a0.5-2', '$\\sigma = 0.5$, $\\sigma_{\\mathrm{late}} = 2$'),
           ('a2-0.5', '$\\sigma = 2$, $\\sigma_{\\mathrm{late}} = 0.5$')]
WIN_COLS = [('fd', 'Tone F1 $\\uparrow$', 'act_f1', 'f3'),
            ('fd', '1 fr.\\ $\\uparrow$', 'lag_within_1vf', 'f3'),
            ('fd', 'Sync $\\uparrow$', 'sync_tol1', 'f3'),
            ('vas', '1 fr.\\ $\\uparrow$', 'lag_within_1vf', 'pct'),
            ('vas', '$|$Lag$|$ $\\downarrow$', 'abs_lag_ms', 'ms'),
            ('vas', 'Sync $\\uparrow$', 'sync_tol1', 'f3'),
            ('vas', 'FID $\\downarrow$', 'fid', 'f1')]


def n_cand_of(abl, default=8):
    cs = {j.get('n_cand') for _, j in abl.values()}
    return cs.pop() if len(cs) == 1 else default


def tab_window(runs, src):
    src.table('tab:window (tab_window.tex)')
    abl = dict(fd=load_ablation(runs / 'rr_ablation_fd', 'FlashDot'),
               vas=load_ablation(runs / 'rr_ablation_vas', 'VAS'))
    C = {k: n_cand_of(v) for k, v in abl.items()}
    if C['fd'] != C['vas']:
        warn(f'ablation candidate counts differ: FlashDot {C["fd"]}, VAS {C["vas"]}')
    nc = C['fd'] if abl['fd'] else C['vas']
    rows = [('One sample', {'fd': 'first', 'vas': 'first'})]
    rows += [(f'Best of {nc}, {lab}', {d: f'best{C[d]}_{w}' for d in abl}) for w, lab in WINDOWS]
    rows += [None, ('Oracle (upper bound)', {'fd': 'oracle_actf1', 'vas': 'oracle_lag'})]
    n_fd = len(abl['fd'])
    body = []
    for r in rows:
        if r is None:
            body.append('    \\midrule')
            continue
        label, sels = r
        cells = []
        for d, col, key, kind in WIN_COLS:
            c = abl_cell(abl[d], sels[d], key)
            v = fmt_cell(c, kind, n_fd if d == 'fd' else None)
            cells.append(v)
            src.add(latex_label(label), f'{"FlashDot" if d == "fd" else "VAS"} {latex_label(col)}', v,
                    c.srcs or [f'missing: work/runs/rr_ablation_{d}/rerank_ablation_test_*.json'],
                    f'means/{sels[d]}/{key}', f'mean over {c.n} generator seeds' if c.n > 1 else '')
        body.append(f'    {label:<52s} & ' + ' & '.join(cells) + ' \\\\')
    nfd_clips = {j['n'] for _, j in abl['fd'].values()}
    nvas_clips = {j['n'] for _, j in abl['vas'].values()}
    fd_n = str(nfd_clips.pop()) if len(nfd_clips) == 1 else PH
    vas_n = str(nvas_clips.pop()) if len(nvas_clips) == 1 else PH
    word = {1: 'one', 2: 'two', 3: 'three'}.get(n_fd, str(n_fd) if n_fd else PH)
    fd_mark = f' {MARK_TEXT[n_fd]}.' if n_fd in MARK else ''
    caption = (f'Candidate selection on the test split: every row chooses among the same {nc} free-running samples per '
               f'clip. FlashDot: corruption-trained CE generator, {fd_n} clips, ' +
               (f'mean $\\pm$ std over {word} generator seeds.' if n_fd and n_fd > 1 else 'one generator seed.') +
               f' VAS: pretrained SpecVQGAN, {vas_n} clips, one run; its FID uses these clips only and is not comparable with \\cref{{tab:vas}}. $\\sigma$: width of the binding window '
               f'on the audio-early side, $\\sigma_{{\\mathrm{{late}}}}$ on the audio-late side (\\cref{{eq:window}}). '
               f'Sync: held-out critic at tolerance 1. The oracle keeps the candidate with the best true tone F1 '
               f'(FlashDot) or smallest $|$lag$|$ (VAS).')
    tex = '\n'.join([
        '% Generated by perceptalign/paper_tables.py; do not edit by hand (sources: paper/generated/sources.md).',
        STD_MACRO,
        '\\begin{table}[t]', '  \\centering', f'  \\caption{{{caption}}}', '  \\label{tab:window}', '  \\small',
        '  \\setlength{\\tabcolsep}{3.2pt}', '  \\begin{tabular}{lccccccc}', '    \\toprule',
        '    & \\multicolumn{3}{c}{FlashDot} & \\multicolumn{4}{c}{VAS} \\\\',
        '    \\cmidrule(lr){2-4} \\cmidrule(lr){5-8}',
        '    Selection & ' + ' & '.join(c[1] for c in WIN_COLS) + ' \\\\', '    \\midrule',
        *body, '    \\bottomrule', '  \\end{tabular}', '\\end{table}', ''])
    return tex, abl


def bestofn(abl):
    """Best-of-N curves (n = 1, 2, 4, ..., n_cand) under s1 and s0 for both datasets."""
    out = {}
    for d, name in (('fd', 'flashdot'), ('vas', 'vas')):
        a = abl[d]
        C = n_cand_of(a)
        ns = [1] + sorted({2 ** i for i in range(1, C.bit_length())} | {C})
        metrics = sorted({m for _, j in a.values() for s in j.get('means', {}).values() for m in s})
        entry = dict(sources={g: rel(f) for g, (f, _) in sorted(a.items())}, n_generators=len(a),
                     n_clips=sorted({j['n'] for _, j in a.values()}), curves={})
        for w in ('s1', 's0'):
            sels = ['first' if n == 1 else f'best{n}_{w}' for n in ns]
            cur = dict(n=ns, selection=sels)
            for m in metrics:
                cs = [abl_cell(a, s, m) for s in sels]
                cur[m] = dict(mean=[c.mean for c in cs], std=[c.std for c in cs], n_seeds=[c.n for c in cs])
            entry['curves'][w] = cur
        out[name] = entry
    return out


# ----------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--work', default='work')
    ap.add_argument('--out', default='paper/generated')
    ap.add_argument('--sig_vas', default='work/results/sig_vas_test.json',
                    help='significance.py json with a rand:pal pair; marks the PAL row of tab:vas (if present)')
    ap.add_argument('--alpha', type=float, default=0.05)
    args = ap.parse_args()
    work, out = Path(args.work), Path(args.out)
    runs = work / 'runs'
    out.mkdir(parents=True, exist_ok=True)
    src = Sources()

    (out / 'tab_flashdot.tex').write_text(tab_flashdot(work, runs, src))
    (out / 'tab_vas.tex').write_text(tab_vas(work, runs, src, Path(args.sig_vas), args.alpha))
    tex, abl = tab_window(runs, src)
    (out / 'tab_window.tex').write_text(tex)
    (out / 'bestofn.json').write_text(json.dumps(bestofn(abl), indent=1))

    sig_line = (f'Significance file for tab:vas: `{args.sig_vas}` '
                f'({"present" if Path(args.sig_vas).exists() else "not present, no marks"}).')
    head = ['# Sources of every generated table cell', '',
            'Generated by `PYTHONPATH=. python -m perceptalign.paper_tables`. Each cell is the mean over the listed '
            'files of the listed key (sample std, ddof 1, over training/generator seeds when more than one file). '
            'Metric keys are those of `metrics_*.json` (sample_eval.py); `means/<selection>/<metric>` refers to '
            '`rerank_ablation_*.json`. `\\placeholder{}` marks a value whose file does not exist yet.', '', sig_line,
            '', '## Warnings', '']
    head += [f'- {w}' for w in WARNINGS] or ['- none']
    (out / 'sources.md').write_text('\n'.join(head + src.lines) + '\n')
    n_ph = {f: (out / f).read_text().count(PH) for f in ('tab_flashdot.tex', 'tab_vas.tex', 'tab_window.tex')}
    print('wrote', ', '.join(str(out / f) for f in n_ph), 'bestofn.json, sources.md')
    print('placeholders:', n_ph)


if __name__ == '__main__':
    main()

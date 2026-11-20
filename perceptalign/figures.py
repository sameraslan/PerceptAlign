"""Paper figures.  python -m perceptalign.figures --which all|<name>

Each figure is one function that reads results already on disk and writes
paper/figures/<name>.pdf (vector) plus a .png preview.  CPU only; large
arrays are opened with mmap_mode='r' and only the needed rows are read.
"""
import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
WORK, OUT = ROOT / 'work', ROOT / 'paper' / 'figures'
FD, VAS = WORK / 'flashdot_std', WORK / 'vas'
FPS = 22050 / 256          # mel frames per second
COL_W, FULL_W = 3.2, 6.5   # inches

# Validated categorical slots (dataviz reference palette, light mode) + neutrals.
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
GRAY, INK, MUTED = '#8a8984', '#0b0b0b', '#52514e'
SPEC_CMAP = LinearSegmentedColormap.from_list(
    'spec', ['#fcfcfb', '#cde2fb', '#6da7ec', '#2a78d6', '#1c5cab', '#0d366b'])


# ---------------------------------------------------------------- helpers
def _style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'STIX Two Text', 'STIXGeneral', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'axes.linewidth': 0.6, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
        'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.color': MUTED, 'ytick.color': MUTED,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'axes.grid': True, 'axes.grid.axis': 'y', 'grid.color': '#e4e3df', 'grid.linewidth': 0.5,
        'axes.axisbelow': True, 'lines.linewidth': 1.4, 'lines.markersize': 4,
        'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'savefig.dpi': 300, 'figure.dpi': 150,
    })


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{name}.{ext}', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print('wrote', OUT / f'{name}.pdf')


def _json(path):
    return json.loads(Path(path).read_text())


def _row(path, i):
    """One clip from a [N, 80, T] (or [N, S, 80, T]) array without loading the rest."""
    a = np.asarray(np.load(path, mmap_mode='r')[i], np.float32)
    return a[0] if a.ndim == 3 else a


def _spec(ax, mel, label):
    t = mel.shape[-1] / FPS
    ax.imshow(mel, origin='lower', aspect='auto', cmap=SPEC_CMAP, vmin=0, vmax=1,
              extent=(0, t, 0, mel.shape[0]), interpolation='nearest', rasterized=True)
    ax.set_yticks([0, 40, 80])
    ax.set_ylabel('mel bin')
    ax.grid(False)
    ax.text(0.005, 0.96, label, transform=ax.transAxes, ha='left', va='top', color=INK,
            bbox=dict(boxstyle='round,pad=0.2', fc='#fcfcfb', ec='none', alpha=0.85))


def _spec_stack(entries, idx, name, gt_path, act_path=None):
    """Ground truth on top, then one row per (label, path); optional activity row."""
    rows = [('ground truth', _row(gt_path, idx))] + [(l, _row(p, idx)) for l, p in entries]
    n = len(rows) + (act_path is not None)
    ratios = [1] * len(rows) + ([0.3] if act_path is not None else [])
    fig, axes = plt.subplots(n, 1, figsize=(FULL_W, 0.85 * sum(ratios) + 0.35), sharex=True,
                             gridspec_kw=dict(height_ratios=ratios, hspace=0.2))
    axes = np.atleast_1d(axes)
    for ax, (label, mel) in zip(axes, rows):
        _spec(ax, mel, label)
    if act_path is not None:
        act = np.asarray(np.load(act_path, mmap_mode='r')[idx], np.float32)
        t = np.arange(len(act)) / FPS
        ax = axes[-1]
        ax.fill_between(t, act, step='mid', color=BLUE, lw=0, alpha=0.9)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 1], ['off', 'on'])
        ax.set_ylabel('tone')
        ax.grid(False)
    axes[-1].set_xlabel('time (s)')
    axes[-1].set_xlim(0, rows[0][1].shape[-1] / FPS)
    _save(fig, name)


# ---------------------------------------------------------------- figures
def metric_sensitivity():
    d = _json(WORK / 'results' / 'metric_sensitivity_flashdot.json')
    keys = sorted((k for k in d if k.startswith('delay_')), key=lambda k: int(k[6:-2]))
    delays = [int(k[6:-2]) for k in keys]
    ref = d['other_clip_same_class']
    panels = [('dtw', r'DTW $\downarrow$'), ('mel_l1', r'mel L1 $\downarrow$'),
              ('onset_f1', r'onset F1 $\uparrow$'), ('env_corr', r'envelope corr. $\uparrow$'),
              ('abs_lag_ms', r'|lag| (ms) $\downarrow$')]
    x = np.arange(len(delays))
    fig, axes = plt.subplots(1, 5, figsize=(FULL_W, 1.55), gridspec_kw=dict(wspace=0.55))
    for ax, (key, title) in zip(axes, panels):
        y = [d[k][key] for k in keys]
        ax.plot(x, y, color=BLUE, marker='o', mec='white', mew=0.6, zorder=3)
        r = ref[key]
        ax.axhline(r, color=GRAY, ls=(0, (3, 2)), lw=0.9, zorder=2)
        top = max(max(y), r)
        below = key in ('env_corr', 'abs_lag_ms')  # keep the label clear of the curve
        ax.set_ylim(min(0, min(y)) - (0.3 if key == 'env_corr' else 0.04) * top, top * 1.18)
        ax.text(x[0], r + (-0.04 if below else 0.03) * top, 'unrelated\nclip', color=MUTED,
                fontsize=6.5, linespacing=0.95, va='top' if below else 'bottom')
        if key == 'abs_lag_ms':
            ax.text(x[0], 1.16 * top, '|lag| equals the\ninjected delay', color=MUTED,
                    fontsize=6.5, linespacing=0.95, va='top')
        ax.set_title(title, loc='left', pad=3)
        ax.set_xticks(x, [str(v) for v in delays], rotation=90)
        ax.set_xlim(-0.4, len(x) - 0.6)
    axes[2].axvspan(3.5, len(x) - 0.6, color='#f1f0ec', lw=0, zorder=0)  # onset F1 collapse
    fig.supxlabel('injected delay (ms)', fontsize=8, y=-0.14)
    _save(fig, 'metric_sensitivity')


def critic_attack():
    names = [('mel', 'mel'), ('b8', '8\nbands'), ('b4', '4\nbands'), ('b1', '1\nband')]
    res = [_json(WORK / 'results' / f'attack_fd_{k}.json') for k, _ in names]
    groups = [('random', 'random logits', GRAY, 'o'),
              ('optimised', 'PAL-optimised', ORANGE, 'D'),
              ('gt_codes', 'ground-truth codes', BLUE, 's')]
    panels = [(lambda r, g: r[f'pal_{g}'], r'PAL $\downarrow$', 'linear'),
              (lambda r, g: r[g]['act_f1'], r'tone-activity F1 $\uparrow$', 'linear'),
              (lambda r, g: r[g]['abs_lag_ms'], r'|lag| (ms) $\downarrow$', 'log')]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 1.9), gridspec_kw=dict(wspace=0.38))
    for ax, (get, title, scale) in zip(axes, panels):
        for j, (g, label, c, m) in enumerate(groups):
            ax.plot(x + (j - 1) * 0.2, [get(r, g) for r in res], ls='none', marker=m, ms=5,
                    color=c, mec='white', mew=0.6, label=label)
        ax.set_yscale(scale)
        ax.set_title(title, loc='left', pad=3)
        ax.set_xticks(x, [n for _, n in names])
        ax.set_xlim(-0.5, len(x) - 0.5)
        ax.grid(True, axis='y', which='major')
        if scale == 'log':
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
    axes[0].set_ylim(0, None)
    axes[1].set_ylim(0, 1.05)
    fig.supxlabel('critic (audio input)', fontsize=8, y=-0.1)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, 0.97), handletextpad=0.3, columnspacing=1.4)
    _save(fig, 'critic_attack')


def flashdot_example(idx=0):
    _spec_stack([('VQGAN reconstruction', FD / 'dev_recon.npy')], idx, 'flashdot_example',
                FD / 'dev_mel.npy', act_path=FD / 'dev_act.npy')


def _median_gain_clip(single_perclip, best_perclip, key='abs_lag_ms'):
    """Clip whose per-clip |lag| gain from re-ranking is the median gain: a typical case, not the best one."""
    a, b = np.load(single_perclip)[key], np.load(best_perclip)[key]
    gain = a - b
    ok = np.flatnonzero(~np.isnan(gain))
    order = ok[np.argsort(gain[ok], kind='stable')]
    return int(order[len(order) // 2])


def flashdot_samples(name='flashdot_samples'):
    """FlashDot dev: CE + PAL (teacher-forced), corruption-trained CE, and its best of 8."""
    ce = WORK / 'runs' / 'fd2_ce_pk05'
    idx = _median_gain_clip(ce / 'samples_dev_s0_perclip.npz', ce / 'samples_dev_s0_rr8_perclip.npz')
    entries = [('CE + PAL, teacher-forced', WORK / 'runs' / 'fd_pal_s0' / 'samples_dev_s0.npy'),
               ('CE, corrupted prefix: one sample', ce / 'samples_dev_s0.npy'),
               ('CE, corrupted prefix: best of 8 by PAL', ce / 'samples_dev_s0_rr8.npy')]
    _spec_stack(entries, idx, name, FD / 'dev_mel.npy', act_path=FD / 'dev_act.npy')
    return idx


def vas_samples(name='vas_samples'):
    """VAS test: pretrained SpecVQGAN, its best of 4, and PAL self-training."""
    runs = WORK / 'runs'
    single = runs / 'vas_base' / 'samples_test_s0_b940f7'
    best = runs / 'vas_base' / 'samples_test_s0_rr4_735d74'
    idx = _median_gain_clip(f'{single}_perclip.npz', f'{best}_perclip.npz')
    entries = [('pretrained SpecVQGAN: one sample', f'{single}.npy'),
               ('pretrained SpecVQGAN: best of 4 by PAL', f'{best}.npy'),
               ('self-training, PAL selection', runs / 'vas_st_pal' / 'samples_test_s0_6cf256.npy')]
    _spec_stack(entries, idx, name, VAS / 'test_mel.npy')
    return idx


FIGURES = dict(metric_sensitivity=metric_sensitivity, critic_attack=critic_attack,
               flashdot_example=flashdot_example, flashdot_samples=flashdot_samples,
               vas_samples=vas_samples)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--which', default='all', choices=['all', *FIGURES])
    a = p.parse_args()
    _style()
    for n, f in FIGURES.items():
        if a.which in ('all', n):
            f()


if __name__ == '__main__':
    main()

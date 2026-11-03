"""Temporal-alignment metrics between generated and ground-truth mel spectrograms.

All inputs are log-mel spectrograms in the SpecVQGAN convention: (80, T) arrays in [0, 1]
at 22050 Hz with hop 256 (86.13 frames/s). Metrics:

onset_f1      onset F1 (mir_eval matching, +-tol s) between onsets detected in the generated
              and the ground-truth spectrogram
onset_ap      average precision of generated onset peaks ranked by onset strength, a peak
              counting as a hit when it matches an unmatched ground-truth onset within tol
env_corr      Pearson correlation of the two onset-strength envelopes (Gaussian-smoothed, at the
              mel frame rate)
lag_ms        lag (ms) that maximises the cross-correlation of the two envelopes within
              +-max_lag_s; |lag| is a direct estimate of the audio-video offset
dtw           DTW-Euclidean on per-clip standardised mel frames (the metric of the earlier
              draft; kept to show it is insensitive to timing)
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

SR, HOP = 22050, 256
FPS_MEL = SR / HOP


def onset_envelope(mel):
    """Spectral flux on the log-mel: sum over bins of positive frame differences."""
    d = np.diff(mel.astype(np.float64), axis=1)
    env = np.maximum(d, 0).sum(0)
    return np.concatenate([[0.0], env])


def detect_onsets(env, delta=None, min_gap_s=0.05):
    """Peak picking on a (smoothed) onset envelope. Returns times (s) and peak strengths."""
    e = gaussian_filter1d(env, 1.0)
    if delta is None:
        delta = max(0.5, np.median(e) + 1.5 * e.std())
    peaks, props = find_peaks(e, height=delta, distance=max(1, int(min_gap_s * FPS_MEL)))
    return peaks / FPS_MEL, props['peak_heights']


def match_onsets(ref, est, tol):
    """One-to-one matching (Hopcroft-Karp as in mir_eval). Returns number of matches."""
    import mir_eval
    if len(ref) == 0 or len(est) == 0:
        return 0
    return len(mir_eval.util.match_events(np.asarray(ref), np.asarray(est), tol))


def onset_scores(mel_gen, mel_gt, tol=0.1):
    t_gt, _ = detect_onsets(onset_envelope(mel_gt))
    t_gen, s_gen = detect_onsets(onset_envelope(mel_gen))
    n_match = match_onsets(t_gt, t_gen, tol)
    if len(t_gt) == 0 and len(t_gen) == 0:
        f1 = 1.0
    else:
        f1 = 2 * n_match / (len(t_gt) + len(t_gen))
    # AP: rank generated onsets by strength; greedy hit assignment in rank order
    if len(t_gt) == 0:
        ap = np.nan
    else:
        order = np.argsort(-s_gen)
        used = np.zeros(len(t_gt), bool)
        hits = []
        for i in order:
            d = np.abs(t_gt - t_gen[i])
            d[used] = np.inf
            j = int(np.argmin(d)) if len(d) else -1
            if j >= 0 and d[j] <= tol:
                used[j] = True
                hits.append(1)
            else:
                hits.append(0)
        hits = np.array(hits)
        if hits.sum() == 0:
            ap = 0.0
        else:
            prec = np.cumsum(hits) / (np.arange(len(hits)) + 1)
            ap = float((prec * hits).sum() / len(t_gt))
    return dict(onset_f1=f1, onset_ap=ap, n_onsets_gt=len(t_gt), n_onsets_gen=len(t_gen),
                onset_count_match=float(len(t_gt) == len(t_gen)))


def video_rate_envelope(mel, n_frames=212):
    env = gaussian_filter1d(onset_envelope(mel), 2.0)
    x = np.linspace(0, len(env) - 1, n_frames)
    return np.interp(x, np.arange(len(env)), env)


def envelope_scores(mel_gen, mel_gt, max_lag_s=1.0):
    eg = gaussian_filter1d(onset_envelope(mel_gen), 2.0)
    et = gaussian_filter1d(onset_envelope(mel_gt), 2.0)
    eg, et = eg - eg.mean(), et - et.mean()
    denom = np.sqrt((eg ** 2).sum() * (et ** 2).sum()) + 1e-12
    corr0 = float((eg * et).sum() / denom)
    L = int(max_lag_s * FPS_MEL)
    xc = np.correlate(eg, et, mode='full')[len(et) - 1 - L:len(et) + L]
    lag = int(np.argmax(xc)) - L  # positive: generated audio is late
    return dict(env_corr=corr0, lag_ms=1000 * lag / FPS_MEL, xcorr_peak=float(xc.max() / denom))


def dtw_euclidean(mel_gen, mel_gt):
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean

    def standardise(m):
        m = m.T.astype(np.float64)
        return (m - m.mean(0)) / (m.std(0) + 1e-8)
    d, _ = fastdtw(standardise(mel_gen), standardise(mel_gt), dist=euclidean)
    return float(d)


def mel_l1(mel_gen, mel_gt):
    return float(np.abs(mel_gen.astype(np.float32) - mel_gt.astype(np.float32)).mean())


def clip_scores(mel_gen, mel_gt, with_dtw=False, tol=0.1):
    s = dict(**onset_scores(mel_gen, mel_gt, tol), **envelope_scores(mel_gen, mel_gt),
             mel_l1=mel_l1(mel_gen, mel_gt))
    if with_dtw:
        s['dtw'] = dtw_euclidean(mel_gen, mel_gt)
    return s


def activity_scores(act_pred, act_gt):
    """Frame-level F1 / IoU of binary sound activity (FlashDot)."""
    p, g = act_pred.astype(bool), act_gt.astype(bool)
    tp = (p & g).sum()
    f1 = 2 * tp / max(1, p.sum() + g.sum())
    iou = tp / max(1, (p | g).sum())
    return dict(act_f1=float(f1), act_iou=float(iou))


def aggregate(rows):
    keys = rows[0].keys()
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=np.float64)
        out[k] = float(np.nanmean(v))
        if k == 'lag_ms':
            out['abs_lag_ms'] = float(np.nanmean(np.abs(v)))
            out['lag_within_1vf'] = float(np.nanmean(np.abs(v) <= 1000 / 21.5 + 1e-6))
    return out

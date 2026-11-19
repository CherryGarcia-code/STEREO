"""
Surface-licking state analysis of striatal photometry.

State labelling follows the spirit of the VizDetect behavioural state labeller
(vis_detect_analysis_Sep2025): a state is defined by *local composition in a
window*, not frame by frame, and stretches that are not compositionally uniform
are left unassigned rather than forced into a class. Here the behaviour is
already tagged by STEREO, so no human labelling loop is needed -- what is
labelled is the *stretch*, i.e. whether a run of frames is uniform enough to
count as a reliable surface-licking (SL) or non-surface-licking (non-SL) state.

Contrasts
  1  PRIMARY    SL vs non-SL stretches (cocaine sessions)
  2  SECONDARY  SL stretches early vs late within a session
  3  state x session third (2 x 3)
  4  across cocaine exposure, cocaine1 -> cocaine5
  5  saline sessions as negative control

Signals: both are reported.
  z   per-session z-scored. Common scale, so it pools across mice/sessions.
      Blind to across-session amplitude by construction (every session has SD=1).
  df  the same trace before that rescaling (corr(df, z) = 1.000000 exactly,
      verified over all 118 hemisphere-sessions; df is already mean-centred).
      Retains absolute amplitude, so it is the only one that can speak to
      across-session growth -- median SD(df) runs 0.428 -> 1.131 from
      cocaine1 to cocaine5.

Because of that, within-session panels are near-identical between the two by
construction; the across-session panels are where they genuinely differ.

Bleaching is NOT corrected for here and must not be: the upstream pipeline
(analyses/photom_PP_validation.py) already removes it ratiometrically, and on
39 saline sessions there is no residual drift (corr of z with time r = +0.001,
|r| > 0.2 in 0% of sessions; peak rate, prominence and width all flat).

Outputs -> output/Photometry/SurfaceLickingStates/
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'revision'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.signal
from scipy import stats
from collections import defaultdict

import data_io
from revision_utils import mm, photom_exclusions, get_pathway, setup_style, save_fig

setup_style()

# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
FPS = 15
FL, WL = 2, 3                       # surface licking = floor + wall licking

COCAINE = ['cocaine1', 'cocaine2', 'cocaine3', 'cocaine4', 'cocaine5']
SALINE = ['saline1', 'saline2', 'saline3', 'salineOnly']

W = 15 * FPS                        # rolling composition window
HI, LO = 0.50, 0.10                 # SL / non-SL admission thresholds
MIN_DUR = 3 * FPS                   # minimum stretch duration
MIN_PURITY = 0.0                    # reported; admission is via HI/LO

PEAK_WIDTH = 5                      # frames, repo convention
PEAK_PROM_Z = 2.0                   # z units, repo convention (Figure3, SuppFig8)
PEAK_PROM_DF = 1.2                  # absolute dF/F units (~2x corpus median SD)

W_SWEEP = [5 * FPS, 10 * FPS, 15 * FPS, 30 * FPS]
THR_SWEEP = [(0.4, 0.20), (0.5, 0.10), (0.6, 0.05)]

# Event-aligned windows. Repo convention is pre=3-5 s, post=5 s with a 2 s
# baseline (Figure3:405, Figure7:300-302); post is extended here because SL
# stretches run 30-80 s, so 5 s would not reach the settled state.
PRE_EV = 5 * FPS
POST_EV = 10 * FPS
BASE_EV = 2 * FPS          # baseline = the first 2 s of the pre-window
RESP_EV = 5 * FPS          # response window = first 5 s after the transition

PATHWAY_COLORS = {'dSPN': '#d73027', 'iSPN': '#721515'}
STATE_COLORS = {'SL': '#d73027', 'nonSL': '#1f4e79'}
DAY_COLORS = ['#FFA500', '#FF8C00', '#FF6347', '#E60000', '#990000']

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   '..', 'output', 'Photometry', 'SurfaceLickingStates')
OUT = os.path.normpath(OUT)
os.makedirs(OUT, exist_ok=True)


# --------------------------------------------------------------------------
# State labelling
# --------------------------------------------------------------------------
def rolling_mean(x, w):
    """Centred rolling mean, edges shrink (min_periods style)."""
    c = np.concatenate(([0.0], np.cumsum(x, dtype=float)))
    idx = np.arange(x.size)
    a = np.maximum(0, idx - w // 2)
    b = np.minimum(x.size, idx + w // 2 + 1)
    return (c[b] - c[a]) / (b - a)


def runs_of(mask, min_dur):
    m = mask.astype(np.int8)
    d = np.diff(np.concatenate(([0], m, [0])))
    return [(s, e) for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0])
            if e - s >= min_dur]


def label_stretches(preds, w=W, hi=HI, lo=LO, min_dur=MIN_DUR):
    """Return [(state, start, end, purity)] for reliable stretches only.

    Frames whose local SL fraction falls between `lo` and `hi` belong to no
    state -- the analogue of VizDetect's confidence gate, so compositionally
    mixed stretches contaminate neither arm.
    """
    sl = np.isin(preds, [FL, WL]).astype(float)
    frac = rolling_mean(sl, w)
    out = []
    for state, runs in (('SL', runs_of(frac >= hi, min_dur)),
                        ('nonSL', runs_of(frac <= lo, min_dur))):
        for a, b in runs:
            purity = sl[a:b].mean() if state == 'SL' else 1.0 - sl[a:b].mean()
            if purity >= MIN_PURITY:
                out.append((state, a, b, float(purity)))
    return out


# --------------------------------------------------------------------------
# Per-stretch photometry metrics
# --------------------------------------------------------------------------
def stretch_metrics(sig, a, b, prominence):
    """Signal metrics for one stretch. Peak latency is from stretch onset."""
    seg = sig[a:b]
    seg = seg[np.isfinite(seg)]
    dur_s = (b - a) / FPS
    if seg.size < PEAK_WIDTH + 1:
        return None
    pk, props = scipy.signal.find_peaks(seg, prominence=prominence, width=PEAK_WIDTH)
    return dict(
        mean=float(np.mean(seg)),
        sd=float(np.std(seg)),
        auc=float(np.trapz(seg) / FPS),
        peak_rate=len(pk) / (dur_s / 60.0),
        n_peaks=len(pk),
        prominence=float(np.mean(props['prominences'])) if len(pk) else np.nan,
        width=float(np.mean(props['widths'])) / FPS if len(pk) else np.nan,
        first_peak_s=float(pk[0]) / FPS if len(pk) else np.nan,
        duration_s=dur_s,
    )


# NOTE: AUC is deliberately not in this list. AUC == mean x duration, so it is
# not length-normalised and carries no information beyond `mean` once stretch
# duration is accounted for -- per-mouse early/late duration ratios range 0.31
# to 2.15, which inflates AUC effect sizes without adding evidence. It stays in
# the per-stretch CSV; `duration_s` is reported here instead, as a descriptor.
METRICS = ['mean', 'sd', 'peak_rate', 'prominence', 'width', 'first_peak_s', 'duration_s']
METRIC_LABELS = {
    'mean': 'Mean signal', 'sd': 'SD of signal', 'auc': 'AUC',
    'peak_rate': 'Peak rate (/min)', 'prominence': 'Peak amplitude',
    'width': 'Peak width (s)', 'first_peak_s': 'Latency to 1st peak (s)',
    'duration_s': 'Stretch duration (s)',
}


# --------------------------------------------------------------------------
# Collection
# --------------------------------------------------------------------------
def collect(CMT, w=W, hi=HI, lo=LO):
    """One record per (stretch, signal). Hemispheres averaged after exclusion."""
    rm_mouse = {m for _, m in photom_exclusions['remove_mouse']}
    rm_hem = {(m, h) for _, m, h in photom_exclusions['remove_hemisphere']}
    rm_trial = {(m, t) for _, m, t in photom_exclusions['remove_trial']}

    records = []
    for c in CMT:
        for m in CMT[c]:
            if m in rm_mouse:
                continue
            pathway = 'dSPN' if get_pathway(c, m) == 'drd1' else 'iSPN'
            for t in CMT[c][m]:
                if t not in COCAINE + SALINE or (m, t) in rm_trial:
                    continue
                e = CMT[c][m][t]
                if 'photom' not in e:
                    continue
                hems = [h for h in ('left', 'right')
                        if isinstance(e['photom'].get(h), dict) and (m, h) not in rm_hem]
                if not hems:
                    continue

                sigs = {k: np.nanmean(np.vstack([e['photom'][h][k] for h in hems]), axis=0)
                        for k in ('z', 'df')}
                vel = e['topcam']['velocity']
                preds = e['merged']
                n = preds.size
                drug = 'cocaine' if t in COCAINE else 'saline'

                for state, a, b, purity in label_stretches(preds, w, hi, lo):
                    mid = (a + b) / 2
                    third = 0 if mid < n / 3 else (1 if mid < 2 * n / 3 else 2)
                    v = vel[a:b]
                    for sig_name, prom in (('z', PEAK_PROM_Z), ('df', PEAK_PROM_DF)):
                        met = stretch_metrics(sigs[sig_name], a, b, prom)
                        if met is None:
                            continue
                        records.append(dict(
                            cohort=c, mouse=m, pathway=pathway, trial=t, drug=drug,
                            signal=sig_name, state=state, third=third,
                            onset_frac=float(mid / n), purity=purity,
                            velocity=float(np.nanmean(v)) if np.isfinite(v).any() else np.nan,
                            **met))
    return records


MIN_BOUT_EV = 1 * FPS      # shortest licking bout that counts as "the" trigger


def _bouts(sl_slice, min_len):
    """(start, end) of contiguous licking runs of at least min_len, slice-local."""
    d = np.diff(np.concatenate(([0], sl_slice.astype(np.int8), [0])))
    return [(s, e) for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0])
            if e - s >= min_len]


def behavioural_index(sl, idx, state, kind, w=W):
    """Re-align a state-label boundary onto the licking event that caused it.

    The state boundary is where a *centred* rolling window crosses a threshold,
    so it lags the actual behaviour by roughly half the window. This finds the
    licking transition responsible, inside the window that triggered the flip:

      SL onset     -> first frame of the first licking bout in [idx-w, idx]
      SL offset    -> last frame of the last licking bout in [idx-w, idx]
      nonSL onset  -> last frame of the last licking bout in [idx-w, idx]
                      (i.e. the moment licking actually stopped)

    Returns None when no qualifying bout exists, so the event is dropped rather
    than silently mis-aligned.
    """
    lo = max(0, idx - w)
    runs = _bouts(sl[lo:idx + 1], MIN_BOUT_EV)
    if not runs:
        return None
    if state == 'SL' and kind == 'onset':
        return lo + runs[0][0]
    return lo + runs[-1][1] - 1        # SL offset, and nonSL onset


def collect_events(CMT, align='state'):
    """Baseline-corrected traces aligned to state transitions.

    Events are stretch boundaries: entering SL, leaving SL, and entering non-SL
    as a control. An event is only kept if the pre-window is clean, i.e. no
    other stretch of the *target* state overlaps it -- otherwise the baseline
    would sit inside a neighbouring bout of the same state.

    align='state'       t=0 is the state-label boundary
    align='behaviour'   t=0 is the licking transition that caused it, so the
                        baseline is genuinely pre-licking and latency is readable
    """
    rm_mouse = {m for _, m in photom_exclusions['remove_mouse']}
    rm_hem = {(m, h) for _, m, h in photom_exclusions['remove_hemisphere']}
    rm_trial = {(m, t) for _, m, t in photom_exclusions['remove_trial']}

    events = []
    for c in CMT:
        for m in CMT[c]:
            if m in rm_mouse:
                continue
            pathway = 'dSPN' if get_pathway(c, m) == 'drd1' else 'iSPN'
            for t in CMT[c][m]:
                if t not in COCAINE + SALINE or (m, t) in rm_trial:
                    continue
                e = CMT[c][m][t]
                if 'photom' not in e:
                    continue
                hems = [h for h in ('left', 'right')
                        if isinstance(e['photom'].get(h), dict) and (m, h) not in rm_hem]
                if not hems:
                    continue
                sigs = {k: np.nanmean(np.vstack([e['photom'][h][k] for h in hems]), axis=0)
                        for k in ('z', 'df')}
                preds = e['merged']
                n = preds.size
                drug = 'cocaine' if t in COCAINE else 'saline'
                stretches = label_stretches(preds)
                sl_mask = np.isin(preds, [FL, WL])
                by_state = defaultdict(list)
                for st, a, b, _ in stretches:
                    by_state[st].append((a, b))

                for state in ('SL', 'nonSL'):
                    spans = sorted(by_state[state])
                    for i, (a, b) in enumerate(spans):
                        prev_end = spans[i - 1][1] if i > 0 else -10 ** 9
                        next_start = spans[i + 1][0] if i + 1 < len(spans) else 10 ** 9
                        for kind, idx, clean in (('onset', a, prev_end <= a - PRE_EV),
                                                 ('offset', b, next_start >= b + POST_EV)):
                            if not clean:
                                continue
                            if align == 'behaviour':
                                if state == 'nonSL' and kind == 'offset':
                                    continue          # not a meaningful licking event
                                idx = behavioural_index(sl_mask, idx, state, kind)
                                if idx is None:
                                    continue
                            if idx < PRE_EV or idx + POST_EV > n:
                                continue
                            mid = (a + b) / 2
                            third = 0 if mid < n / 3 else (1 if mid < 2 * n / 3 else 2)
                            rec = dict(mouse=m, pathway=pathway, trial=t, drug=drug,
                                       state=state, kind=kind, third=third)
                            for sig_name, sig in sigs.items():
                                seg = sig[idx - PRE_EV: idx + POST_EV].astype(float)
                                if seg.size != PRE_EV + POST_EV or not np.isfinite(seg).all():
                                    continue
                                rec[f'trace_{sig_name}'] = seg - np.mean(seg[:BASE_EV])
                            if 'trace_z' in rec and 'trace_df' in rec:
                                events.append(rec)
    return events


def event_mouse_traces(events, signal, pathway, state, kind, drug='cocaine', third=None):
    """Average traces within mouse, then return the stack over mice."""
    acc = defaultdict(list)
    for ev in events:
        if (ev['pathway'] == pathway and ev['state'] == state and ev['kind'] == kind
                and ev['drug'] == drug and (third is None or ev['third'] == third)):
            acc[ev['mouse']].append(ev[f'trace_{signal}'])
    mice = sorted(acc)
    if not mice:
        return np.empty((0, PRE_EV + POST_EV)), [], 0
    stack = np.vstack([np.mean(acc[mo], axis=0) for mo in mice])
    return stack, mice, sum(len(v) for v in acc.values())


def mouse_means(records, metric, signal, drug, filt=None, key=('state',)):
    """Aggregate stretches -> mouse-level means. Returns {mouse: {cellkey: value}}."""
    acc = defaultdict(list)
    for r in records:
        if r['signal'] != signal or r['drug'] != drug:
            continue
        if filt and not filt(r):
            continue
        if not np.isfinite(r[metric]):
            continue
        acc[(r['mouse'], tuple(r[k] for k in key))].append(r[metric])
    out = defaultdict(dict)
    for (mouse, cell), vals in acc.items():
        out[mouse][cell] = float(np.mean(vals))
    return out


def paired(mm_dict, cell_a, cell_b, mice):
    """Paired arrays over mice that have both cells."""
    a, b, used = [], [], []
    for mo in mice:
        d = mm_dict.get(mo, {})
        if cell_a in d and cell_b in d:
            a.append(d[cell_a]); b.append(d[cell_b]); used.append(mo)
    return np.array(a), np.array(b), used


def test_pair(a, b):
    if len(a) < 3:
        return dict(n=len(a), t=np.nan, p=np.nan, w=np.nan, pw=np.nan, d=np.nan)
    t, p = stats.ttest_rel(a, b)
    try:
        w, pw = stats.wilcoxon(a, b)
    except ValueError:
        w, pw = np.nan, np.nan
    diff = a - b
    d = float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else np.nan
    return dict(n=len(a), t=float(t), p=float(p), w=w, pw=pw, d=d)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    print('Loading Apr26 dictionaries...')
    CMT = data_io.load_CMT(flatten=True)
    print('Labelling stretches and extracting photometry metrics...')
    records = collect(CMT)
    print('Extracting transition-aligned traces...')
    events_state = collect_events(CMT, align='state')
    events_beh = collect_events(CMT, align='behaviour')
    print(f'  {len(events_state)} state-aligned, {len(events_beh)} behaviour-aligned events')
    print('Rendering example ethograms...')

    mice_by_path = defaultdict(set)
    for r in records:
        mice_by_path[r['pathway']].add(r['mouse'])
    print(f'{len(records)//2} stretches; '
          + '; '.join(f'{p}: {len(s)} mice' for p, s in sorted(mice_by_path.items())))

    # per-stretch table for downstream exploration
    import csv
    csv_path = os.path.join(OUT, 'stretch_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        wtr = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        wtr.writeheader()
        wtr.writerows(records)
    print(f'  per-stretch table -> {os.path.basename(csv_path)} ({len(records)} rows)')

    sink = open(os.path.join(OUT, 'statistics.txt'), 'w')
    sink.write('EXAMPLE SESSIONS (see ethogram_*.png for the re-shade review)\n')
    example_ethograms(CMT, sink)
    del CMT
    sink.write('\n')
    sink.write('Surface-licking state analysis of striatal photometry\n')
    sink.write('=' * 70 + '\n')
    sink.write(f'window={W//FPS}s  SL>={HI}  nonSL<={LO}  min duration={MIN_DUR//FPS}s\n')
    sink.write(f'peaks: z prominence={PEAK_PROM_Z} (repo convention); '
               f'df prominence={PEAK_PROM_DF} absolute; width>={PEAK_WIDTH} frames\n')
    sink.write('Unit of analysis: stretches -> mouse means -> test across mice.\n\n')

    for signal in ('z', 'df'):
        for pathway in ('dSPN', 'iSPN'):
            mice = sorted(mice_by_path[pathway])
            fp = (lambda p: (lambda r: r['pathway'] == p))(pathway)

            sink.write('=' * 70 + f'\nSIGNAL = {signal}   PATHWAY = {pathway}\n' + '=' * 70 + '\n')

            # ---- 1 PRIMARY: SL vs non-SL on cocaine ----
            sink.write('\n[1] PRIMARY  SL vs non-SL stretches (cocaine)\n')
            sink.write(f'  {"metric":<24}{"SL":>10}{"nonSL":>10}{"diff":>10}'
                       f'{"t":>8}{"p":>10}{"wilcox p":>10}{"dz":>7}{"N":>4}\n')
            for met in METRICS:
                mmv = mouse_means(records, met, signal, 'cocaine', fp, ('state',))
                a, b, used = paired(mmv, ('SL',), ('nonSL',), mice)
                if len(a) < 3:
                    continue
                r_ = test_pair(a, b)
                sink.write(f'  {METRIC_LABELS[met]:<24}{a.mean():>10.3f}{b.mean():>10.3f}'
                           f'{a.mean()-b.mean():>10.3f}{r_["t"]:>8.2f}{r_["p"]:>10.4f}'
                           f'{r_["pw"]:>10.4f}{r_["d"]:>7.2f}{r_["n"]:>4d}\n')

            # ---- 2 SECONDARY: SL early vs late ----
            sink.write('\n[2] SECONDARY  SL stretches, early (1st third) vs late (3rd third), cocaine\n')
            sink.write(f'  {"metric":<24}{"early":>10}{"late":>10}{"diff":>10}'
                       f'{"t":>8}{"p":>10}{"dz":>7}{"N":>4}\n')
            for met in METRICS:
                f2 = (lambda p: (lambda r: r['pathway'] == p and r['state'] == 'SL'))(pathway)
                mmv = mouse_means(records, met, signal, 'cocaine', f2, ('third',))
                a, b, used = paired(mmv, (0,), (2,), mice)
                if len(a) < 3:
                    continue
                r_ = test_pair(a, b)
                sink.write(f'  {METRIC_LABELS[met]:<24}{a.mean():>10.3f}{b.mean():>10.3f}'
                           f'{a.mean()-b.mean():>10.3f}{r_["t"]:>8.2f}{r_["p"]:>10.4f}'
                           f'{r_["d"]:>7.2f}{r_["n"]:>4d}\n')

            # ---- 3 state x third ----
            sink.write('\n[3] state x session third (cell means over mice)\n')
            for met in ('peak_rate', 'prominence', 'mean'):
                mmv = mouse_means(records, met, signal, 'cocaine', fp, ('state', 'third'))
                sink.write(f'  {METRIC_LABELS[met]}\n')
                for st in ('SL', 'nonSL'):
                    vals = []
                    for th in (0, 1, 2):
                        v = [mmv[mo][(st, th)] for mo in mice if (st, th) in mmv.get(mo, {})]
                        vals.append(f'{np.mean(v):.3f} (n={len(v)})' if v else '-')
                    sink.write(f'    {st:<7}' + ''.join(f'{x:>18}' for x in vals) + '\n')

            # ---- 4 across cocaine days ----
            sink.write('\n[4] across cocaine exposure (SL stretches)\n')
            for met in ('peak_rate', 'prominence', 'sd', 'mean'):
                f4 = (lambda p: (lambda r: r['pathway'] == p and r['state'] == 'SL'))(pathway)
                mmv = mouse_means(records, met, signal, 'cocaine', f4, ('trial',))
                day_means, ns = [], []
                for t in COCAINE:
                    v = [mmv[mo][(t,)] for mo in mice if (t,) in mmv.get(mo, {})]
                    day_means.append(np.mean(v) if v else np.nan)
                    ns.append(len(v))
                x = np.arange(5)[np.isfinite(day_means)]
                y = np.array(day_means)[np.isfinite(day_means)]
                slope = stats.linregress(x, y) if len(y) >= 3 else None
                sink.write(f'  {METRIC_LABELS[met]:<24}'
                           + ''.join(f'{v:>9.3f}' for v in day_means)
                           + (f'   trend r={slope.rvalue:+.2f} p={slope.pvalue:.3f}'
                              if slope else '') + '\n')
                sink.write(f'  {"  (n mice)":<24}' + ''.join(f'{n:>9d}' for n in ns) + '\n')

            # ---- 5 saline control ----
            sink.write('\n[5] SALINE control  SL vs non-SL\n')
            for met in ('peak_rate', 'prominence', 'mean'):
                mmv = mouse_means(records, met, signal, 'saline', fp, ('state',))
                a, b, used = paired(mmv, ('SL',), ('nonSL',), mice)
                if len(a) < 3:
                    sink.write(f'  {METRIC_LABELS[met]:<24} insufficient mice (N={len(a)})\n')
                    continue
                r_ = test_pair(a, b)
                sink.write(f'  {METRIC_LABELS[met]:<24}{a.mean():>10.3f}{b.mean():>10.3f}'
                           f'{a.mean()-b.mean():>10.3f}  p={r_["p"]:.4f}  N={r_["n"]}\n')

            # ---- velocity confound ----
            vm = mouse_means(records, 'velocity', signal, 'cocaine', fp, ('state',))
            a, b, _ = paired(vm, ('SL',), ('nonSL',), mice)
            if len(a) >= 3:
                r_ = test_pair(a, b)
                sink.write(f'\n[confound] mean velocity  SL={a.mean():.5f}  '
                           f'nonSL={b.mean():.5f}  ratio={b.mean()/a.mean():.2f}x  '
                           f'p={r_["p"]:.4f}\n')
            sink.write('\n')

    make_figures(records, mice_by_path, sink)
    event_analysis(
        events_state, sink, tag='state', figures=False,
        note='SUPERSEDED -- kept for the record, no figures emitted.\n'
             't=0 is the state-label boundary. Because the labelling window is\n'
             'centred, this LAGS the behaviour by roughly half the window, so the\n'
             'baseline can already contain licking. Aligning SL onsets to a real\n'
             'behavioural transition while aligning the non-SL control to a\n'
             'smoothed threshold crossing manufactured an apparent iSPN effect\n'
             '(p=0.0029) that vanishes under correct alignment (p=0.76). Use the\n'
             'trueonset section below instead.')
    event_analysis(
        events_beh, sink, tag='trueonset',
        note='t=0 is the licking transition that caused the state flip (first frame\n'
             'of the triggering bout for SL onsets; last frame of the final bout for\n'
             'SL offsets and for licking cessation before non-SL). Baseline is\n'
             'genuinely pre-licking, so latency IS readable here.')
    cluster_analysis(events_beh, sink, tag='trueonset')
    robustness(sink)
    sink.close()
    print(f'\nDone. Outputs in {OUT}')


def make_figures(records, mice_by_path, sink):
    for signal in ('z', 'df'):
        fig = plt.figure(figsize=(210 * mm, 150 * mm))
        gs = gridspec.GridSpec(3, 3, figure=fig, left=0.09, right=0.97,
                               top=0.90, bottom=0.08, wspace=0.45, hspace=0.75)
        unit = 'z' if signal == 'z' else r'$\Delta$F/F'

        for col, met in enumerate(('peak_rate', 'prominence', 'mean')):
            # row 0 : SL vs nonSL per pathway
            ax = fig.add_subplot(gs[0, col])
            for xi, pathway in enumerate(('dSPN', 'iSPN')):
                mice = sorted(mice_by_path[pathway])
                fp = (lambda p: (lambda r: r['pathway'] == p))(pathway)
                mmv = mouse_means(records, met, signal, 'cocaine', fp, ('state',))
                a, b, _ = paired(mmv, ('SL',), ('nonSL',), mice)
                if len(a) == 0:
                    continue
                for j, (vals, st) in enumerate(((b, 'nonSL'), (a, 'SL'))):
                    xpos = xi * 3 + j
                    ax.bar(xpos, vals.mean(), yerr=stats.sem(vals), width=0.75,
                           color=STATE_COLORS[st], alpha=0.85, capsize=2,
                           label=st if xi == 0 else None)
                    ax.plot(np.full(len(vals), xpos), vals, 'o', ms=2,
                            color='k', alpha=0.5, zorder=3)
                for k in range(len(a)):
                    ax.plot([xi * 3, xi * 3 + 1], [b[k], a[k]], '-', lw=0.4,
                            color='k', alpha=0.35, zorder=2)
            ax.set_xticks([0.5, 3.5]); ax.set_xticklabels(['dSPN', 'iSPN'], fontsize=7)
            ax.set_ylabel(f'{METRIC_LABELS[met]} ({unit})' if met != 'peak_rate'
                          else METRIC_LABELS[met], fontsize=7)
            ax.set_title('SL vs non-SL (cocaine)', fontsize=7, fontweight='bold')
            if col == 0:
                ax.legend(fontsize=5.5, frameon=False, loc='upper left')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

            # row 1 : SL across session thirds
            ax = fig.add_subplot(gs[1, col])
            for pathway in ('dSPN', 'iSPN'):
                mice = sorted(mice_by_path[pathway])
                f2 = (lambda p: (lambda r: r['pathway'] == p and r['state'] == 'SL'))(pathway)
                mmv = mouse_means(records, met, signal, 'cocaine', f2, ('third',))
                means, sems = [], []
                for th in (0, 1, 2):
                    v = [mmv[mo][(th,)] for mo in mice if (th,) in mmv.get(mo, {})]
                    means.append(np.mean(v) if v else np.nan)
                    sems.append(stats.sem(v) if len(v) > 1 else 0)
                ax.errorbar([0, 1, 2], means, yerr=sems, marker='o', ms=3, lw=1,
                            capsize=2, color=PATHWAY_COLORS[pathway], label=pathway)
            ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['early', 'mid', 'late'], fontsize=7)
            ax.set_ylabel(METRIC_LABELS[met], fontsize=7)
            ax.set_title('SL stretches across session', fontsize=7, fontweight='bold')
            if col == 0:
                ax.legend(fontsize=5.5, frameon=False)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

            # row 2 : across cocaine days
            ax = fig.add_subplot(gs[2, col])
            for pathway in ('dSPN', 'iSPN'):
                mice = sorted(mice_by_path[pathway])
                f4 = (lambda p: (lambda r: r['pathway'] == p and r['state'] == 'SL'))(pathway)
                mmv = mouse_means(records, met, signal, 'cocaine', f4, ('trial',))
                means, sems = [], []
                for t in COCAINE:
                    v = [mmv[mo][(t,)] for mo in mice if (t,) in mmv.get(mo, {})]
                    means.append(np.mean(v) if v else np.nan)
                    sems.append(stats.sem(v) if len(v) > 1 else 0)
                ax.errorbar(np.arange(5), means, yerr=sems, marker='o', ms=3, lw=1,
                            capsize=2, color=PATHWAY_COLORS[pathway], label=pathway)
            ax.set_xticks(range(5)); ax.set_xticklabels([f'C{i+1}' for i in range(5)], fontsize=7)
            ax.set_xlabel('Cocaine session', fontsize=7)
            ax.set_ylabel(METRIC_LABELS[met], fontsize=7)
            ax.set_title('SL stretches across exposure', fontsize=7, fontweight='bold')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

        fig.suptitle(f'Surface-licking states, photometry signal = {signal}',
                     fontsize=10, fontweight='bold')
        save_fig(fig, OUT, f'SL_states_{signal}')
        print(f'  figure -> SL_states_{signal}')


def find_boolean_clusters(mask):
    idx = np.where(np.asarray(mask, dtype=bool))[0]
    if idx.size == 0:
        return []
    return [c for c in np.split(idx, np.where(np.diff(idx) != 1)[0] + 1) if c.size]


def cluster_permutation(A, B=None, n_perm=10000, cluster_alpha=0.05, seed=0):
    """Paired two-sided sign-flip cluster permutation over timepoints.

    Same procedure as Int3 / upstream FigureS2. B=None tests A against zero.
    Handles the multiple comparisons across timepoints, and unlike a fixed
    response-window mean it is sensitive to biphasic shapes.
    """
    A = np.asarray(A, dtype=float)
    D = A if B is None else A - np.asarray(B, dtype=float)
    D = D[~np.isnan(D).any(axis=1)]
    n = D.shape[0]
    if n < 3:
        return dict(clusters=[], p_values=[], n=n)

    def tstat(d):
        mean = np.nanmean(d, axis=0)
        sem = stats.sem(d, axis=0, nan_policy='omit')
        return np.divide(mean, sem, out=np.zeros_like(mean),
                         where=np.isfinite(sem) & (sem > 0))

    t_obs = tstat(D)
    thr = stats.t.ppf(1 - cluster_alpha / 2, n - 1)
    clusters = find_boolean_clusters(np.abs(t_obs) > thr)
    masses = [float(np.sum(np.abs(t_obs[c]))) for c in clusters]

    rng = np.random.default_rng(seed)
    null = np.zeros(n_perm)
    for i in range(n_perm):
        tp = tstat(D * rng.choice([-1, 1], size=(n, 1)))
        pc = find_boolean_clusters(np.abs(tp) > thr)
        if pc:
            null[i] = max(np.sum(np.abs(tp[c])) for c in pc)
    p = [float((np.sum(null >= m) + 1) / (n_perm + 1)) for m in masses]
    return dict(clusters=clusters, p_values=p, masses=masses, t_obs=t_obs, n=n)


def cluster_analysis(events, sink, tag):
    """Time-resolved test of state specificity: SL onset vs non-SL onset."""
    x = (np.arange(PRE_EV + POST_EV) - PRE_EV) / FPS
    sink.write('=' * 70 + f'\nTIME-RESOLVED CLUSTER PERMUTATION  (alignment = {tag})\n')
    sink.write('=' * 70 + '\n')
    sink.write('Paired sign-flip cluster permutation over timepoints, 10000 perms,\n')
    sink.write('cluster-forming alpha 0.05. Sensitive to shape, unlike a window mean.\n')

    for signal in ('z', 'df'):
        sink.write(f'\n--- signal = {signal} ---\n')
        for pathway in ('dSPN', 'iSPN'):
            sa, ma, _ = event_mouse_traces(events, signal, pathway, 'SL', 'onset')
            sb, mb, _ = event_mouse_traces(events, signal, pathway, 'nonSL', 'onset')

            # each response vs its own baseline
            for stack, mice, name in ((sa, ma, 'SL onset vs 0'),
                                      (sb, mb, 'nonSL onset vs 0')):
                if stack.shape[0] < 3:
                    continue
                res = cluster_permutation(stack, seed=0)
                sink.write(f'  {pathway:<6}{name:<22}N={res["n"]}  ')
                sig = [(c, p) for c, p in zip(res['clusters'], res['p_values']) if p < 0.05]
                sink.write('no significant cluster\n' if not sig else
                           '; '.join(f'{x[c[0]]:+.1f} to {x[c[-1]]:+.1f}s p={p:.4f}'
                                     for c, p in sig) + '\n')

            # the state-specificity test
            common = [mo for mo in ma if mo in mb]
            if len(common) < 3:
                sink.write(f'  {pathway:<6}{"SL vs nonSL":<22}too few shared mice\n')
                continue
            A = np.vstack([sa[ma.index(mo)] for mo in common])
            B = np.vstack([sb[mb.index(mo)] for mo in common])
            res = cluster_permutation(A, B, seed=1)
            sig = [(c, p) for c, p in zip(res['clusters'], res['p_values']) if p < 0.05]
            sink.write(f'  {pathway:<6}{"SL vs nonSL":<22}N={res["n"]}  ')
            sink.write('no significant cluster\n' if not sig else
                       '; '.join(f'{x[c[0]]:+.1f} to {x[c[-1]]:+.1f}s p={p:.4f}'
                                 for c, p in sig) + '\n')
    sink.write('\n')


def example_ethograms(CMT, sink, n_examples=4):
    """Show what a 'stretch' actually is, on real sessions.

    The VizDetect design calls this the re-shade review: render the labeller's
    output over the raw behaviour so the boundaries can be eyeballed.
    """
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    beh_names = data_io.CANONICAL_BEHAVIORS
    cmap = ListedColormap(data_io.CANONICAL_COLORS)
    rm_mouse = {m for _, m in photom_exclusions['remove_mouse']}
    rm_hem = {(m, h) for _, m, h in photom_exclusions['remove_hemisphere']}

    picks = []
    for c in CMT:
        for m in CMT[c]:
            if m in rm_mouse:
                continue
            pathway = 'dSPN' if get_pathway(c, m) == 'drd1' else 'iSPN'
            for t in ('cocaine1', 'cocaine5'):
                if t not in CMT[c][m] or 'photom' not in CMT[c][m][t]:
                    continue
                picks.append((c, m, t, pathway))
    # one cocaine1 and one cocaine5 per pathway
    chosen = []
    for pathway in ('dSPN', 'iSPN'):
        for t in ('cocaine1', 'cocaine5'):
            cand = [p for p in picks if p[3] == pathway and p[2] == t]
            if cand:
                chosen.append(cand[0])
    chosen = chosen[:n_examples]

    for c, m, t, pathway in chosen:
        e = CMT[c][m][t]
        preds = e['merged']
        n = preds.size
        sl = np.isin(preds, [FL, WL]).astype(float)
        frac = rolling_mean(sl, W)
        stretches = label_stretches(preds)
        hems = [h for h in ('left', 'right')
                if isinstance(e['photom'].get(h), dict) and (m, h) not in rm_hem]
        z = np.nanmean(np.vstack([e['photom'][h]['z'] for h in hems]), axis=0)
        tmin = np.arange(n) / FPS / 60

        fig = plt.figure(figsize=(200 * mm, 105 * mm))
        gs = gridspec.GridSpec(4, 1, figure=fig, left=0.09, right=0.98,
                               top=0.87, bottom=0.30, hspace=0.45,
                               height_ratios=[0.5, 1.1, 1.1, 1.1])

        ax0 = fig.add_subplot(gs[0])
        ax0.imshow(preds.reshape(1, -1), aspect='auto', cmap=cmap, vmin=-0.5, vmax=8.5,
                   extent=[tmin[0], tmin[-1], 0, 1], interpolation='nearest')
        ax0.set_yticks([]); ax0.set_ylabel('Behaviour', fontsize=6, rotation=0,
                                           ha='right', va='center')
        ax0.set_xticks([])

        ax1 = fig.add_subplot(gs[1])
        ax1.plot(tmin, frac, lw=0.7, color='k')
        ax1.axhline(HI, color=STATE_COLORS['SL'], ls='--', lw=0.7)
        ax1.axhline(LO, color=STATE_COLORS['nonSL'], ls='--', lw=0.7)
        ax1.set_ylabel(f'SL fraction\n({W//FPS}s window)', fontsize=6)
        ax1.set_ylim(-0.05, 1.05); ax1.set_xticks([])

        ax2 = fig.add_subplot(gs[2])
        for st, a, b, purity in stretches:
            ax2.axvspan(tmin[a], tmin[min(b, n - 1)], color=STATE_COLORS[st], alpha=0.55, lw=0)
        ax2.set_ylim(0, 1); ax2.set_yticks([])
        ax2.set_ylabel('Admitted\nstretches', fontsize=6)
        ax2.set_xticks([])

        ax3 = fig.add_subplot(gs[3])
        ax3.plot(tmin, z, lw=0.35, color='#333333')
        for st, a, b, purity in stretches:
            ax3.axvspan(tmin[a], tmin[min(b, n - 1)], color=STATE_COLORS[st], alpha=0.18, lw=0)
        ax3.set_ylabel('Photometry\n(z)', fontsize=6)
        ax3.set_xlabel('Time in session (min)', fontsize=7)

        for ax in (ax0, ax1, ax2, ax3):
            ax.set_xlim(tmin[0], tmin[-1])
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
            ax.tick_params(labelsize=6)

        n_sl = sum(1 for s in stretches if s[0] == 'SL')
        n_ns = len(stretches) - n_sl
        unassigned = 100 * (1 - sum(b - a for _, a, b, _ in stretches) / n)
        fig.suptitle(f'{pathway} · {m} · {t}   —   {n_sl} SL and {n_ns} non-SL stretches, '
                     f'{unassigned:.0f}% of session unassigned',
                     fontsize=8, fontweight='bold')
        handles = [Patch(facecolor=data_io.CANONICAL_COLORS[i], label=beh_names[i])
                   for i in range(9)]
        handles += [Patch(facecolor=STATE_COLORS['SL'], alpha=0.55, label='SL stretch'),
                    Patch(facecolor=STATE_COLORS['nonSL'], alpha=0.55, label='non-SL stretch')]
        fig.legend(handles=handles, ncol=6, loc='lower center',
                   bbox_to_anchor=(0.5, 0.02), fontsize=5.5, frameon=False)
        save_fig(fig, OUT, f'ethogram_{pathway}_{m}_{t}')
        print(f'  ethogram -> ethogram_{pathway}_{m}_{t}')
        sink.write(f'  example {pathway} {m} {t}: {n_sl} SL, {n_ns} non-SL stretches, '
                   f'{unassigned:.0f}% unassigned\n')


def event_analysis(events, sink, tag='state', note='', figures=True):
    """Transition-aligned traces: what the per-stretch averages would wash out."""
    sink.write('=' * 70 + f'\nEVENT-ALIGNED ANALYSIS  (alignment = {tag})\n' + '=' * 70 + '\n')
    if note:
        sink.write(note + '\n')
    sink.write(f'pre={PRE_EV//FPS}s  post={POST_EV//FPS}s  '
               f'baseline=first {BASE_EV//FPS}s of pre-window  '
               f'response=first {RESP_EV//FPS}s after transition\n')
    sink.write('Events with a contaminated pre-window (another stretch of the same\n'
               'state inside it) are excluded. Traces averaged within mouse first.\n\n')

    x = (np.arange(PRE_EV + POST_EV) - PRE_EV) / FPS
    resp = slice(PRE_EV, PRE_EV + RESP_EV)

    for signal in ('z', 'df'):
        fig = plt.figure(figsize=(200 * mm, 110 * mm))
        gs = gridspec.GridSpec(2, 3, figure=fig, left=0.09, right=0.97,
                               top=0.88, bottom=0.10, wspace=0.35, hspace=0.55)
        unit = 'z' if signal == 'z' else r'$\Delta$F/F'
        sink.write(f'--- signal = {signal} ---\n')
        sink.write(f'  {"pathway":<8}{"event":<16}{"n mice":>7}{"n events":>10}'
                   f'{"resp mean":>11}{"t":>8}{"p":>9}{"dz":>7}\n')

        for row, pathway in enumerate(('dSPN', 'iSPN')):
            for col, (state, kind, title) in enumerate((
                    ('SL', 'onset', 'Entering surface licking'),
                    ('SL', 'offset', 'Leaving surface licking'),
                    ('nonSL', 'onset', 'Entering non-SL (control)'))):
                ax = fig.add_subplot(gs[row, col])
                stack, mice, n_ev = event_mouse_traces(events, signal, pathway, state, kind)
                if stack.shape[0] >= 2:
                    mean = np.nanmean(stack, axis=0)
                    sem = stats.sem(stack, axis=0, nan_policy='omit')
                    ax.plot(x, mean, lw=1.2, color=PATHWAY_COLORS[pathway])
                    ax.fill_between(x, mean - sem, mean + sem, alpha=0.25,
                                    color=PATHWAY_COLORS[pathway], lw=0)
                    r_ = np.nanmean(stack[:, resp], axis=1)
                    t_, p_ = stats.ttest_1samp(r_, 0.0)
                    d_ = float(np.mean(r_) / np.std(r_, ddof=1)) if np.std(r_, ddof=1) > 0 else np.nan
                    sink.write(f'  {pathway:<8}{state + " " + kind:<16}{len(mice):>7}'
                               f'{n_ev:>10}{np.mean(r_):>11.3f}{t_:>8.2f}{p_:>9.4f}{d_:>7.2f}\n')
                    if p_ < 0.05:
                        ax.axhspan(*ax.get_ylim(), xmin=0.33, xmax=0.5, alpha=0)
                        ax.text(0.98, 0.95, f'p={p_:.3f}', transform=ax.transAxes,
                                ha='right', va='top', fontsize=6, fontweight='bold')
                ax.axvline(0, color='k', ls='--', lw=0.7)
                ax.axhline(0, color='gray', lw=0.5)
                ax.set_title(f'{pathway} · {title}\n(n={len(mice)} mice, {n_ev} events)',
                             fontsize=7, fontweight='bold')
                ax.set_xlabel('Time from transition (s)', fontsize=7)
                if col == 0:
                    ax.set_ylabel(f'{unit}, baseline-subtracted', fontsize=7)
                ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

        fig.suptitle(f'Photometry aligned to surface-licking state transitions '
                     f'(signal = {signal})', fontsize=10, fontweight='bold')
        if figures:
            save_fig(fig, OUT, f'SL_transitions_{tag}_{signal}')
            print(f'  figure -> SL_transitions_{tag}_{signal}')
        else:
            plt.close(fig)

        # SL onset vs non-SL onset, paired within mouse
        sink.write(f'\n  SL onset vs non-SL onset (paired within mouse, {signal}):\n')
        for pathway in ('dSPN', 'iSPN'):
            sa, ma, _ = event_mouse_traces(events, signal, pathway, 'SL', 'onset')
            sb, mb, _ = event_mouse_traces(events, signal, pathway, 'nonSL', 'onset')
            common = [mo for mo in ma if mo in mb]
            if len(common) < 3:
                sink.write(f'    {pathway}: too few shared mice (N={len(common)})\n')
                continue
            a = np.array([np.nanmean(sa[ma.index(mo), resp]) for mo in common])
            b = np.array([np.nanmean(sb[mb.index(mo), resp]) for mo in common])
            r_ = test_pair(a, b)
            sink.write(f'    {pathway}: SL={a.mean():+.3f}  nonSL={b.mean():+.3f}  '
                       f'diff={a.mean()-b.mean():+.3f}  t={r_["t"]:.2f}  '
                       f'p={r_["p"]:.4f}  dz={r_["d"]:.2f}  N={r_["n"]}\n')
        sink.write('\n')


def robustness(sink):
    """Does the primary contrast survive other window/threshold choices?"""
    sink.write('=' * 70 + '\nROBUSTNESS: primary contrast under other parameters\n')
    sink.write('=' * 70 + '\n')
    CMT = data_io.load_CMT(flatten=True, verbose=False)
    sink.write(f'  {"window":>8}{"hi/lo":>12}{"pathway":>9}{"SL":>9}{"nonSL":>9}'
               f'{"p":>9}{"N":>4}   (peak rate, z)\n')
    for w in W_SWEEP:
        for hi, lo in THR_SWEEP:
            recs = collect(CMT, w=w, hi=hi, lo=lo)
            mbp = defaultdict(set)
            for r in recs:
                mbp[r['pathway']].add(r['mouse'])
            for pathway in ('dSPN', 'iSPN'):
                mice = sorted(mbp[pathway])
                fp = (lambda p: (lambda r: r['pathway'] == p))(pathway)
                mmv = mouse_means(recs, 'peak_rate', 'z', 'cocaine', fp, ('state',))
                a, b, _ = paired(mmv, ('SL',), ('nonSL',), mice)
                if len(a) < 3:
                    continue
                r_ = test_pair(a, b)
                sink.write(f'  {w//FPS:>7}s{f"{hi}/{lo}":>12}{pathway:>9}'
                           f'{a.mean():>9.2f}{b.mean():>9.2f}{r_["p"]:>9.4f}{r_["n"]:>4d}\n')
    del CMT


if __name__ == '__main__':
    main()

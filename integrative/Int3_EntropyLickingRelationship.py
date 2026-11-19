"""
Integrative Analysis 3: Entropy-Licking Relationship (manuscript method)

Switch entropy is measured the way the manuscript measures it (companion repo
shalomitay/opponent-vls-circuits-code, code/FigureS2.py panel S2E): in sliding
180 s windows with a 30 s stride, over saline3 + cocaine1-5, compared against
saline3 with a paired sign-flip cluster permutation test.

Why not a single number per session: switch entropy is computed on the *bout*
sequence, so it is sensitive to how the 9-class alphabet is partitioned. The
Apr26 locomotion threshold splits locomotor episodes into move/pause
micro-episodes -- in saline3 the Locomotion<->Stationary pair alone accounts
for 42.6% of all bout transitions (29.6% under Dec24), which drags the
occupancy-weighted entropy down at the saline end while barely touching
cocaine5. A whole-session scalar therefore mostly tracks that repartitioning.
The sliding-window trace plus cluster inference is what the manuscript reports
and is what this script now produces.

Panels
  A  switch entropy, sliding traces, cluster bars vs saline3
  B  surface licking fraction, same treatment
  C  within-mouse coupling between the two, across matched windows

Outputs
  Int3_EntropyLickingRelationship_{4,6}cohorts.png / .pdf
  Int3_EntropyLickingRelationship_{4,6}cohorts_statistics.txt
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'revision'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

from revision_utils import (
    mm, n_behaviors, FPS, second,
    cohorts_4, cohorts_6,
    load_CMT, flatten_CMT, setup_style, save_fig,
    calc_switch_entropy, trace_trials, trace_colors,
)

setup_style()
output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(output_folder, exist_ok=True)

FL, WL = 2, 3                      # surface licking classes
WINDOW = 180 * second              # manuscript: 180 s
STRIDE = 30 * second               # manuscript: 30 s
MIN_SESSION_MINUTES = 29           # manuscript inclusion rule
N_PERMUTATIONS = 10_000
CLUSTER_ALPHA = 0.05


# ---------------------------------------------------------------------------
# Sliding metrics
# ---------------------------------------------------------------------------
def sliding_switch_entropy(preds):
    """Occupancy-weighted switch entropy per window (revision_utils == upstream)."""
    return calc_switch_entropy(preds, WINDOW, STRIDE, n_behaviors)


def sliding_surface_licking(preds):
    """Fraction of frames spent floor- or wall-licking, per window."""
    values = []
    for start in range(0, preds.size, STRIDE):
        stop = start + WINDOW
        if stop <= preds.size:
            w = preds[start:stop]
            values.append(float(np.mean((w == FL) | (w == WL))))
    return np.asarray(values, dtype=float)


def prepare_metric_traces(MT, mice, trials, metric):
    """One sliding trace per (trial, mouse), applying the session-length rule."""
    traces = {t: {} for t in trials}
    for t in trials:
        for m in mice:
            if t not in MT[m]:
                continue
            preds = MT[m][t]['merged']
            if preds.size // (60 * second) < MIN_SESSION_MINUTES:
                continue
            traces[t][m] = metric(preds)
    return traces


# ---------------------------------------------------------------------------
# Cluster-based inference (mirrors upstream FigureS2)
# ---------------------------------------------------------------------------
def find_boolean_clusters(mask):
    idx = np.where(np.asarray(mask, dtype=bool))[0]
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) != 1)[0] + 1
    return [c for c in np.split(idx, splits) if c.size]


def paired_cluster_permutation_test(first, second_, n_permutations=N_PERMUTATIONS,
                                    cluster_alpha=CLUSTER_ALPHA, random_seed=0):
    """Paired, two-sided, sign-flip cluster permutation test across mice."""
    first = np.asarray(first, dtype=float)
    second_ = np.asarray(second_, dtype=float)
    if first.shape != second_.shape:
        raise ValueError('Paired matrices must have identical shapes.')

    diffs = first - second_
    diffs = diffs[~np.isnan(diffs).any(axis=1)]
    n_mice = diffs.shape[0]
    if n_mice < 2:
        raise ValueError(f'Not enough valid mice for cluster test: N={n_mice}')

    def t_stat(d):
        mean = np.nanmean(d, axis=0)
        sem = stats.sem(d, axis=0, nan_policy='omit')
        return np.divide(mean, sem, out=np.zeros_like(mean),
                         where=np.isfinite(sem) & (sem > 0))

    t_obs = t_stat(diffs)
    threshold = stats.t.ppf(1 - cluster_alpha / 2, n_mice - 1)
    clusters = find_boolean_clusters(np.abs(t_obs) > threshold)
    masses = [float(np.sum(np.abs(t_obs[c]))) for c in clusters]

    rng = np.random.default_rng(random_seed)
    null_max = np.zeros(n_permutations, dtype=float)
    for i in range(n_permutations):
        signs = rng.choice([-1, 1], size=(n_mice, 1))
        t_perm = t_stat(diffs * signs)
        perm_clusters = find_boolean_clusters(np.abs(t_perm) > threshold)
        if perm_clusters:
            null_max[i] = max(np.sum(np.abs(t_perm[c])) for c in perm_clusters)

    p_values = [float((np.sum(null_max >= m) + 1) / (n_permutations + 1)) for m in masses]
    return {'t_observed': t_obs, 'clusters': clusters, 'cluster_masses': masses,
            'cluster_p_values': p_values, 'n_mice': n_mice,
            'significant_clusters': [(c, p) for c, p in zip(clusters, p_values) if p < 0.05]}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_sliding_metric(ax, traces, ylabel, title, panel, sink, metric_name):
    lengths = [len(v) for tr in traces.values() for v in tr.values()]
    if not lengths:
        raise ValueError(f'No traces available for {metric_name}')
    n = int(np.min(lengths))
    x_min = np.arange(n) * (STRIDE / (60 * second))   # window index -> minutes

    means, uppers = {}, []
    for ti, t in enumerate(trace_trials):
        arrays = [traces[t][m][:n] for m in sorted(traces[t])]
        if not arrays:
            continue
        M = np.asarray(arrays, dtype=float)
        mean = np.nanmean(M, axis=0)
        sem = stats.sem(M, axis=0, nan_policy='omit')
        means[t] = mean
        uppers.append(mean + sem)
        ax.plot(x_min, mean, label=t, color=trace_colors[ti], lw=1)
        ax.fill_between(x_min, mean - sem, mean + sem, alpha=0.2, color=trace_colors[ti])

    y_base = float(np.nanmax(np.concatenate(uppers)))
    stacked = np.concatenate(list(means.values()))
    y_range = float(np.nanmax(stacked) - np.nanmin(stacked))
    y_step = 0.04 * y_range if y_range > 0 else 0.05

    sink.write(f'\n================ {metric_name} ================\n')
    sink.write('Reference condition: saline3\n')
    sink.write('Paired sign-flip cluster permutation test\n')
    sink.write(f'Window = {WINDOW // second} s; stride = {STRIDE // second} s; '
               f'permutations = {N_PERMUTATIONS}\n')
    sink.write(f'Cluster-forming alpha = {CLUSTER_ALPHA}; cluster alpha = 0.05\n')

    for bar_index, t in enumerate([t for t in trace_trials if t.startswith('cocaine')]):
        common = sorted(set(traces['saline3']) & set(traces[t]))
        if len(common) < 2:
            sink.write(f'{t} vs saline3: skipped; N={len(common)}\n')
            continue
        m_len = min([n] + [min(len(traces['saline3'][m]), len(traces[t][m])) for m in common])
        sal = np.asarray([traces['saline3'][m][:m_len] for m in common])
        coc = np.asarray([traces[t][m][:m_len] for m in common])

        res = paired_cluster_permutation_test(coc, sal, random_seed=bar_index)
        y_bar = y_base + y_step * (bar_index + 1)
        for cluster, p in res['significant_clusters']:
            ax.hlines(y=y_bar, xmin=x_min[cluster[0]], xmax=x_min[cluster[-1]],
                      color=trace_colors[trace_trials.index(t)], lw=1)
            ax.vlines([x_min[cluster[0]], x_min[cluster[-1]]],
                      y_bar - y_step * 0.12, y_bar + y_step * 0.12,
                      color=trace_colors[trace_trials.index(t)], lw=1)

        sink.write(f'\n{t} vs saline3 (N={res["n_mice"]} mice, {m_len} windows)\n')
        if not res['clusters']:
            sink.write('  no supra-threshold clusters\n')
        for ci, (cluster, mass, p) in enumerate(
                zip(res['clusters'], res['cluster_masses'], res['cluster_p_values']), start=1):
            sink.write(f'  cluster {ci}: windows {cluster[0]}-{cluster[-1]} '
                       f'({x_min[cluster[0]]:.1f}-{x_min[cluster[-1]]:.1f} min), '
                       f'mass={mass:.2f}, p={p:.4f}'
                       f'{"  *" if p < 0.05 else ""}\n')

    ax.set_xlabel('Time in session (min)', fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=8, fontweight='bold', pad=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(-0.28, 1.15, panel, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')


# ---------------------------------------------------------------------------
def run_analysis(cohorts, suffix):
    print(f'\n=== Entropy-Licking Relationship, manuscript method ({suffix}) ===')
    CMT = load_CMT()
    MT, mouse_to_cohort = flatten_CMT(CMT, cohorts)
    del CMT

    mice = [m for m in MT if all(t in MT[m] for t in trace_trials)]
    print(f'{len(mice)} mice with saline3 + cocaine1-5')

    ent_traces = prepare_metric_traces(MT, mice, trace_trials, sliding_switch_entropy)
    sl_traces = prepare_metric_traces(MT, mice, trace_trials, sliding_surface_licking)
    n_incl = len(set(m for t in trace_trials for m in ent_traces[t]))
    print(f'{n_incl} mice pass the >={MIN_SESSION_MINUTES} min session rule')

    stats_path = os.path.join(output_folder,
                              f'Int3_EntropyLickingRelationship_{suffix}_statistics.txt')
    with open(stats_path, 'w') as sink:
        sink.write(f'Int3 - switch entropy and surface licking ({suffix})\n')
        sink.write('Method follows the manuscript companion repo, FigureS2 panel S2E.\n')
        sink.write(f'Mice with saline3 + cocaine1-5: {len(mice)}; passing length rule: {n_incl}\n')

        fig = plt.figure(figsize=(200 * mm, 75 * mm))
        gs = gridspec.GridSpec(1, 3, figure=fig, left=0.09, right=0.97,
                               top=0.84, bottom=0.24, wspace=0.45)

        ax_a = fig.add_subplot(gs[0])
        plot_sliding_metric(ax_a, ent_traces, 'Switch entropy (bits)',
                            'Switch entropy\nwithin sessions', 'A', sink, 'Switch entropy')

        ax_b = fig.add_subplot(gs[1])
        plot_sliding_metric(ax_b, sl_traces, 'Surface licking fraction',
                            'Surface licking\nwithin sessions', 'B', sink, 'Surface licking')

        # --- Panel C: within-mouse coupling across matched windows ---
        ax_c = fig.add_subplot(gs[2])
        per_mouse_r, pooled_x, pooled_y = [], [], []
        for m in mice:
            xs, ys = [], []
            for t in trace_trials:
                if m in ent_traces[t] and m in sl_traces[t]:
                    k = min(len(ent_traces[t][m]), len(sl_traces[t][m]))
                    ys.append(ent_traces[t][m][:k])
                    xs.append(sl_traces[t][m][:k])
            if not xs:
                continue
            xs = np.concatenate(xs)
            ys = np.concatenate(ys)
            ok = np.isfinite(xs) & np.isfinite(ys)
            if ok.sum() < 10 or np.std(xs[ok]) == 0:
                continue
            r, _ = stats.pearsonr(xs[ok], ys[ok])
            per_mouse_r.append(r)
            pooled_x.append(xs[ok])
            pooled_y.append(ys[ok])

        per_mouse_r = np.asarray(per_mouse_r, dtype=float)
        z = np.arctanh(np.clip(per_mouse_r, -0.999999, 0.999999))
        t_stat_c, p_c = stats.ttest_1samp(z, 0.0)
        mean_r = float(np.tanh(np.mean(z)))

        ax_c.scatter(np.concatenate(pooled_x), np.concatenate(pooled_y),
                     s=2, alpha=0.12, color='#3399cc', edgecolors='none', rasterized=True)
        ax_c.set_xlabel('Surface licking fraction (window)', fontsize=8)
        ax_c.set_ylabel('Switch entropy (bits, window)', fontsize=8)
        sig = '***' if p_c < 0.001 else '**' if p_c < 0.01 else '*' if p_c < 0.05 else 'n.s.'
        ax_c.set_title(f'Within-mouse coupling\nmean r={mean_r:.2f}, {sig} (N={len(per_mouse_r)} mice)',
                       fontsize=8, fontweight='bold', pad=14)
        ax_c.spines['top'].set_visible(False)
        ax_c.spines['right'].set_visible(False)
        ax_c.text(-0.28, 1.15, 'C', transform=ax_c.transAxes,
                  fontsize=12, fontweight='bold', va='top')

        sink.write('\n================ Within-mouse coupling ================\n')
        sink.write('Per-mouse Pearson r between window-wise surface licking and switch\n')
        sink.write('entropy, pooled over saline3 + cocaine1-5; Fisher-z, one-sample t-test.\n')
        sink.write(f'N = {len(per_mouse_r)} mice\n')
        sink.write(f'mean r (back-transformed) = {mean_r:.4f}\n')
        sink.write(f't({len(per_mouse_r) - 1}) = {t_stat_c:.3f}, p = {p_c:.3g}\n')
        sink.write(f'range of per-mouse r: [{per_mouse_r.min():.3f}, {per_mouse_r.max():.3f}]\n')
        print(f'Within-mouse coupling: mean r={mean_r:.3f}, '
              f't={t_stat_c:.2f}, p={p_c:.3g}, N={len(per_mouse_r)}')

        handles, labels = ax_a.get_legend_handles_labels()
        fig.legend(handles, labels, ncol=len(labels), loc='lower center',
                   bbox_to_anchor=(0.5, 0.01), fontsize=6.5, frameon=False)

        save_fig(fig, output_folder, f'Int3_EntropyLickingRelationship_{suffix}')

    print(f'  statistics -> {os.path.basename(stats_path)}')


run_analysis(cohorts_4, '4cohorts')
run_analysis(cohorts_6, '6cohorts')
print('\nDone - entropy figures saved (manuscript method).')

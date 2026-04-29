#%% SuppFig: All 9 Behaviors Across Cocaine Sessions
"""
Addresses R4-Fig2 (expand quantification), R4-BehavExpand (other behaviors),
R4-Fig6 (persistence claim).

Panels:
  Row 1 (A–I): Fraction-of-frames for each of 9 behaviors across saline1-3 + cocaine1-5
  Row 2 (J–L): Transition rate, distinct behaviors per window, switch entropy timecourses
  Row 3 (M–Q): Bout duration CDFs for key behaviors (floor lick, wall lick, grooming, locomotion, rearing)

Uses CMT_Aug24.pkl, 4 cohorts (excl. hm3Dq).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats
import helper_functions as hf

from revision_utils import (
    mm, second, minute, n_behaviors, behaviors, colors, short_labels,
    trials_sal_coc as trials, trial_colors, cohorts_6 as cohorts,
    cohort_labels, lut, FPS,
    load_CMT, flatten_CMT, setup_style, save_fig,
    calc_switch_entropy, calc_behavioral_entropy, transition_rate
)

setup_style()
output_folder = 'revision/output/'
os.makedirs(output_folder, exist_ok=True)

# ---------------------------------------------------------------------------
# Load & flatten
# ---------------------------------------------------------------------------
CMT = load_CMT('Dec24')
MT, mouse_to_cohort = flatten_CMT(CMT, cohorts)
del CMT
print(f'Loaded {len(MT)} mice from {len(cohorts)} cohorts')

# Identify mice with all 8 sessions
valid_mice = [m for m in MT if all(t in MT[m] for t in trials)]
n_mice = len(valid_mice)
print(f'N = {n_mice} mice with complete data')

# ---------------------------------------------------------------------------
# Panel A–I: Per-behavior fraction across sessions
# ---------------------------------------------------------------------------
print('\nComputing per-behavior fractions...')
fractions = np.full((n_mice, n_behaviors, len(trials)), np.nan)

for m_idx, mouse in enumerate(valid_mice):
    for t_idx, trial in enumerate(trials):
        preds = MT[mouse][trial]['merged']
        total = preds.size
        for b in range(n_behaviors):
            fractions[m_idx, b, t_idx] = np.count_nonzero(preds == b) / total

# Stats: Spearman correlation across sessions + paired t-test (saline vs late cocaine)
print('\n--- Per-behavior statistics (saline avg vs late cocaine avg) ---')
stat_results = {}
for b in range(n_behaviors):
    sal_vals = np.nanmean(fractions[:, b, :3], axis=1)   # saline1-3
    late_vals = np.nanmean(fractions[:, b, 5:8], axis=1)  # cocaine3-5
    t_stat, p_val = stats.ttest_rel(sal_vals, late_vals)
    d = np.mean(sal_vals - late_vals) / (np.std(sal_vals - late_vals, ddof=1) + 1e-12)
    session_means = np.nanmean(fractions[:, b, :], axis=0)
    rho, p_rho = stats.spearmanr(np.arange(len(trials)), session_means)
    stat_results[b] = {'t': t_stat, 'p': p_val, 'd': d, 'rho': rho, 'p_rho': p_rho}
    print(f'  {behaviors[b]:<16} sal={np.mean(sal_vals):.3f} late={np.mean(late_vals):.3f} '
          f't={t_stat:.2f} p={p_val:.4f} d={d:.2f} rho={rho:.2f}')

# ---------------------------------------------------------------------------
# Panel J–L: Within-session timecourse metrics
# ---------------------------------------------------------------------------
print('\nComputing within-session metrics...')
window_frames = 3 * minute
step_frames = 30 * second
max_duration = 27 * minute
n_windows = int((max_duration - window_frames) / step_frames) + 1
window_centers_min = np.array([(w * step_frames + window_frames / 2) / minute for w in range(n_windows)])

trace_trials = ['saline3', 'cocaine1', 'cocaine2', 'cocaine3', 'cocaine4', 'cocaine5']
trace_colors_local = ['#808080', '#FFA500', '#FF8C00', '#FF6347', '#E60000', '#990000']

def n_distinct(preds, n_beh):
    return len(np.unique(preds.astype(int)))

metric_fns = {
    'Transition rate (switches/min)': transition_rate,
    'Distinct behaviors': n_distinct,
    'Switch entropy (bits)': lambda p, n: calc_switch_entropy(p, len(p), len(p), n)[0] if len(p) > 0 else np.nan,
}

# For switch entropy, use the windowed version directly
# Pre-compute all metric time courses
metric_traces = {}
for metric_name in ['Transition rate (switches/min)', 'Distinct behaviors']:
    metric_traces[metric_name] = {}
    fn = metric_fns[metric_name]
    for trial in trace_trials:
        traces = np.full((n_mice, n_windows), np.nan)
        for m_idx, mouse in enumerate(valid_mice):
            if trial in MT[mouse]:
                preds = MT[mouse][trial]['merged']
                for w in range(n_windows):
                    start = w * step_frames
                    end = start + window_frames
                    if end <= len(preds):
                        traces[m_idx, w] = fn(preds[start:end], n_behaviors)
        metric_traces[metric_name][trial] = traces

# Switch entropy — use windowed calc directly
metric_traces['Switch entropy (bits)'] = {}
for trial in trace_trials:
    traces = np.full((n_mice, n_windows), np.nan)
    for m_idx, mouse in enumerate(valid_mice):
        if trial in MT[mouse]:
            preds = MT[mouse][trial]['merged']
            sw = calc_switch_entropy(preds, window_frames, step_frames, n_behaviors)
            traces[m_idx, :min(len(sw), n_windows)] = sw[:n_windows]
    metric_traces['Switch entropy (bits)'][trial] = traces

print('Within-session traces computed')

# ---------------------------------------------------------------------------
# Panel M–Q: Bout duration CDFs (saline3 vs cocaine5)
# ---------------------------------------------------------------------------
print('\nComputing bout durations...')
bout_behaviors = [2, 3, 4, 7, 6]  # Floor lick, Wall lick, Grooming, Locomotion, Rearing
bout_behavior_names = ['Floor licking', 'Wall licking', 'Grooming', 'Locomotion', 'Rearing']
bout_theta = 8  # frames ~0.5s

bout_durations = {}
for trial in ['saline3', 'cocaine5']:
    bout_durations[trial] = {b: [] for b in bout_behaviors}
    for mouse in valid_mice:
        if trial in MT[mouse]:
            preds = MT[mouse][trial]['merged']
            for b in bout_behaviors:
                bouts = hf.segment_bouts(preds, b, bout_theta)
                bout_durations[trial][b].extend([l / FPS for l in bouts['length']])

# KS tests
print('\n--- Bout duration KS tests (saline3 vs cocaine5) ---')
for b, bname in zip(bout_behaviors, bout_behavior_names):
    s3 = np.array(bout_durations['saline3'][b])
    c5 = np.array(bout_durations['cocaine5'][b])
    if len(s3) > 2 and len(c5) > 2:
        ks, p = stats.ks_2samp(s3, c5)
        print(f'  {bname:<16} saline3={np.median(s3):.1f}s (n={len(s3)}) '
              f'cocaine5={np.median(c5):.1f}s (n={len(c5)}) KS={ks:.3f} p={p:.4f}')

# ===========================================================================
# BUILD THE FIGURE
# ===========================================================================
print('\nGenerating figure...')
fig = plt.figure(figsize=(180 * mm, 220 * mm))
outer = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[1.4, 0.8, 0.8], hspace=0.45)

# --- Row 1: Per-behavior fraction (3x3 grid) ---
gs_top = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=outer[0], hspace=0.75, wspace=0.35)

for b in range(n_behaviors):
    row, col = divmod(b, 3)
    ax = fig.add_subplot(gs_top[row, col])
    mean_vals = np.nanmean(fractions[:, b, :], axis=0)
    sem_vals = np.nanstd(fractions[:, b, :], axis=0, ddof=0) / np.sqrt(n_mice)

    # Bar plot with session colors
    for t_idx in range(len(trials)):
        ax.bar(t_idx, mean_vals[t_idx], color=trial_colors[t_idx], width=0.7, alpha=0.8)
        ax.errorbar(t_idx, mean_vals[t_idx], yerr=sem_vals[t_idx],
                    color='black', capsize=2, capthick=0.5, lw=0.5, fmt='none')

    ax.set_title(behaviors[b], fontsize=7, fontweight='bold', color=colors[b], pad=12)
    ax.set_xticks(range(len(trials)))
    ax.set_xticklabels(['S1','S2','S3','C1','C2','C3','C4','C5'], fontsize=4.5, rotation=45)
    ax.tick_params(labelsize=5)
    if col == 0:
        ax.set_ylabel('Fraction', fontsize=6)

    # Significance marker
    p = stat_results[b]['p']
    if p < 0.001:
        sig = '***'
    elif p < 0.01:
        sig = '**'
    elif p < 0.05:
        sig = '*'
    else:
        sig = ''
    if sig:
        ymax = max(mean_vals) + max(sem_vals) * 1.5
        ax.text(5.5, ymax, sig, ha='center', fontsize=7, fontweight='bold')

    panel_label = chr(ord('A') + b)
    ax.text(-0.2, 1.15, panel_label, transform=ax.transAxes, fontsize=9, fontweight='bold', va='top')

# --- Row 2: Within-session metric timecourses (1x3) ---
gs_mid = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], wspace=0.35)

for m_idx, (metric_name, panel_label) in enumerate([
    ('Switch entropy (bits)', 'J'),
    ('Distinct behaviors', 'K'),
    ('Transition rate (switches/min)', 'L'),
]):
    ax = fig.add_subplot(gs_mid[m_idx])
    for t_idx, trial in enumerate(trace_trials):
        traces = metric_traces[metric_name][trial]
        valid_n = np.maximum(np.sum(~np.isnan(traces), axis=0), 1)
        mean_tr = np.nanmean(traces, axis=0)
        sem_tr = np.sqrt(np.nanvar(traces, axis=0, ddof=0) / valid_n)
        ax.plot(window_centers_min, mean_tr, color=trace_colors_local[t_idx], lw=1, label=trial, zorder=5)
        ax.fill_between(window_centers_min, mean_tr - sem_tr, mean_tr + sem_tr,
                        color=trace_colors_local[t_idx], alpha=0.2, zorder=2)
    ax.set_xlabel('Time (min)', fontsize=7)
    ax.set_ylabel(metric_name, fontsize=7)
    ax.tick_params(labelsize=6)
    if m_idx == 0:
        ax.legend(fontsize=5, frameon=False, loc='upper right')
    ax.text(-0.15, 1.15, panel_label, transform=ax.transAxes, fontsize=9, fontweight='bold', va='top')

# --- Row 3: Bout duration CDFs (1x5) ---
gs_bot = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[2], wspace=0.35)

for b_idx, (b, bname) in enumerate(zip(bout_behaviors, bout_behavior_names)):
    ax = fig.add_subplot(gs_bot[b_idx])
    for trial, color, ls in [('saline3', '#808080', '-'), ('cocaine5', '#990000', '-')]:
        durs = np.array(bout_durations[trial][b])
        if len(durs) > 0:
            sorted_d = np.sort(durs)
            cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
            ax.step(sorted_d, cdf, color=color, lw=1, label=f'{trial} (n={len(durs)})')
    ax.set_title(bname, fontsize=6, fontweight='bold', color=colors[b], pad=12)
    ax.set_xlabel('Duration (s)', fontsize=5)
    ax.tick_params(labelsize=5)
    ax.set_xlim(0, 30)
    ax.legend(fontsize=4, loc='lower right')
    if b_idx == 0:
        ax.set_ylabel('CDF', fontsize=6)
    panel_label = chr(ord('M') + b_idx)
    ax.text(-0.15, 1.15, panel_label, transform=ax.transAxes, fontsize=9, fontweight='bold', va='top')

save_fig(fig, output_folder, 'SuppFig2_AllBehaviors_Development')
print(f'\nDone — N={n_mice} mice, {len(cohorts)} cohorts')

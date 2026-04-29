#%% Bout rate vs. bout duration decomposition
"""
Decomposes the increase in surface licking into:
  - Bout initiation rate (bouts per minute)
  - Mean bout duration (seconds)
for floor licking and wall licking separately, across sessions.
Output: revision/output/
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
    mm, second, minute, n_behaviors, behaviors, colors, FPS,
    trials_sal_coc as trials, trial_colors,
    cohorts_6 as cohorts,
    load_CMT, flatten_CMT, setup_style, save_fig
)

setup_style()
output_folder = 'revision/output/'
os.makedirs(output_folder, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
CMT = load_CMT('Dec24')
MT, mouse_to_cohort = flatten_CMT(CMT, cohorts)
del CMT

valid_mice = [m for m in MT if all(t in MT[m] for t in trials)]
n_mice = len(valid_mice)
n_trials = len(trials)
print(f'N = {n_mice} mice')

# Target behaviors
FLOOR_LICKING = 2
WALL_LICKING = 3
target_behaviors = [FLOOR_LICKING, WALL_LICKING]
target_names = ['Floor licking', 'Wall licking']
target_colors = [colors[FLOOR_LICKING], colors[WALL_LICKING]]

# ---------------------------------------------------------------------------
# Compute bout rate and mean bout duration per mouse per session
# ---------------------------------------------------------------------------
bout_rate = np.full((n_mice, len(target_behaviors), n_trials), np.nan)    # bouts/min
mean_dur = np.full((n_mice, len(target_behaviors), n_trials), np.nan)     # seconds
total_frac = np.full((n_mice, len(target_behaviors), n_trials), np.nan)   # fraction

for m_idx, mouse in enumerate(valid_mice):
    for t_idx, trial in enumerate(trials):
        preds = MT[mouse][trial]['merged']
        session_dur_min = preds.size / (FPS * 60)

        for b_idx, beh in enumerate(target_behaviors):
            bouts = hf.segment_bouts(preds, beh, 1)
            n_bouts = bouts['number']
            lengths = bouts['length']

            bout_rate[m_idx, b_idx, t_idx] = n_bouts / session_dur_min if session_dur_min > 0 else 0
            if n_bouts > 0 and len(lengths) > 0:
                mean_dur[m_idx, b_idx, t_idx] = np.mean(lengths) / FPS  # convert frames to seconds
            else:
                mean_dur[m_idx, b_idx, t_idx] = 0
            total_frac[m_idx, b_idx, t_idx] = np.count_nonzero(preds == beh) / preds.size

# Print stats
print('\n--- Bout rate and duration: Saline3 vs Cocaine5 ---')
for b_idx, name in enumerate(target_names):
    for metric_name, metric_data in [('Bout rate (bouts/min)', bout_rate), ('Mean duration (s)', mean_dur)]:
        sal3 = metric_data[:, b_idx, 2]   # saline3
        coc5 = metric_data[:, b_idx, 7]   # cocaine5
        t_stat, p_val = stats.ttest_rel(sal3, coc5)
        print(f'  {name} - {metric_name}: S3={np.mean(sal3):.2f} C5={np.mean(coc5):.2f} '
              f't={t_stat:.2f} p={p_val:.4f}')

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(180 * mm, 120 * mm))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

xlabels = ['S1','S2','S3','C1','C2','C3','C4','C5']
x = np.arange(n_trials)

for b_idx, (beh_name, beh_color) in enumerate(zip(target_names, target_colors)):
    # Column 1: Bout rate
    ax_rate = fig.add_subplot(gs[b_idx, 0])
    mean_r = np.nanmean(bout_rate[:, b_idx, :], axis=0)
    sem_r = np.nanstd(bout_rate[:, b_idx, :], axis=0, ddof=0) / np.sqrt(n_mice)
    ax_rate.fill_between(x, mean_r - sem_r, mean_r + sem_r, color=beh_color, alpha=0.2)
    ax_rate.plot(x, mean_r, color=beh_color, lw=2, marker='o', markersize=3)
    for m_idx in range(n_mice):
        ax_rate.plot(x, bout_rate[m_idx, b_idx, :], color='gray', alpha=0.1, lw=0.3)
    ax_rate.set_xticks(x)
    ax_rate.set_xticklabels(xlabels, fontsize=6)
    ax_rate.set_ylabel('Bouts / min', fontsize=8)
    ax_rate.set_title(f'{beh_name}\nBout initiation rate', fontsize=7, fontweight='bold', color=beh_color, pad=12)
    ax_rate.axvspan(-0.5, 2.5, color='gray', alpha=0.04)
    panel = chr(ord('A') + b_idx * 3)
    ax_rate.text(-0.2, 1.15, panel, transform=ax_rate.transAxes, fontsize=10, fontweight='bold', va='top')

    # Column 2: Mean bout duration
    ax_dur = fig.add_subplot(gs[b_idx, 1])
    mean_d = np.nanmean(mean_dur[:, b_idx, :], axis=0)
    sem_d = np.nanstd(mean_dur[:, b_idx, :], axis=0, ddof=0) / np.sqrt(n_mice)
    ax_dur.fill_between(x, mean_d - sem_d, mean_d + sem_d, color=beh_color, alpha=0.2)
    ax_dur.plot(x, mean_d, color=beh_color, lw=2, marker='o', markersize=3)
    for m_idx in range(n_mice):
        ax_dur.plot(x, mean_dur[m_idx, b_idx, :], color='gray', alpha=0.1, lw=0.3)
    ax_dur.set_xticks(x)
    ax_dur.set_xticklabels(xlabels, fontsize=6)
    ax_dur.set_ylabel('Duration (s)', fontsize=8)
    ax_dur.set_title(f'{beh_name}\nMean bout duration', fontsize=7, fontweight='bold', color=beh_color, pad=12)
    ax_dur.axvspan(-0.5, 2.5, color='gray', alpha=0.04)
    panel = chr(ord('A') + b_idx * 3 + 1)
    ax_dur.text(-0.2, 1.15, panel, transform=ax_dur.transAxes, fontsize=10, fontweight='bold', va='top')

    # Column 3: Scatter — bout rate vs duration change (C5 - S3)
    ax_scat = fig.add_subplot(gs[b_idx, 2])
    delta_rate = bout_rate[:, b_idx, 7] - bout_rate[:, b_idx, 2]  # C5 - S3
    delta_dur = mean_dur[:, b_idx, 7] - mean_dur[:, b_idx, 2]

    ax_scat.scatter(delta_rate, delta_dur, color=beh_color, s=20, alpha=0.7, edgecolors='white', linewidths=0.3)
    ax_scat.axhline(0, color='gray', ls=':', lw=0.5)
    ax_scat.axvline(0, color='gray', ls=':', lw=0.5)

    # Correlation
    valid = ~(np.isnan(delta_rate) | np.isnan(delta_dur))
    if valid.sum() > 3:
        r, p = stats.pearsonr(delta_rate[valid], delta_dur[valid])
        ax_scat.text(0.05, 0.95, f'r={r:.2f}, p={p:.3f}', transform=ax_scat.transAxes,
                     fontsize=6, va='top')

    ax_scat.set_xlabel(r'$\Delta$ Bout rate (C5-S3)', fontsize=7)
    ax_scat.set_ylabel(r'$\Delta$ Duration (C5-S3)', fontsize=7)
    ax_scat.set_title(f'{beh_name}\nRate vs Duration change', fontsize=7, fontweight='bold', color=beh_color, pad=12)
    panel = chr(ord('A') + b_idx * 3 + 2)
    ax_scat.text(-0.2, 1.15, panel, transform=ax_scat.transAxes, fontsize=10, fontweight='bold', va='top')

fig.suptitle(f'Bout Decomposition: Rate vs. Duration (N={n_mice})', fontsize=9, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.96])
save_fig(fig, output_folder, 'Impact_BoutDecomposition')
print(f'\nDone — N={n_mice} mice')

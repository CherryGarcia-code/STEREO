#%% SuppFig: Stacked Behavioral Distributions
"""
Addresses R4-Fig2EF (stacked graphs instead of overlapping curves).

Panels:
  A-H: Stacked area plots showing full behavioral distributions per session (saline1-3 + cocaine1-5)
  I-K: Within-session stackplots for saline3, cocaine1, cocaine5

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

from revision_utils import (
    mm, second, minute, n_behaviors, behaviors, colors,
    trials_sal_coc as trials, cohorts_6 as cohorts,
    load_CMT, flatten_CMT, setup_style, save_fig
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

valid_mice = [m for m in MT if all(t in MT[m] for t in trials)]
n_mice = len(valid_mice)
print(f'N = {n_mice} mice with complete data')

# ---------------------------------------------------------------------------
# Panel A-H: Across-session stacked bar chart (mean fractions)
# ---------------------------------------------------------------------------
fractions = np.zeros((n_mice, n_behaviors, len(trials)))
for m_idx, mouse in enumerate(valid_mice):
    for t_idx, trial in enumerate(trials):
        preds = MT[mouse][trial]['merged']
        total = preds.size
        for b in range(n_behaviors):
            fractions[m_idx, b, t_idx] = np.count_nonzero(preds == b) / total

mean_frac = np.nanmean(fractions, axis=0)  # (n_behaviors, n_trials)

# Reorder behaviors for stacking: meaningful order bottom to top
# Stationary, Locomotion, Rearing, Body licking, Grooming, Floor licking, Wall licking, Undefined, Jump
stack_order = [8, 7, 6, 5, 4, 3, 2, 1, 0]

fig1 = plt.figure(figsize=(100 * mm, 70 * mm))
ax = fig1.add_subplot(111)
x = np.arange(len(trials))

bottom = np.zeros(len(trials))
for b_idx in stack_order:
    ax.bar(x, mean_frac[b_idx, :], bottom=bottom, color=colors[b_idx],
           width=0.7, label=behaviors[b_idx], edgecolor='white', linewidth=0.3)
    bottom += mean_frac[b_idx, :]

ax.set_xticks(x)
ax.set_xticklabels(['S1','S2','S3','C1','C2','C3','C4','C5'], fontsize=7)
ax.set_ylabel('Fraction of time', fontsize=8)
ax.set_xlabel('Session', fontsize=8)
ax.set_ylim(0, 1)
ax.legend(fontsize=5, bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
ax.set_title(f'Behavioral distribution across sessions (N={n_mice})', fontsize=8, fontweight='bold', pad=12)

save_fig(fig1, output_folder, 'SuppFig6a_Stacked_AcrossSessions')

# ---------------------------------------------------------------------------
# Panel I-K: Within-session stacked area plots
# ---------------------------------------------------------------------------
window_frames = 1 * minute
step_frames = 30 * second
max_duration = 27 * minute
n_windows = int((max_duration - window_frames) / step_frames) + 1
window_centers_min = np.array([(w * step_frames + window_frames / 2) / minute for w in range(n_windows)])

target_sessions = ['saline3', 'cocaine1', 'cocaine5']
target_titles = ['Saline 3', 'Cocaine 1', 'Cocaine 5']

fig2, axes = plt.subplots(1, 3, figsize=(180 * mm, 55 * mm), sharey=True)

for s_idx, (trial, title) in enumerate(zip(target_sessions, target_titles)):
    ax = axes[s_idx]

    # Compute per-window fractions
    windowed_frac = np.zeros((n_mice, n_behaviors, n_windows))
    for m_idx, mouse in enumerate(valid_mice):
        if trial not in MT[mouse]:
            continue
        preds = MT[mouse][trial]['merged']
        for w in range(n_windows):
            start = w * step_frames
            end = start + window_frames
            if end <= len(preds):
                window = preds[start:end]
                for b in range(n_behaviors):
                    windowed_frac[m_idx, b, w] = np.count_nonzero(window == b) / window.size

    mean_wf = np.nanmean(windowed_frac, axis=0)  # (n_behaviors, n_windows)

    # Stacked area plot
    stacked_data = np.array([mean_wf[b_idx, :] for b_idx in stack_order])
    stack_labels = [behaviors[b_idx] for b_idx in stack_order]
    stack_colors = [colors[b_idx] for b_idx in stack_order]

    ax.stackplot(window_centers_min, stacked_data, labels=stack_labels,
                 colors=stack_colors, alpha=0.85)
    ax.set_xlabel('Time (min)', fontsize=8)
    ax.set_title(title, fontsize=8, fontweight='bold', pad=12)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=6)
    panel_lbl = chr(ord('A') + s_idx)
    ax.text(-0.1, 1.15, panel_lbl, transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')

axes[0].set_ylabel('Fraction of time', fontsize=8)
# Legend below
handles, labels = axes[0].get_legend_handles_labels()
fig2.legend(handles, labels, loc='lower center', ncol=5, fontsize=5, frameon=False,
            bbox_to_anchor=(0.5, -0.02))

fig2.tight_layout(rect=[0, 0.06, 1, 1], pad=0.5)
save_fig(fig2, output_folder, 'SuppFig6b_Stacked_WithinSession')
print(f'\nDone — N={n_mice} mice')

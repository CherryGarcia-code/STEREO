#%% Individual mouse trajectories for all 9 behaviors
"""
Spaghetti plots showing individual mouse trajectories overlaid with group mean,
for all 9 behaviors across saline1-3 + cocaine1-5.
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

from revision_utils import (
    mm, n_behaviors, behaviors, colors, short_labels,
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

# Compute fractions
fractions = np.full((n_mice, n_behaviors, n_trials), np.nan)
for m_idx, mouse in enumerate(valid_mice):
    for t_idx, trial in enumerate(trials):
        preds = MT[mouse][trial]['merged']
        total = preds.size
        for b in range(n_behaviors):
            fractions[m_idx, b, t_idx] = np.count_nonzero(preds == b) / total

x = np.arange(n_trials)
xlabels = ['S1','S2','S3','C1','C2','C3','C4','C5']

# ---------------------------------------------------------------------------
# Figure: 3x3 grid, one panel per behavior
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(180 * mm, 150 * mm))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

for b in range(n_behaviors):
    row, col = divmod(b, 3)
    ax = fig.add_subplot(gs[row, col])

    # Individual mice
    for m_idx in range(n_mice):
        ax.plot(x, fractions[m_idx, b, :], color='gray', alpha=0.15, lw=0.4)

    # Group mean + SEM
    mean_b = np.nanmean(fractions[:, b, :], axis=0)
    sem_b = np.nanstd(fractions[:, b, :], axis=0, ddof=0) / np.sqrt(n_mice)
    ax.fill_between(x, mean_b - sem_b, mean_b + sem_b, color=colors[b], alpha=0.25)
    ax.plot(x, mean_b, color=colors[b], lw=2, marker='o', markersize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=5.5)
    ax.set_title(behaviors[b], fontsize=7, fontweight='bold', color=colors[b], pad=12)
    ax.tick_params(labelsize=6)

    if col == 0:
        ax.set_ylabel('Fraction', fontsize=7)
    if row == 2:
        ax.set_xlabel('Session', fontsize=7)

    # Panel letter
    panel = chr(ord('A') + b)
    ax.text(-0.15, 1.15, panel, transform=ax.transAxes, fontsize=9, fontweight='bold', va='top')

    # Saline background
    ax.axvspan(-0.5, 2.5, color='gray', alpha=0.04)

fig.suptitle(f'Individual Mouse Trajectories — All Behaviors (N={n_mice})',
             fontsize=9, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.96])
save_fig(fig, output_folder, 'Impact_IndividualTrajectories')
print(f'Done — N={n_mice} mice')

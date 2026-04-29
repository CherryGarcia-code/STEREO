"""
Integrative Analysis 1: Behavioral State-Space (PCA)

Uses PCA on per-session per-mouse behavior occupancy vectors (9 behaviors)
to show how mice traverse behavioral state-space across cocaine sensitization.

Outputs (both with 4-cohort and 6-cohort versions):
  - Int1_BehavioralStateSpace_4cohorts.png/pdf
  - Int1_BehavioralStateSpace_6cohorts.png/pdf
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'revision'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from sklearn.decomposition import PCA
from scipy import stats

from revision_utils import (
    mm, n_behaviors, behaviors, colors, trial_colors,
    cohorts_4, cohorts_6, cohort_labels,
    load_CMT, flatten_CMT, setup_style, save_fig
)

setup_style()
output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(output_folder, exist_ok=True)

trials = ['saline1', 'saline2', 'saline3',
          'cocaine1', 'cocaine2', 'cocaine3', 'cocaine4', 'cocaine5']
session_labels = ['S1', 'S2', 'S3', 'C1', 'C2', 'C3', 'C4', 'C5']
session_colors = ['#808080', '#808080', '#808080',
                  '#FFA500', '#FF8C00', '#FF6347', '#E60000', '#990000']


def compute_occupancy(preds, n_beh):
    """Fraction of frames in each behavior. Sums to 1."""
    counts = np.bincount(preds.astype(int), minlength=n_beh).astype(float)
    total = counts.sum()
    return counts / total if total > 0 else np.full(n_beh, np.nan)


def run_analysis(cohorts, suffix):
    print(f'\n=== PCA State-Space ({suffix}) ===')
    CMT = load_CMT('Dec24')
    MT, mouse_to_cohort = flatten_CMT(CMT, cohorts)
    del CMT

    # Only mice with all 8 sessions
    valid_mice = [m for m in MT if all(t in MT[m] for t in trials)]
    n_mice = len(valid_mice)
    print(f'N = {n_mice} mice with complete data')

    # Build occupancy matrix: rows = (mouse, session), cols = 9 behaviors
    occ_matrix = []   # shape: (n_mice * n_sessions, n_behaviors)
    mouse_ids = []
    session_ids = []
    cohort_ids = []

    for m in valid_mice:
        for s_idx, trial in enumerate(trials):
            preds = MT[m][trial]['merged']
            occ = compute_occupancy(preds, n_behaviors)
            occ_matrix.append(occ)
            mouse_ids.append(m)
            session_ids.append(s_idx)
            cohort_ids.append(mouse_to_cohort[m])

    occ_matrix = np.array(occ_matrix)  # (N*8, 9)
    session_ids = np.array(session_ids)
    n_obs = len(occ_matrix)

    # Fit PCA on the full matrix
    pca = PCA(n_components=3)
    coords = pca.fit_transform(occ_matrix)  # (N*8, 3)
    var_explained = pca.explained_variance_ratio_ * 100

    # Per-session mean trajectory (across mice)
    mean_traj = np.array([
        coords[session_ids == s_idx].mean(axis=0)
        for s_idx in range(len(trials))
    ])  # (8, 3)

    # Per-session SEM
    sem_traj = np.array([
        coords[session_ids == s_idx].std(axis=0) / np.sqrt((session_ids == s_idx).sum())
        for s_idx in range(len(trials))
    ])

    print('Variance explained PC1-3:', [f'{v:.1f}%' for v in var_explained])
    print('PC1 loadings (top behaviors):')
    for i in np.argsort(np.abs(pca.components_[0]))[::-1][:4]:
        print(f'  {behaviors[i]}: {pca.components_[0][i]:.3f}')

    # ---------------------------------------------------------------------------
    # Figure
    # ---------------------------------------------------------------------------
    fig = plt.figure(figsize=(200 * mm, 160 * mm))
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           left=0.1, right=0.95, top=0.88, bottom=0.12,
                           wspace=0.45, hspace=0.5)

    # --- Panel A: PC1 vs PC2 scatter (all points, colored by session) ---
    ax_a = fig.add_subplot(gs[0, 0])
    for s_idx in range(len(trials)):
        mask = session_ids == s_idx
        ax_a.scatter(coords[mask, 0], coords[mask, 1],
                     c=session_colors[s_idx], s=10, alpha=0.45,
                     linewidths=0, zorder=2)
    # Mean trajectory with arrows
    ax_a.plot(mean_traj[:3, 0], mean_traj[:3, 1],
              '-', color='#808080', lw=2, zorder=4, alpha=0.8)
    ax_a.plot(mean_traj[2:, 0], mean_traj[2:, 1],
              '-', color='#E60000', lw=2, zorder=4, alpha=0.8)
    # Filled markers for mean
    for s_idx in range(len(trials)):
        ax_a.scatter(mean_traj[s_idx, 0], mean_traj[s_idx, 1],
                     c=session_colors[s_idx], s=60, zorder=5,
                     edgecolors='white', linewidths=0.8)
        ax_a.text(mean_traj[s_idx, 0], mean_traj[s_idx, 1] + 0.005,
                  session_labels[s_idx], fontsize=5, ha='center',
                  color=session_colors[s_idx], zorder=6)
    ax_a.set_xlabel(f'PC1 ({var_explained[0]:.1f}%)', fontsize=9)
    ax_a.set_ylabel(f'PC2 ({var_explained[1]:.1f}%)', fontsize=9)
    ax_a.set_title('Behavioral state-space\n(PC1 vs PC2)', fontsize=8, fontweight='bold', pad=14)
    ax_a.text(-0.18, 1.15, 'A', transform=ax_a.transAxes,
              fontsize=12, fontweight='bold', va='top')

    # --- Panel B: PC1 vs PC3 ---
    ax_b = fig.add_subplot(gs[0, 1])
    for s_idx in range(len(trials)):
        mask = session_ids == s_idx
        ax_b.scatter(coords[mask, 0], coords[mask, 2],
                     c=session_colors[s_idx], s=10, alpha=0.45,
                     linewidths=0, zorder=2)
    ax_b.plot(mean_traj[:3, 0], mean_traj[:3, 2], '-', color='#808080', lw=2, zorder=4, alpha=0.8)
    ax_b.plot(mean_traj[2:, 0], mean_traj[2:, 2], '-', color='#E60000', lw=2, zorder=4, alpha=0.8)
    for s_idx in range(len(trials)):
        ax_b.scatter(mean_traj[s_idx, 0], mean_traj[s_idx, 2],
                     c=session_colors[s_idx], s=60, zorder=5,
                     edgecolors='white', linewidths=0.8)
    ax_b.set_xlabel(f'PC1 ({var_explained[0]:.1f}%)', fontsize=9)
    ax_b.set_ylabel(f'PC3 ({var_explained[2]:.1f}%)', fontsize=9)
    ax_b.set_title('Behavioral state-space\n(PC1 vs PC3)', fontsize=8, fontweight='bold', pad=14)
    ax_b.text(-0.18, 1.15, 'B', transform=ax_b.transAxes,
              fontsize=12, fontweight='bold', va='top')

    # --- Panel C: PC1 across sessions (violin + mean) ---
    ax_c = fig.add_subplot(gs[1, 0])
    pc1_per_session = [coords[session_ids == s_idx, 0] for s_idx in range(len(trials))]

    parts = ax_c.violinplot(pc1_per_session, positions=range(len(trials)),
                            showmedians=False, showextrema=False)
    for i, body in enumerate(parts['bodies']):
        body.set_facecolor(session_colors[i])
        body.set_alpha(0.5)
    # Overlay means +/- SEM
    for s_idx in range(len(trials)):
        m = mean_traj[s_idx, 0]
        e = sem_traj[s_idx, 0]
        ax_c.errorbar(s_idx, m, yerr=e, fmt='o', color=session_colors[s_idx],
                      markersize=5, capsize=3, lw=1.2, zorder=5)
    ax_c.set_xticks(range(len(trials)))
    ax_c.set_xticklabels(session_labels, fontsize=7)
    ax_c.set_ylabel(f'PC1 score ({var_explained[0]:.1f}%)', fontsize=9)
    ax_c.set_title('PC1 across sessions', fontsize=8, fontweight='bold', pad=14)
    ax_c.axvline(2.5, color='k', ls='--', lw=0.7, alpha=0.5)
    ax_c.text(0.5, 0.97, 'Saline', transform=ax_c.transAxes, fontsize=6,
              va='top', ha='center', color='#808080')
    ax_c.text(0.75, 0.97, 'Cocaine', transform=ax_c.transAxes, fontsize=6,
              va='top', ha='center', color='#E60000')
    ax_c.text(-0.18, 1.15, 'C', transform=ax_c.transAxes,
              fontsize=12, fontweight='bold', va='top')

    # --- Panel D: PC loadings (behavior contributions) ---
    ax_d = fig.add_subplot(gs[1, 1])
    x_beh = np.arange(n_behaviors)
    bar_w = 0.28
    ax_d.bar(x_beh - bar_w, pca.components_[0], bar_w,
             color=[colors[i] for i in range(n_behaviors)],
             alpha=0.9, label='PC1', edgecolor='none')
    ax_d.bar(x_beh, pca.components_[1], bar_w,
             color=[colors[i] for i in range(n_behaviors)],
             alpha=0.5, label='PC2', edgecolor='none')
    ax_d.bar(x_beh + bar_w, pca.components_[2], bar_w,
             color=[colors[i] for i in range(n_behaviors)],
             alpha=0.25, label='PC3', edgecolor='none')
    ax_d.axhline(0, color='k', lw=0.6)
    ax_d.set_xticks(x_beh)
    beh_short = ['Jmp', 'Udf', 'FL', 'WL', 'Grm', 'BL', 'Rer', 'Loc', 'Stn']
    ax_d.set_xticklabels(beh_short, fontsize=6.5, rotation=40, ha='right')
    ax_d.set_ylabel('PC loading', fontsize=9)
    ax_d.set_title('PCA loadings per behavior', fontsize=8, fontweight='bold', pad=14)
    ax_d.legend(fontsize=6, loc='upper right', frameon=False)
    ax_d.text(-0.18, 1.15, 'D', transform=ax_d.transAxes,
              fontsize=12, fontweight='bold', va='top')

    # Session color legend at top
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=session_colors[i],
               markersize=6, label=session_labels[i])
        for i in range(len(trials))
    ]
    fig.legend(handles=legend_elements, ncol=8, loc='upper center',
               bbox_to_anchor=(0.5, 0.96), fontsize=6.5, frameon=False,
               title='Session', title_fontsize=7)

    save_fig(fig, output_folder, f'Int1_BehavioralStateSpace_{suffix}')


run_analysis(cohorts_4, '4cohorts')
run_analysis(cohorts_6, '6cohorts')
print('\nDone — state-space figures saved.')

"""
Integrative Analysis 3: Entropy–Licking Inverse Relationship

As cocaine sensitization develops, surface licking fraction rises while
behavioral switch entropy falls. This script quantifies that inverse
relationship at the population level (across sessions) and at the individual
mouse level (across animals at peak sensitization).

Outputs:
  - Int3_EntropyLickingRelationship_4cohorts.png/pdf
  - Int3_EntropyLickingRelationship_6cohorts.png/pdf
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
    mm, n_behaviors, behaviors, colors, FPS, second, minute,
    cohorts_4, cohorts_6, cohort_labels,
    load_CMT, flatten_CMT, setup_style, save_fig,
    calc_switch_entropy, transition_rate
)

setup_style()
output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(output_folder, exist_ok=True)

FL, WL = 2, 3   # behavior indices for surface licking

trials = ['saline1', 'saline2', 'saline3',
          'cocaine1', 'cocaine2', 'cocaine3', 'cocaine4', 'cocaine5']
session_labels = ['S1', 'S2', 'S3', 'C1', 'C2', 'C3', 'C4', 'C5']
session_colors = ['#808080', '#808080', '#808080',
                  '#FFA500', '#FF8C00', '#FF6347', '#E60000', '#990000']


def compute_session_metrics(preds):
    """Return (surface_licking_frac, switch_entropy) for a prediction array."""
    n = len(preds)
    occ = np.bincount(preds.astype(int), minlength=n_behaviors) / n
    sl = occ[FL] + occ[WL]

    sw = calc_switch_entropy(preds, n, n, n_behaviors)
    ent = sw[0] if len(sw) > 0 and not np.isnan(sw[0]) else np.nan

    return sl, ent


def run_analysis(cohorts, suffix):
    print(f'\n=== Entropy–Licking Relationship ({suffix}) ===')
    CMT = load_CMT('Dec24')
    MT, mouse_to_cohort = flatten_CMT(CMT, cohorts)
    del CMT

    # Only mice with all 8 sessions
    valid_mice = [m for m in MT if all(t in MT[m] for t in trials)]
    n_mice = len(valid_mice)
    print(f'N = {n_mice} mice with complete data')

    # Build per-mouse per-session arrays
    sl_matrix  = np.full((n_mice, len(trials)), np.nan)
    ent_matrix = np.full((n_mice, len(trials)), np.nan)

    for mi, m in enumerate(valid_mice):
        for si, t in enumerate(trials):
            sl, ent = compute_session_metrics(MT[m][t]['merged'])
            sl_matrix[mi, si]  = sl
            ent_matrix[mi, si] = ent

    # Population means and SEMs per session
    sl_mean  = np.nanmean(sl_matrix,  axis=0)
    sl_sem   = np.nanstd(sl_matrix,   axis=0) / np.sqrt(np.sum(~np.isnan(sl_matrix),  axis=0))
    ent_mean = np.nanmean(ent_matrix, axis=0)
    ent_sem  = np.nanstd(ent_matrix,  axis=0) / np.sqrt(np.sum(~np.isnan(ent_matrix), axis=0))

    # Cross-session correlation (N=8 sessions, population means)
    r_sess, p_sess = stats.pearsonr(sl_mean, ent_mean)
    print(f'Cross-session correlation (means): r={r_sess:.3f}, p={p_sess:.4f}')

    # Cross-mouse correlation at cocaine5 (index 7)
    valid_c5 = ~np.isnan(sl_matrix[:, 7]) & ~np.isnan(ent_matrix[:, 7])
    sl_c5  = sl_matrix[valid_c5, 7]
    ent_c5 = ent_matrix[valid_c5, 7]
    r_mouse, p_mouse = stats.pearsonr(sl_c5, ent_c5)
    print(f'Cross-mouse correlation at cocaine5: r={r_mouse:.3f}, p={p_mouse:.4f}, N={valid_c5.sum()}')

    # ---------------------------------------------------------------------------
    # Figure — 3 panels
    # ---------------------------------------------------------------------------
    fig = plt.figure(figsize=(200 * mm, 75 * mm))
    gs = gridspec.GridSpec(1, 3, figure=fig,
                           left=0.1, right=0.96, top=0.84, bottom=0.22,
                           wspace=0.52)

    # --- Panel A: Time courses of surface licking and entropy (dual y-axis) ---
    ax_a = fig.add_subplot(gs[0])
    x = np.arange(len(trials))

    ax_a.errorbar(x, sl_mean, yerr=sl_sem, fmt='o-', color='#d73027',
                  markersize=4, lw=1.5, capsize=2, label='Surface licking', zorder=4)
    ax_a.set_xlabel('Session', fontsize=8)
    ax_a.set_ylabel('Surface licking fraction', fontsize=8, color='#d73027')
    ax_a.tick_params(axis='y', labelcolor='#d73027', labelsize=7)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(session_labels, fontsize=7)
    ax_a.axvline(2.5, color='k', ls='--', lw=0.7, alpha=0.5)

    ax_a2 = ax_a.twinx()
    ax_a2.errorbar(x, ent_mean, yerr=ent_sem, fmt='s--', color='#3399cc',
                   markersize=4, lw=1.5, capsize=2, label='Switch entropy', zorder=3)
    ax_a2.set_ylabel('Switch entropy (bits)', fontsize=8, color='#3399cc')
    ax_a2.tick_params(axis='y', labelcolor='#3399cc', labelsize=7)
    ax_a2.spines['right'].set_visible(True)

    ax_a.set_title('Population time courses', fontsize=8, fontweight='bold', pad=14)
    ax_a.text(-0.22, 1.15, 'A', transform=ax_a.transAxes,
              fontsize=12, fontweight='bold', va='top')
    # Combined legend
    handles = [
        plt.Line2D([0], [0], color='#d73027', marker='o', lw=1.5, ms=4, label='Surface licking'),
        plt.Line2D([0], [0], color='#3399cc', marker='s', ls='--', lw=1.5, ms=4, label='Switch entropy'),
    ]
    ax_a.legend(handles=handles, fontsize=6, loc='upper left',
                bbox_to_anchor=(0.0, 0.98), frameon=False)

    # --- Panel B: Session-level scatter (means) ---
    ax_b = fig.add_subplot(gs[1])
    for si in range(len(trials)):
        ax_b.scatter(sl_mean[si], ent_mean[si], color=session_colors[si],
                     s=50, zorder=4, edgecolors='white', linewidths=0.6)
        ax_b.text(sl_mean[si] + 0.003, ent_mean[si], session_labels[si],
                  fontsize=5.5, va='center', color=session_colors[si])

    # Regression line (across 8 session means)
    slope, intercept, _, _, _ = stats.linregress(sl_mean, ent_mean)
    x_range = np.linspace(sl_mean.min(), sl_mean.max(), 100)
    ax_b.plot(x_range, slope * x_range + intercept, 'k--', lw=0.9, alpha=0.5)

    sig_str = '***' if p_sess < 0.001 else '**' if p_sess < 0.01 else '*' if p_sess < 0.05 else 'n.s.'
    ax_b.set_xlabel('Surface licking fraction', fontsize=8)
    ax_b.set_ylabel('Switch entropy (bits)', fontsize=8)
    ax_b.set_title(f'Session-level inverse relationship\nr={r_sess:.2f}, {sig_str} (N=8 sessions)',
                   fontsize=8, fontweight='bold', pad=14)
    ax_b.text(-0.3, 1.15, 'B', transform=ax_b.transAxes,
              fontsize=12, fontweight='bold', va='top')

    # --- Panel C: Mouse-level scatter at cocaine5 ---
    ax_c = fig.add_subplot(gs[2])

    cohort_color_map = {
        'drd1_hm4di': '#d73027', 'drd1_hm3dq': '#f4a582',
        'controls': '#808080', 'a2a_hm4di': '#721515',
        'a2a_hm3dq': '#d9a0a0', 'a2a_opto': '#3399cc'
    }
    idx = 0
    for mi, m in enumerate(valid_mice):
        if valid_c5[mi]:
            c = cohort_color_map.get(mouse_to_cohort[m], '#808080')
            ax_c.scatter(sl_c5[idx], ent_c5[idx], color=c, s=30, alpha=0.8,
                         edgecolors='white', linewidths=0.5, zorder=3)
            idx += 1

    slope2, intercept2, _, _, _ = stats.linregress(sl_c5, ent_c5)
    x2 = np.linspace(sl_c5.min(), sl_c5.max(), 100)
    ax_c.plot(x2, slope2 * x2 + intercept2, 'k--', lw=0.9, alpha=0.5)

    sig_str2 = '***' if p_mouse < 0.001 else '**' if p_mouse < 0.01 else '*' if p_mouse < 0.05 else 'n.s.'
    ax_c.set_xlabel('Cocaine 5 surface licking', fontsize=8)
    ax_c.set_ylabel('Cocaine 5 switch entropy (bits)', fontsize=8)
    ax_c.set_title(f'Individual mice at peak sensitization\nr={r_mouse:.2f}, {sig_str2} (N={valid_c5.sum()})',
                   fontsize=8, fontweight='bold', pad=14)
    ax_c.text(-0.3, 1.15, 'C', transform=ax_c.transAxes,
              fontsize=12, fontweight='bold', va='top')

    # Cohort legend
    from matplotlib.lines import Line2D
    from revision_utils import cohort_labels as cl
    legend_cohorts = [c for c in cohorts if c in cohort_color_map]
    legend_elems = [Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=cohort_color_map[c], markersize=7,
                           label=cl[c])
                    for c in legend_cohorts]
    fig.legend(handles=legend_elems, ncol=len(legend_cohorts),
               loc='lower center', bbox_to_anchor=(0.5, 0.01),
               fontsize=6.5, frameon=False)

    save_fig(fig, output_folder, f'Int3_EntropyLickingRelationship_{suffix}')


run_analysis(cohorts_4, '4cohorts')
run_analysis(cohorts_6, '6cohorts')
print('\nDone — entropy–licking relationship figures saved.')

#%% SuppFig: Context Specificity — SalineOnly vs Saline Baseline
"""
Addresses R3-3 (is licking context-specific?).

Compares salineOnly (post-cocaine, in cocaine-paired chamber) vs saline1-3 baseline.
Panels:
  A. Per-behavior fraction comparison (saline avg vs salineOnly)
  B. Surface licking paired comparison (individual mice)
  C. Behavioral entropy comparison

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

from revision_utils import (
    mm, second, minute, n_behaviors, behaviors, colors,
    cohorts_6 as cohorts, cohort_labels,
    load_CMT, flatten_CMT, setup_style, save_fig,
    calc_behavioral_entropy, PATHO_LICKING, grouping_lut
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

saline_trials = ['saline1', 'saline2', 'saline3']

# Find mice with salineOnly + at least 2 saline baselines
valid_mice = []
for m in MT:
    if 'salineOnly' in MT[m]:
        n_sal = sum(1 for t in saline_trials if t in MT[m])
        if n_sal >= 2:
            valid_mice.append(m)

n_mice = len(valid_mice)
print(f'N = {n_mice} mice with salineOnly + saline baseline')

# ---------------------------------------------------------------------------
# Compute fractions
# ---------------------------------------------------------------------------
saline_frac = np.full((n_mice, n_behaviors), np.nan)    # avg across saline1-3
salineOnly_frac = np.full((n_mice, n_behaviors), np.nan)

saline_entropy = np.full(n_mice, np.nan)
salineOnly_entropy = np.full(n_mice, np.nan)

for m_idx, mouse in enumerate(valid_mice):
    # Saline baseline: average across available saline sessions
    sal_fracs = []
    sal_preds_all = []
    for trial in saline_trials:
        if trial in MT[mouse]:
            preds = MT[mouse][trial]['merged']
            frac = np.array([np.count_nonzero(preds == b) / preds.size for b in range(n_behaviors)])
            sal_fracs.append(frac)
            sal_preds_all.append(preds)
    saline_frac[m_idx, :] = np.mean(sal_fracs, axis=0)

    # Compute entropy on concatenated saline predictions
    sal_concat = np.concatenate(sal_preds_all)
    beh_ent = calc_behavioral_entropy(sal_concat, sal_concat.size, sal_concat.size, n_behaviors)
    saline_entropy[m_idx] = beh_ent[0] if len(beh_ent) > 0 else np.nan

    # SalineOnly
    preds_so = MT[mouse]['salineOnly']['merged']
    for b in range(n_behaviors):
        salineOnly_frac[m_idx, b] = np.count_nonzero(preds_so == b) / preds_so.size
    beh_ent_so = calc_behavioral_entropy(preds_so, preds_so.size, preds_so.size, n_behaviors)
    salineOnly_entropy[m_idx] = beh_ent_so[0] if len(beh_ent_so) > 0 else np.nan

# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
print('\n--- Per-behavior: Saline baseline vs SalineOnly ---')
for b in range(n_behaviors):
    t_stat, p_val = stats.ttest_rel(saline_frac[:, b], salineOnly_frac[:, b])
    print(f'  {behaviors[b]:<16} sal={np.mean(saline_frac[:, b]):.3f} salOnly={np.mean(salineOnly_frac[:, b]):.3f} '
          f't={t_stat:.2f} p={p_val:.4f}')

# Surface licking grouped
sal_surface = np.sum(saline_frac[:, [2, 3]], axis=1)
so_surface = np.sum(salineOnly_frac[:, [2, 3]], axis=1)
t_surf, p_surf = stats.ttest_rel(sal_surface, so_surface)
print(f'\n  Surface licking (grouped): sal={np.mean(sal_surface):.3f} salOnly={np.mean(so_surface):.3f} '
      f't={t_surf:.2f} p={p_surf:.4f}')

t_ent, p_ent = stats.ttest_rel(saline_entropy, salineOnly_entropy)
print(f'  Behavioral entropy: sal={np.mean(saline_entropy):.3f} salOnly={np.mean(salineOnly_entropy):.3f} '
      f't={t_ent:.2f} p={p_ent:.4f}')

# ===========================================================================
# Compute fold change (salineOnly / saline baseline) — per behavior
# ===========================================================================
fold_change = np.full((n_mice, n_behaviors), np.nan)
for m_idx in range(n_mice):
    for b in range(n_behaviors):
        if saline_frac[m_idx, b] > 0:
            fold_change[m_idx, b] = salineOnly_frac[m_idx, b] / saline_frac[m_idx, b]

mean_fc = np.nanmean(fold_change, axis=0)
sem_fc = np.nanstd(fold_change, axis=0, ddof=0) / np.sqrt(n_mice)

# ===========================================================================
# BUILD THE FIGURE
# ===========================================================================
fig = plt.figure(figsize=(180 * mm, 100 * mm))
gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.5,
                       height_ratios=[1.4, 1])

# --- Panel A: Fold-change horizontal bars (matching attached figure style) ---
ax_a = fig.add_subplot(gs[0, :])
beh_order = [8, 7, 6, 5, 4, 3, 2, 1, 0]  # Stationary at top, Jump at bottom
beh_labels_display = [behaviors[i] for i in beh_order]
beh_colors_display = [colors[i] for i in beh_order]

y_pos = np.arange(len(beh_order))
fc_ordered = [mean_fc[i] for i in beh_order]
sem_ordered = [sem_fc[i] for i in beh_order]

ax_a.barh(y_pos, fc_ordered, color=beh_colors_display, edgecolor='none', height=0.7)
ax_a.errorbar(fc_ordered, y_pos, xerr=sem_ordered, fmt='none', color='black',
              capsize=2, capthick=0.5, lw=0.5)

# Significance markers
for idx, b in enumerate(beh_order):
    _, p = stats.ttest_rel(saline_frac[:, b], salineOnly_frac[:, b])
    if p < 0.05:
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*'
        x_pos = fc_ordered[idx] + sem_ordered[idx] + 0.05
        ax_a.text(x_pos, idx, sig, va='center', fontsize=7, fontweight='bold')

ax_a.axvline(x=1.0, color='black', ls='--', lw=0.8, alpha=0.5)
ax_a.set_yticks(y_pos)
ax_a.set_yticklabels(beh_labels_display, fontsize=8)
ax_a.set_xlabel('% Time spent (fold change)', fontsize=10)
ax_a.text(-0.08, 1.15, 'A', transform=ax_a.transAxes, fontsize=12, fontweight='bold', va='top')

# --- Panel B: Surface licking paired comparison ---
ax_b = fig.add_subplot(gs[1, 0])
for m in range(n_mice):
    ax_b.plot([0, 1], [sal_surface[m], so_surface[m]], color='gray', alpha=0.3, lw=0.5)
ax_b.errorbar([0], [np.mean(sal_surface)], yerr=[np.std(sal_surface, ddof=0)/np.sqrt(n_mice)],
              color='#808080', marker='o', markersize=6, capsize=4, lw=2)
ax_b.errorbar([1], [np.mean(so_surface)], yerr=[np.std(so_surface, ddof=0)/np.sqrt(n_mice)],
              color='#FFA500', marker='o', markersize=6, capsize=4, lw=2)

sig_surf = '***' if p_surf < 0.001 else '**' if p_surf < 0.01 else '*' if p_surf < 0.05 else 'n.s.'
ymax = max(np.max(sal_surface), np.max(so_surface)) * 1.05
ax_b.plot([0, 1], [ymax, ymax], 'k-', lw=0.8)
ax_b.text(0.5, ymax * 1.02, sig_surf, ha='center', fontsize=7)

ax_b.set_xticks([0, 1])
ax_b.set_xticklabels(['Saline\nbaseline', 'SalineOnly\n(post-cocaine)'], fontsize=7)
ax_b.set_ylabel('Surface licking fraction', fontsize=8)
ax_b.set_title('Surface licking', fontsize=8, fontweight='bold', pad=12)
ax_b.text(-0.2, 1.15, 'B', transform=ax_b.transAxes, fontsize=10, fontweight='bold', va='top')

# --- Panel C: Behavioral entropy comparison ---
ax_c = fig.add_subplot(gs[1, 1])
for m in range(n_mice):
    ax_c.plot([0, 1], [saline_entropy[m], salineOnly_entropy[m]], color='gray', alpha=0.3, lw=0.5)
ax_c.errorbar([0], [np.mean(saline_entropy)], yerr=[np.std(saline_entropy, ddof=0)/np.sqrt(n_mice)],
              color='#808080', marker='o', markersize=6, capsize=4, lw=2)
ax_c.errorbar([1], [np.mean(salineOnly_entropy)], yerr=[np.std(salineOnly_entropy, ddof=0)/np.sqrt(n_mice)],
              color='#FFA500', marker='o', markersize=6, capsize=4, lw=2)

sig_ent = '***' if p_ent < 0.001 else '**' if p_ent < 0.01 else '*' if p_ent < 0.05 else 'n.s.'
ymax = max(np.max(saline_entropy), np.max(salineOnly_entropy)) * 1.05
ax_c.plot([0, 1], [ymax, ymax], 'k-', lw=0.8)
ax_c.text(0.5, ymax * 1.02, sig_ent, ha='center', fontsize=7)

ax_c.set_xticks([0, 1])
ax_c.set_xticklabels(['Saline\nbaseline', 'SalineOnly\n(post-cocaine)'], fontsize=7)
ax_c.set_ylabel('Behavioral entropy (norm.)', fontsize=8)
ax_c.set_title('Behavioral entropy', fontsize=8, fontweight='bold', pad=12)
ax_c.text(-0.2, 1.15, 'C', transform=ax_c.transAxes, fontsize=10, fontweight='bold', va='top')

save_fig(fig, output_folder, 'SuppFig11_Context_Specificity')
print(f'\nDone — N={n_mice} mice')

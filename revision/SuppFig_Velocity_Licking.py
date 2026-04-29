#%% SuppFig: Velocity During Licking — Locomotive vs Stationary
"""
Addresses R4-Fig4HI (clarify locomotive licking; <0.5 cm/s stationary?).

Panels:
  A-B: Velocity CDFs during floor/wall licking across sessions
  C: Fraction of licking frames below 0.5 cm/s threshold across sessions
  D: Velocity distribution comparison for saline3 vs cocaine5

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
    mm, second, minute, FPS, n_behaviors, behaviors, colors,
    trials_sal_coc as trials, trial_colors, cohorts_6 as cohorts,
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

FLOOR_LICKING = 2
WALL_LICKING = 3
VEL_THRESHOLD = 0.5  # cm/s

# ---------------------------------------------------------------------------
# Compute velocity during licking per session
# ---------------------------------------------------------------------------
# Per mouse per trial: fraction of floor/wall licking frames below threshold
frac_below_threshold = np.full((n_mice, len(trials)), np.nan)
# Also collect pooled velocities for CDF
velocity_during_licking = {trial: [] for trial in trials}

for m_idx, mouse in enumerate(valid_mice):
    for t_idx, trial in enumerate(trials):
        if trial not in MT[mouse]:
            continue
        preds = MT[mouse][trial]['merged']
        if 'topcam' not in MT[mouse][trial] or 'velocity' not in MT[mouse][trial]['topcam']:
            continue
        velocity = MT[mouse][trial]['topcam']['velocity'] * FPS * 10  # convert to cm/s

        # Licking frames (floor + wall)
        lick_mask = (preds == FLOOR_LICKING) | (preds == WALL_LICKING)
        min_len = min(len(preds), len(velocity))
        lick_mask = lick_mask[:min_len]
        vel = velocity[:min_len]

        lick_vel = vel[lick_mask]
        if len(lick_vel) > 0:
            frac_below_threshold[m_idx, t_idx] = np.mean(lick_vel < VEL_THRESHOLD)
            velocity_during_licking[trial].extend(lick_vel.tolist())

# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
print('\n--- Fraction of licking frames below 0.5 cm/s ---')
for t_idx, trial in enumerate(trials):
    vals = frac_below_threshold[:, t_idx]
    valid = ~np.isnan(vals)
    if valid.sum() > 0:
        print(f'  {trial}: {np.nanmean(vals):.3f} ± {np.nanstd(vals, ddof=0)/np.sqrt(valid.sum()):.3f} '
              f'(n={valid.sum()})')

# ===========================================================================
# BUILD THE FIGURE
# ===========================================================================
print('\nGenerating figure...')
fig = plt.figure(figsize=(180 * mm, 120 * mm))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# --- Panel A: Velocity CDF during floor licking (saline3 vs cocaine5) ---
ax_a = fig.add_subplot(gs[0, 0])
for trial, color, label in [('saline3', '#808080', 'Saline 3'), ('cocaine5', '#990000', 'Cocaine 5')]:
    vels = np.array(velocity_during_licking[trial])
    if len(vels) > 0:
        sorted_v = np.sort(vels)
        # Clip for display
        sorted_v = sorted_v[sorted_v < 5]
        cdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
        ax_a.step(sorted_v, cdf, color=color, lw=1.2, label=f'{label} (n={len(vels)})')

ax_a.axvline(x=VEL_THRESHOLD, color='gray', ls='--', lw=0.8, alpha=0.5)
ax_a.set_xlabel('Velocity (cm/s)', fontsize=8)
ax_a.set_ylabel('CDF', fontsize=8)
ax_a.set_title('Velocity during surface licking', fontsize=8, fontweight='bold', pad=12)
ax_a.legend(fontsize=6, loc='lower right')
ax_a.set_xlim(0, 5)
ax_a.text(-0.15, 1.15, 'A', transform=ax_a.transAxes, fontsize=10, fontweight='bold', va='top')

# --- Panel B: Velocity histograms during licking ---
ax_b = fig.add_subplot(gs[0, 1])
bins_v = np.linspace(0, 5, 100)
for trial, color, label in [('saline3', '#808080', 'Saline 3'), ('cocaine5', '#990000', 'Cocaine 5')]:
    vels = np.array(velocity_during_licking[trial])
    vels = vels[vels < 5]
    if len(vels) > 0:
        ax_b.hist(vels, bins=bins_v, density=True, alpha=0.5, color=color, label=label)

ax_b.axvline(x=VEL_THRESHOLD, color='gray', ls='--', lw=0.8, alpha=0.5)
ax_b.set_xlabel('Velocity (cm/s)', fontsize=8)
ax_b.set_ylabel('Density', fontsize=8)
ax_b.set_title('Velocity distribution during licking', fontsize=8, fontweight='bold', pad=12)
ax_b.legend(fontsize=6, loc='upper right')
ax_b.text(-0.15, 1.15, 'B', transform=ax_b.transAxes, fontsize=10, fontweight='bold', va='top')

# --- Panel C: Fraction below threshold across sessions ---
ax_c = fig.add_subplot(gs[1, 0])
mean_frac = np.nanmean(frac_below_threshold, axis=0)
sem_frac = np.nanstd(frac_below_threshold, axis=0, ddof=0) / np.sqrt(np.sum(~np.isnan(frac_below_threshold), axis=0).clip(1))

for t_idx in range(len(trials)):
    ax_c.bar(t_idx, mean_frac[t_idx], color=trial_colors[t_idx], width=0.7, alpha=0.8)
    ax_c.errorbar(t_idx, mean_frac[t_idx], yerr=sem_frac[t_idx],
                  color='black', capsize=2, capthick=0.5, lw=0.5, fmt='none')

ax_c.set_xticks(range(len(trials)))
ax_c.set_xticklabels(['S1','S2','S3','C1','C2','C3','C4','C5'], fontsize=7)
ax_c.set_ylabel(f'Fraction < {VEL_THRESHOLD} cm/s', fontsize=8)
ax_c.set_title('Fraction of stationary licking frames', fontsize=8, fontweight='bold', pad=12)
ax_c.tick_params(labelsize=7)
ax_c.text(-0.15, 1.15, 'C', transform=ax_c.transAxes, fontsize=10, fontweight='bold', va='top')

# --- Panel D: Floor lick vs wall lick velocity comparison ---
ax_d = fig.add_subplot(gs[1, 1])
for b, bname, bcolor in [(FLOOR_LICKING, 'Floor lick', colors[FLOOR_LICKING]),
                          (WALL_LICKING, 'Wall lick', colors[WALL_LICKING])]:
    mean_vals = []
    sem_vals = []
    for trial in trials:
        vels_per_mouse = []
        for m_idx, mouse in enumerate(valid_mice):
            if trial not in MT[mouse]:
                continue
            preds = MT[mouse][trial]['merged']
            if 'topcam' not in MT[mouse][trial] or 'velocity' not in MT[mouse][trial]['topcam']:
                continue
            velocity = MT[mouse][trial]['topcam']['velocity'] * FPS * 10
            min_len = min(len(preds), len(velocity))
            mask = preds[:min_len] == b
            v = velocity[:min_len][mask]
            if len(v) > 0:
                vels_per_mouse.append(np.mean(v))
        if vels_per_mouse:
            mean_vals.append(np.mean(vels_per_mouse))
            sem_vals.append(np.std(vels_per_mouse, ddof=0) / np.sqrt(len(vels_per_mouse)))
        else:
            mean_vals.append(np.nan)
            sem_vals.append(np.nan)

    ax_d.errorbar(range(len(trials)), mean_vals, yerr=sem_vals,
                  color=bcolor, lw=1.2, capsize=2, marker='o', markersize=3, label=bname)

ax_d.set_xticks(range(len(trials)))
ax_d.set_xticklabels(['S1','S2','S3','C1','C2','C3','C4','C5'], fontsize=7)
ax_d.set_ylabel('Mean velocity (cm/s)', fontsize=8)
ax_d.set_title('Velocity by licking type across sessions', fontsize=8, fontweight='bold', pad=12)
ax_d.legend(fontsize=6, loc='lower left')
ax_d.text(-0.15, 1.15, 'D', transform=ax_d.transAxes, fontsize=10, fontweight='bold', va='top')

save_fig(fig, output_folder, 'SuppFig12_Velocity_Licking')
print(f'\nDone — N={n_mice} mice')

#%% SuppFig: dSPN vs iSPN Bout Duration Control (Behavioral Only)
"""
Addresses R4-Fig2L (activation difference explained by bout duration?).

NOTE: Full analysis requires photometry data (CMT_Dec24.pkl / master_table.pkl)
which are not available locally. This script provides the behavioral component:
comparing bout durations of floor/wall licking between dSPN-recorded and
iSPN-recorded animals during saline and cocaine sessions.

Panels:
  A-B: Bout duration CDFs for floor licking and wall licking, dSPN vs iSPN mice
  C: Mean bout duration across sessions for both groups

Uses CMT_Aug24.pkl. dSPN mice = drd1_hm4di gcamp_mice. iSPN mice = a2a_hm3dq/a2a_hm4di gcamp_mice.
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
    mm, second, minute, FPS, n_behaviors, behaviors, colors,
    trials_sal_coc as trials, trial_colors,
    dSPN_color, iSPN_color,
    load_CMT, setup_style, save_fig
)

setup_style()
output_folder = 'revision/output/'
os.makedirs(output_folder, exist_ok=True)

# ---------------------------------------------------------------------------
# Identify GCaMP mice by pathway
# ---------------------------------------------------------------------------
# From CLAUDE.md and Figure5 exclusion lists:
dSPN_mice = ['c528m5', 'c528m10', 'c548m1']          # drd1_hm4di gcamp mice
iSPN_mice = ['c514Bm2', 'c514Bm8',                    # drd1_hm3dq gcamp
             'cA184m4', 'cA184m7',                     # a2a_hm3dq gcamp
             'cA242m5', 'cA242m6', 'cA242m8']          # a2a_hm3dq gcamp

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
CMT = load_CMT('Dec24')

# Flatten to find our target mice
MT = {}
for cohort in CMT:
    for m in CMT[cohort]:
        if m in dSPN_mice or m in iSPN_mice:
            MT[m] = CMT[cohort][m]
del CMT

found_dSPN = [m for m in dSPN_mice if m in MT]
found_iSPN = [m for m in iSPN_mice if m in MT]
print(f'dSPN mice found: {len(found_dSPN)} — {found_dSPN}')
print(f'iSPN mice found: {len(found_iSPN)} — {found_iSPN}')

target_trials = ['saline3', 'cocaine3', 'cocaine5']
bout_theta = 8  # frames

# ---------------------------------------------------------------------------
# Compute bout durations per group
# ---------------------------------------------------------------------------
FLOOR_LICKING = 2
WALL_LICKING = 3
target_behaviors = [FLOOR_LICKING, WALL_LICKING]
beh_names = ['Floor licking', 'Wall licking']

bout_durs = {}
for group_name, group_mice, color in [('dSPN', found_dSPN, dSPN_color), ('iSPN', found_iSPN, iSPN_color)]:
    bout_durs[group_name] = {}
    for trial in target_trials:
        bout_durs[group_name][trial] = {b: [] for b in target_behaviors}
        for mouse in group_mice:
            if trial not in MT[mouse]:
                continue
            preds = MT[mouse][trial]['merged']
            for b in target_behaviors:
                bouts = hf.segment_bouts(preds, b, bout_theta)
                bout_durs[group_name][trial][b].extend([l / FPS for l in bouts['length']])

# Per-mouse mean bout duration across sessions
mean_dur_per_mouse = {}
for group_name, group_mice in [('dSPN', found_dSPN), ('iSPN', found_iSPN)]:
    mean_dur_per_mouse[group_name] = {b: np.full((len(group_mice), len(trials)), np.nan)
                                      for b in target_behaviors}
    for m_idx, mouse in enumerate(group_mice):
        for t_idx, trial in enumerate(trials):
            if trial not in MT[mouse]:
                continue
            preds = MT[mouse][trial]['merged']
            for b in target_behaviors:
                bouts = hf.segment_bouts(preds, b, bout_theta)
                if len(bouts['length']) > 0:
                    mean_dur_per_mouse[group_name][b][m_idx, t_idx] = np.mean(bouts['length']) / FPS

# Stats
print('\n--- Bout duration comparison (cocaine5) ---')
for b, bname in zip(target_behaviors, beh_names):
    d_durs = np.array(bout_durs['dSPN'].get('cocaine5', {}).get(b, []))
    i_durs = np.array(bout_durs['iSPN'].get('cocaine5', {}).get(b, []))
    if len(d_durs) > 2 and len(i_durs) > 2:
        ks, p = stats.ks_2samp(d_durs, i_durs)
        mw, p_mw = stats.mannwhitneyu(d_durs, i_durs)
        print(f'  {bname}: dSPN median={np.median(d_durs):.1f}s (n={len(d_durs)}) '
              f'iSPN median={np.median(i_durs):.1f}s (n={len(i_durs)}) '
              f'KS={ks:.3f} p={p:.4f} MWU p={p_mw:.4f}')

# ===========================================================================
# BUILD THE FIGURE
# ===========================================================================
print('\nGenerating figure...')
fig = plt.figure(figsize=(180 * mm, 65 * mm))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4)

# --- Panels A-B: Bout duration CDFs for floor/wall licking ---
for b_idx, (b, bname) in enumerate(zip(target_behaviors, beh_names)):
    ax = fig.add_subplot(gs[b_idx])
    for group_name, color, ls in [('dSPN', dSPN_color, '-'), ('iSPN', iSPN_color, '--')]:
        durs = np.array(bout_durs[group_name].get('cocaine5', {}).get(b, []))
        if len(durs) > 0:
            sorted_d = np.sort(durs)
            cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
            ax.step(sorted_d, cdf, color=color, lw=1.2, ls=ls,
                    label=f'{group_name} (n={len(durs)})')

    ax.set_xlabel('Duration (s)', fontsize=8)
    ax.set_ylabel('CDF', fontsize=8)
    ax.set_title(f'{bname} — Cocaine 5', fontsize=7, fontweight='bold', pad=12)
    ax.set_xlim(0, 30)
    ax.legend(fontsize=5, loc='lower right')
    panel = chr(ord('A') + b_idx)
    ax.text(-0.15, 1.15, panel, transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')

# --- Panel C: Mean bout duration across sessions ---
ax_c = fig.add_subplot(gs[2])
for group_name, color, ls in [('dSPN', dSPN_color, '-'), ('iSPN', iSPN_color, '--')]:
    vals = mean_dur_per_mouse[group_name][FLOOR_LICKING]
    n = vals.shape[0]
    mean_v = np.nanmean(vals, axis=0)
    sem_v = np.nanstd(vals, axis=0, ddof=0) / max(np.sqrt(n), 1)
    ax_c.errorbar(range(len(trials)), mean_v, yerr=sem_v, color=color, lw=1.2,
                  capsize=2, marker='o', markersize=3, ls=ls, label=f'{group_name} (N={n})')

ax_c.set_xticks(range(len(trials)))
ax_c.set_xticklabels(['S1','S2','S3','C1','C2','C3','C4','C5'], fontsize=6)
ax_c.set_ylabel('Mean bout duration (s)', fontsize=8)
ax_c.set_title('Floor licking bout duration', fontsize=7, fontweight='bold', pad=12)
ax_c.legend(fontsize=5, loc='lower right')
ax_c.text(-0.15, 1.15, 'C', transform=ax_c.transAxes, fontsize=10, fontweight='bold', va='top')

save_fig(fig, output_folder, 'SuppFig9_dSPN_vs_iSPN_BoutDur')
print(f'\nDone — dSPN: {len(found_dSPN)} mice, iSPN: {len(found_iSPN)} mice')
print('\nNOTE: Full analysis (photometry amplitude correlation) requires CMT_Dec24.pkl')

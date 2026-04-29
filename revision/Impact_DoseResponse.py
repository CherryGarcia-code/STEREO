#%% Dose-response / session-response curve for surface licking sensitization
"""
Plots surface licking fraction as a function of cocaine session number,
fits with a sigmoid (logistic) curve, and reports the half-max session (EC50).
Output: revision/output/
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.optimize as opt

from revision_utils import (
    mm, n_behaviors, behaviors, colors,
    trials_sal_coc as trials, trial_colors,
    cohorts_6 as cohorts,
    load_CMT, flatten_CMT, setup_style, save_fig,
    PATHO_LICKING, grouping_lut
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
print(f'N = {n_mice} mice')

# Compute surface licking fraction per session per mouse
n_trials = len(trials)
surface_frac = np.full((n_mice, n_trials), np.nan)
for m_idx, mouse in enumerate(valid_mice):
    for t_idx, trial in enumerate(trials):
        preds = MT[mouse][trial]['merged']
        grouped = grouping_lut[preds]
        surface_frac[m_idx, t_idx] = np.count_nonzero(grouped == PATHO_LICKING) / preds.size

mean_sf = np.nanmean(surface_frac, axis=0)
sem_sf = np.nanstd(surface_frac, axis=0, ddof=0) / np.sqrt(n_mice)

# ---------------------------------------------------------------------------
# Fit sigmoid to cocaine sessions only (sessions 4-8, x=1..5)
# ---------------------------------------------------------------------------
x_coc = np.arange(1, 6)  # cocaine session number
y_coc = mean_sf[3:]       # cocaine1-5

def sigmoid(x, L, k, x0, b):
    return L / (1 + np.exp(-k * (x - x0))) + b

try:
    popt, pcov = opt.curve_fit(sigmoid, x_coc, y_coc,
                                p0=[0.3, 2.0, 3.0, 0.1],
                                bounds=([0, 0, 0, 0], [1, 10, 6, 0.5]),
                                maxfev=5000)
    x_fit = np.linspace(0.5, 5.5, 200)
    y_fit = sigmoid(x_fit, *popt)
    ec50 = popt[2]
    fit_success = True
    print(f'Sigmoid fit: L={popt[0]:.3f}, k={popt[1]:.3f}, EC50={ec50:.2f}, b={popt[3]:.3f}')
except Exception as e:
    print(f'Sigmoid fit failed: {e}')
    fit_success = False

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(160 * mm, 60 * mm))

# Panel A: Full trajectory across all sessions
x_all = np.arange(n_trials)
ax1.errorbar(x_all, mean_sf, yerr=sem_sf, color='k', marker='o', markersize=4,
             lw=1.5, capsize=3, capthick=0.8)
for i in range(n_trials):
    ax1.scatter(i, mean_sf[i], color=trial_colors[i], s=40, zorder=5, edgecolors='none')

# Individual mice in background
for m_idx in range(n_mice):
    ax1.plot(x_all, surface_frac[m_idx, :], color='gray', alpha=0.1, lw=0.3)

ax1.set_xticks(x_all)
ax1.set_xticklabels(['S1','S2','S3','C1','C2','C3','C4','C5'], fontsize=7)
ax1.set_ylabel('Surface licking fraction', fontsize=9)
ax1.set_xlabel('Session', fontsize=9)
ax1.set_title('Sensitization trajectory', fontsize=9, fontweight='bold', pad=12)
ax1.axvspan(-0.5, 2.5, color='gray', alpha=0.05)
ax1.text(1, ax1.get_ylim()[1] * 0.95, 'Saline', ha='center', fontsize=6, color='gray')
ax1.text(-0.15, 1.15, 'A', transform=ax1.transAxes, fontsize=12, fontweight='bold', va='top')

# Panel B: Cocaine sessions only + sigmoid fit
ax2.errorbar(x_coc, y_coc, yerr=sem_sf[3:], color='k', marker='o', markersize=5,
             lw=1.5, capsize=3, capthick=0.8, zorder=5)
for i, xc in enumerate(x_coc):
    ax2.scatter(xc, y_coc[i], color=trial_colors[3 + i], s=50, zorder=6, edgecolors='none')

# Individual mice
for m_idx in range(n_mice):
    ax2.plot(x_coc, surface_frac[m_idx, 3:], color='gray', alpha=0.1, lw=0.3)

if fit_success:
    ax2.plot(x_fit, y_fit, color='#d73027', lw=2, ls='--', alpha=0.8, label=f'Sigmoid (EC50={ec50:.1f})')
    ax2.axvline(ec50, color='#d73027', ls=':', lw=0.8, alpha=0.5)
    ax2.legend(fontsize=6, loc='upper left')

ax2.set_xticks(x_coc)
ax2.set_xticklabels(['C1','C2','C3','C4','C5'], fontsize=7)
ax2.set_ylabel('Surface licking fraction', fontsize=9)
ax2.set_xlabel('Cocaine session', fontsize=9)
ax2.set_title('Sigmoid fit (cocaine only)', fontsize=9, fontweight='bold', pad=12)
ax2.text(-0.15, 1.15, 'B', transform=ax2.transAxes, fontsize=12, fontweight='bold', va='top')

fig.tight_layout()
save_fig(fig, output_folder, 'Impact_DoseResponse')
print(f'\nDone — N={n_mice} mice')

#%% DREADDs effect size forest plot
"""
Forest plot showing Cohen's d for surface licking (Pre-CNO vs During-CNO)
across all 5 DREADD cohorts, directly visualizing bidirectional predictions.
Output: revision/output/
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats as stats

from revision_utils import (
    mm, second, minute, n_behaviors, behaviors, colors, FPS,
    grouping_lut, category_names, category_colors,
    PATHO_LICKING, NATURAL_LICKING, NO_LICKING,
    gcamp_mice, subOptimal_infection,
    load_CMT, setup_style, save_fig
)

setup_style()
output_folder = 'revision/output/'
os.makedirs(output_folder, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data + sandwich detection (same as DREADDs script)
# ---------------------------------------------------------------------------
CMT = load_CMT('Dec24')

dreadd_cohorts = ['drd1_hm4di', 'drd1_hm3dq', 'controls', 'a2a_hm4di', 'a2a_hm3dq']
dreadd_labels = {
    'drd1_hm4di': r'Drd1$_{hm4Di}$ (inhib. dSPN)',
    'drd1_hm3dq': r'Drd1$_{hm3Dq}$ (excit. dSPN)',
    'controls': 'Controls (no DREADD)',
    'a2a_hm4di': r'A2a$_{hm4Di}$ (inhib. iSPN)',
    'a2a_hm3dq': r'A2a$_{hm3Dq}$ (excit. iSPN)',
}

# Expected direction: negative d = CNO reduces licking, positive d = CNO increases
expected_sign = {
    'drd1_hm4di': -1,   # inhibit dSPN → reduce licking
    'drd1_hm3dq': +1,   # excite dSPN → increase licking
    'controls': 0,       # no effect
    'a2a_hm4di': +1,    # inhibit iSPN → increase licking
    'a2a_hm3dq': -1,    # excite iSPN → reduce licking
}

CNO2cocaineGap = {
    'drd1_hm4di': {'c512m3':30,'c512m4':30,'c512m7':30,'c526m2':32,'c526m3':35,'c528m5':31,'c528m10':38,'c548m1':31},
    'drd1_hm3dq': {'c514Bm2':35,'c514Bm8':35,'c514m1':35,'c514m3':38,'c514m5':32},
    'controls': {'c548m8':33,'c548m10':32,'c548m11':32,'cA242m4':30,'cA242m9':30},
    'a2a_hm4di': {'cA154m4':35,'cA154m6':36,'cA156m1':35,'cA156m6':35,'cA156m7':35,'cA156m8':34},
    'a2a_hm3dq': {'cA156m2':33,'cA156m5':34,'cA158m2':30,'cA158m3':30,'cA158m4':30,
                   'cA184m4':33,'cA184m7':33,'cA242m5':30,'cA242m6':30,'cA242m8':30},
}
possible_trials = {
    'cocaineOnly': [f'cocaine{i}' for i in range(1, 11)],
    'cocaineCNO': ['cocaine6afterCNO', 'cocaine7afterCNO', 'cocaine8afterCNO', 'cocaine9afterCNO'],
}
CNO_MAX_TIME = 50
UNDEFINED = 1

# Find sandwich days and compute surface licking
effect_sizes = {}
for cohort in dreadd_cohorts:
    MT = CMT[cohort]
    pre_vals = []
    cno_vals = []

    for m in MT:
        if m in gcamp_mice or m in subOptimal_infection:
            continue
        trials = list(MT[m].keys())
        sandwich_found = False
        for t_idx in range(1, len(trials) - 1):
            if (trials[t_idx] in possible_trials['cocaineCNO'] and
                trials[t_idx + 1] in possible_trials['cocaineOnly']):
                if trials[t_idx - 1] in possible_trials['cocaineOnly']:
                    pre_tr = trials[t_idx - 1]
                elif t_idx >= 2 and 'CNOonly' in trials[t_idx - 1] and trials[t_idx - 2] in possible_trials['cocaineOnly']:
                    pre_tr = trials[t_idx - 2]
                else:
                    continue

                cutoff_min = CNO2cocaineGap.get(cohort, {}).get(m, 30)
                cutoff = (CNO_MAX_TIME - cutoff_min) * minute

                # Pre-CNO surface licking
                preds_pre = MT[m][pre_tr]['merged'][:cutoff]
                preds_pre = preds_pre[preds_pre != UNDEFINED]
                grouped_pre = grouping_lut[preds_pre]
                sl_pre = np.count_nonzero(grouped_pre == PATHO_LICKING) / preds_pre.size if preds_pre.size > 0 else np.nan

                # During-CNO surface licking
                preds_cno = MT[m][trials[t_idx]]['merged'][:cutoff]
                preds_cno = preds_cno[preds_cno != UNDEFINED]
                grouped_cno = grouping_lut[preds_cno]
                sl_cno = np.count_nonzero(grouped_cno == PATHO_LICKING) / preds_cno.size if preds_cno.size > 0 else np.nan

                pre_vals.append(sl_pre)
                cno_vals.append(sl_cno)
                sandwich_found = True
                break

    pre_arr = np.array(pre_vals)
    cno_arr = np.array(cno_vals)
    n = len(pre_vals)

    if n >= 2:
        # Cohen's d for paired samples
        diff = cno_arr - pre_arr
        d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
        # 95% CI via t-distribution
        se_d = np.sqrt(1/n + d**2 / (2*n))
        t_crit = stats.t.ppf(0.975, n - 1)
        ci_lo = d - t_crit * se_d
        ci_hi = d + t_crit * se_d
        # p-value
        t_stat, p_val = stats.ttest_rel(pre_arr, cno_arr)
    else:
        d, ci_lo, ci_hi, p_val = 0, 0, 0, 1.0

    effect_sizes[cohort] = {'d': d, 'ci_lo': ci_lo, 'ci_hi': ci_hi, 'n': n, 'p': p_val}
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
    print(f'{cohort}: d={d:.3f} [{ci_lo:.3f}, {ci_hi:.3f}] n={n} p={p_val:.4f} {sig}')

del CMT

# ---------------------------------------------------------------------------
# Forest Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(120 * mm, 75 * mm))

y_pos = np.arange(len(dreadd_cohorts))[::-1]  # top to bottom
cohort_colors = {
    'drd1_hm4di': dSPN_color if 'dSPN_color' in dir() else '#d73027',
    'drd1_hm3dq': '#e57373',
    'controls': '#808080',
    'a2a_hm4di': '#721515',
    'a2a_hm3dq': '#b35a5a',
}
# Use fixed colors
from revision_utils import dSPN_color, iSPN_color
cohort_colors = {
    'drd1_hm4di': dSPN_color,
    'drd1_hm3dq': '#e57373',
    'controls': '#808080',
    'a2a_hm4di': iSPN_color,
    'a2a_hm3dq': '#b35a5a',
}

for idx, cohort in enumerate(dreadd_cohorts):
    es = effect_sizes[cohort]
    y = y_pos[idx]

    # CI line
    ax.plot([es['ci_lo'], es['ci_hi']], [y, y], color=cohort_colors[cohort], lw=2.5, solid_capstyle='round')
    # Point estimate
    ax.scatter(es['d'], y, color=cohort_colors[cohort], s=80, zorder=5, edgecolors='white', linewidths=0.5)

    # Significance marker
    sig = '***' if es['p'] < 0.001 else '**' if es['p'] < 0.01 else '*' if es['p'] < 0.05 else ''
    if sig:
        ax.text(es['ci_hi'] + 0.1, y, sig, va='center', fontsize=8, fontweight='bold',
                color=cohort_colors[cohort])

    # N annotation
    ax.text(es['ci_hi'] + 0.3 + (0.15 if sig else 0), y, f'n={es["n"]}', va='center',
            fontsize=6, color='gray')

ax.axvline(x=0, color='black', ls='--', lw=0.8, alpha=0.5)

# Expected direction annotations
ax.text(ax.get_xlim()[0], len(dreadd_cohorts) + 0.3,
        r'$\leftarrow$ CNO reduces licking', fontsize=6, color='gray', ha='left')
ax.text(ax.get_xlim()[1], len(dreadd_cohorts) + 0.3,
        r'CNO increases licking $\rightarrow$', fontsize=6, color='gray', ha='right')

ax.set_yticks(y_pos)
ax.set_yticklabels([dreadd_labels[c] for c in dreadd_cohorts], fontsize=7)
ax.set_xlabel("Cohen's d (CNO - Pre-CNO)", fontsize=9)
ax.set_title('DREADDs Effect Size: Surface Licking', fontsize=9, fontweight='bold', pad=12)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='y', length=0)

fig.tight_layout()
save_fig(fig, output_folder, 'Impact_ForestPlot')
print(f'\nDone — Forest plot saved')

#%% SuppFig: DREADDs Full Bidirectional Comparison — All 5 Cohorts
"""
Addresses R4-DREADDs (opposite effects for ALL parameters?).

Composite figure: one row per DREADD cohort (drd1_hm4di, drd1_hm3dq, controls,
a2a_hm4di, a2a_hm3dq). Columns: grouped V-graphs, individual behavior V-graphs,
bout duration CDFs.

Uses CMT_Aug24.pkl with sandwich detection logic from Figure5_June2025.py.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats
import helper_functions as hf

from revision_utils import (
    mm, second, minute, n_behaviors, behaviors, colors, lut, FPS,
    grouping_lut, category_names, category_colors,
    PATHO_LICKING, NATURAL_LICKING, NO_LICKING,
    gcamp_mice, subOptimal_infection,
    load_CMT, setup_style, save_fig
)

setup_style()
output_folder = 'revision/output/'
os.makedirs(output_folder, exist_ok=True)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
CMT = load_CMT('Dec24')

dreadd_cohorts = ['drd1_hm4di', 'drd1_hm3dq', 'controls', 'a2a_hm4di', 'a2a_hm3dq']
dreadd_labels = {
    'drd1_hm4di': r'Drd1$_{hm4Di}$', 'drd1_hm3dq': r'Drd1$_{hm3Dq}$',
    'controls': 'Controls', 'a2a_hm4di': r'A2a$_{hm4Di}$', 'a2a_hm3dq': r'A2a$_{hm3Dq}$',
}

# CNO-to-cocaine gap per mouse (from Figure5_June2025.py)
CNO2cocaineGap = {
    'drd1_hm4di': {'c512m3':30,'c512m4':30,'c512m7':30,'c526m2':32,'c526m3':35,'c528m5':31,'c528m10':38,'c548m1':31},
    'drd1_hm3dq': {'c514Bm2':35,'c514Bm8':35,'c514m1':35,'c514m3':38,'c514m5':32},
    'controls': {'c548m8':33,'c548m10':32,'c548m11':32,'cA242m4':30,'cA242m9':30},
    'a2a_hm4di': {'cA154m4':35,'cA154m6':36,'cA156m1':35,'cA156m6':35,'cA156m7':35,'cA156m8':34},
    'a2a_hm3dq': {'cA156m2':33,'cA156m5':34,'cA158m2':30,'cA158m3':30,'cA158m4':30,'cA184m4':33,'cA184m7':33,'cA242m5':30,'cA242m6':30,'cA242m8':30},
}

possible_trials = {
    'cocaineOnly': [f'cocaine{i}' for i in range(1, 11)],
    'cocaineCNO': ['cocaine6afterCNO', 'cocaine7afterCNO', 'cocaine8afterCNO', 'cocaine9afterCNO'],
}

CNO_MAX_TIME = 50
PRE, DURING, POST = 0, 1, 2
UNDEFINED = 1

# Expected CNO directions for stats
expected_direction = {
    'drd1_hm4di': ['greater', 'less'],  # inhibit dSPN → reduce licking
    'drd1_hm3dq': ['less', 'greater'],  # excite dSPN → increase licking
    'controls': ['greater', 'less'],     # no effect expected
    'a2a_hm4di': ['less', 'greater'],   # inhibit iSPN → increase licking
    'a2a_hm3dq': ['greater', 'less'],   # excite iSPN → reduce licking
}

# ---------------------------------------------------------------------------
# Find sandwich days per cohort (adapted from Figure5)
# ---------------------------------------------------------------------------
all_sandwich = {}
for cohort in dreadd_cohorts:
    MT = CMT[cohort]
    sandwich = {}
    for m in MT:
        if m in gcamp_mice or m in subOptimal_infection:
            continue
        trials = list(MT[m].keys())
        for t_idx in range(1, len(trials) - 1):
            if (trials[t_idx] in possible_trials['cocaineCNO'] and
                trials[t_idx + 1] in possible_trials['cocaineOnly']):
                if trials[t_idx - 1] in possible_trials['cocaineOnly']:
                    pre_tr = trials[t_idx - 1]
                elif t_idx >= 2 and 'CNOonly' in trials[t_idx - 1] and trials[t_idx - 2] in possible_trials['cocaineOnly']:
                    pre_tr = trials[t_idx - 2]
                else:
                    continue

                sandwich[m] = {
                    'pre': MT[m][pre_tr]['merged'],
                    'CNO': MT[m][trials[t_idx]]['merged'],
                    'post': MT[m][trials[t_idx + 1]]['merged'],
                }
                # Also store velocity if available
                for key, trial_name in [('pre', pre_tr), ('CNO', trials[t_idx]), ('post', trials[t_idx + 1])]:
                    if 'topcam' in MT[m][trial_name] and 'velocity' in MT[m][trial_name]['topcam']:
                        sandwich[m][key + '_vel'] = MT[m][trial_name]['topcam']['velocity'] * FPS * 10
                break
    all_sandwich[cohort] = sandwich
    print(f'{cohort}: {len(sandwich)} mice with sandwich days')

del CMT

# ---------------------------------------------------------------------------
# Compute grouped fractions per cohort
# ---------------------------------------------------------------------------
results = {}
for cohort in dreadd_cohorts:
    sandwich = all_sandwich[cohort]
    mice = sorted(sandwich.keys())
    n_mice = len(mice)
    if n_mice == 0:
        continue

    grouped_frac = np.full((n_mice, 3, 3), np.nan)  # (mice, category, day)
    indiv_frac = np.full((n_mice, n_behaviors, 3), np.nan)  # (mice, behavior, day)

    for m_idx, m in enumerate(mice):
        cutoff_min = CNO2cocaineGap.get(cohort, {}).get(m, 30)
        cutoff = (CNO_MAX_TIME - cutoff_min) * minute

        for d_idx, key in enumerate(['pre', 'CNO', 'post']):
            preds = sandwich[m][key][:cutoff]
            preds = preds[preds != UNDEFINED]  # exclude undefined
            total = preds.size
            if total == 0:
                continue

            # Grouped fractions
            grouped = grouping_lut[preds]
            for cat in [PATHO_LICKING, NATURAL_LICKING, NO_LICKING]:
                grouped_frac[m_idx, cat, d_idx] = np.count_nonzero(grouped == cat) / total

            # Individual fractions
            for b in range(n_behaviors):
                indiv_frac[m_idx, b, d_idx] = np.count_nonzero(preds == b) / total

    results[cohort] = {
        'mice': mice, 'n_mice': n_mice,
        'grouped_frac': grouped_frac,
        'indiv_frac': indiv_frac,
    }

# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
print('\n--- DREADDs Stats: Grouped behavior fractions ---')
for cohort in dreadd_cohorts:
    if cohort not in results:
        continue
    r = results[cohort]
    n = r['n_mice']
    print(f'\n{cohort} (N={n}):')
    dirs = expected_direction[cohort]
    for cat, cname in enumerate(category_names):
        pre = r['grouped_frac'][:, cat, PRE]
        dur = r['grouped_frac'][:, cat, DURING]
        post = r['grouped_frac'][:, cat, POST]
        # Two-tailed
        t1, p1 = stats.ttest_rel(pre, dur)
        t2, p2 = stats.ttest_rel(dur, post)
        print(f'  {cname}: pre={np.nanmean(pre):.3f} dur={np.nanmean(dur):.3f} post={np.nanmean(post):.3f} | '
              f'pre-dur t={t1:.2f} p={p1:.4f} | dur-post t={t2:.2f} p={p2:.4f}')

# ===========================================================================
# BUILD THE FIGURE: 5 rows × 4 columns
# ===========================================================================
print('\nGenerating figure...')
fig = plt.figure(figsize=(180 * mm, 250 * mm))
outer = gridspec.GridSpec(5, 1, figure=fig, hspace=0.5)

key_behaviors = [2, 3, 7]  # Floor licking, Wall licking, Locomotion
key_beh_names = ['Floor lick', 'Wall lick', 'Locomotion']

for row_idx, cohort in enumerate(dreadd_cohorts):
    if cohort not in results:
        continue
    r = results[cohort]
    n = r['n_mice']

    gs_row = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[row_idx],
                                              wspace=0.4, width_ratios=[1, 1, 1, 1])

    # --- Col 1: Grouped V-graphs (3 categories overlaid) ---
    ax1 = fig.add_subplot(gs_row[0])
    for cat in [PATHO_LICKING, NATURAL_LICKING, NO_LICKING]:
        fracs = r['grouped_frac'][:, cat, :]
        mean_v = np.nanmean(fracs, axis=0)
        sem_v = np.nanstd(fracs, axis=0, ddof=0) / np.sqrt(n)
        ax1.errorbar(range(3), mean_v, yerr=sem_v, color=category_colors[cat],
                     lw=1.5, capsize=3, capthick=1, marker='o', markersize=3,
                     label=category_names[cat])
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(['-', '+', '-'], fontsize=7)
    ax1.set_xlabel('CNO', fontsize=7)
    ax1.set_ylabel('Fraction', fontsize=7)
    ax1.set_title(f'{dreadd_labels[cohort]} (N={n})', fontsize=7, fontweight='bold', pad=12)
    ax1.tick_params(labelsize=6)
    if row_idx == 0:
        ax1.legend(fontsize=4, loc='upper right')

    # --- Cols 2-4: Individual behavior V-graphs ---
    for col_idx, (b, bname) in enumerate(zip(key_behaviors, key_beh_names)):
        ax = fig.add_subplot(gs_row[1 + col_idx])
        fracs = r['indiv_frac'][:, b, :]

        # Individual lines
        for m in range(n):
            ax.plot(range(3), fracs[m, :], color='gray', alpha=0.2, lw=0.4, zorder=1)

        # Mean + SEM
        mean_v = np.nanmean(fracs, axis=0)
        sem_v = np.nanstd(fracs, axis=0, ddof=0) / np.sqrt(n)
        ax.errorbar(range(3), mean_v, yerr=sem_v, color=colors[b],
                    lw=1.5, capsize=3, capthick=1, marker='o', markersize=4, zorder=5)

        ax.set_xticks(range(3))
        ax.set_xticklabels(['-', '+', '-'], fontsize=7)
        ax.set_xlabel('CNO', fontsize=7)
        ax.set_title(bname, fontsize=6, fontweight='bold', color=colors[b], pad=12)
        ax.tick_params(labelsize=6)

        # Stats
        pre = fracs[:, PRE]; dur = fracs[:, DURING]
        valid = ~(np.isnan(pre) | np.isnan(dur))
        if valid.sum() >= 3:
            _, p = stats.ttest_rel(pre[valid], dur[valid])
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
            ymax = np.nanmax(fracs) * 1.05
            ax.text(0.5, ymax, sig, ha='center', fontsize=6)

save_fig(fig, output_folder, 'SuppFig5_DREADDs_AllCohorts')

# ===========================================================================
# Print comprehensive stats table
# ===========================================================================
print('\n' + '='*80)
print('COMPREHENSIVE STATS TABLE')
print('='*80)
for cohort in dreadd_cohorts:
    if cohort not in results:
        continue
    r = results[cohort]
    n = r['n_mice']
    print(f'\n--- {cohort} (N={n}) ---')
    print(f'{"Behavior":<20} {"Pre":<8} {"CNO":<8} {"Post":<8} {"Pre-CNO t":<10} {"Pre-CNO p":<10} {"CNO-Post t":<10} {"CNO-Post p":<10}')
    for b in range(n_behaviors):
        pre = r['indiv_frac'][:, b, PRE]
        dur = r['indiv_frac'][:, b, DURING]
        post = r['indiv_frac'][:, b, POST]
        valid_pd = ~(np.isnan(pre) | np.isnan(dur))
        valid_dp = ~(np.isnan(dur) | np.isnan(post))
        t1, p1 = (np.nan, np.nan) if valid_pd.sum() < 3 else stats.ttest_rel(pre[valid_pd], dur[valid_pd])
        t2, p2 = (np.nan, np.nan) if valid_dp.sum() < 3 else stats.ttest_rel(dur[valid_dp], post[valid_dp])
        print(f'{behaviors[b]:<20} {np.nanmean(pre):<8.3f} {np.nanmean(dur):<8.3f} {np.nanmean(post):<8.3f} '
              f'{t1:<10.2f} {p1:<10.4f} {t2:<10.2f} {p2:<10.4f}')

print(f'\nDone — figures saved to {output_folder}')

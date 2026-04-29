"""
Integrative Analysis 4: Circuit Imbalance Model

Tests the hypothesis that the balance between direct (dSPN) and indirect
(iSPN) pathway activity determines surface licking severity. Uses behavioral
outcomes from each manipulation cohort, assigns a circuit imbalance score
based on circuit logic, and fits a linear model to observed sensitization.

Circuit imbalance score = dSPN_effect - iSPN_effect where:
  dSPN excited  = +1,  dSPN inhibited  = -1,  not manipulated = 0
  iSPN excited  = +1,  iSPN inhibited  = -1,  not manipulated = 0
  Score = dSPN_effect - iSPN_effect
    (positive = dSPN dominant, expected more licking;
     negative = iSPN dominant, expected less licking)

Sensitization index = cocaine5 surface licking – saline3 surface licking.

Outputs:
  - Int4_CircuitImbalanceModel_4cohorts.png/pdf
  - Int4_CircuitImbalanceModel_6cohorts.png/pdf
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
    mm, n_behaviors, cohorts_4, cohorts_6, cohort_labels,
    load_CMT, flatten_CMT, setup_style, save_fig
)

setup_style()
output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(output_folder, exist_ok=True)

FL, WL = 2, 3   # behavior indices

# Circuit imbalance scores per cohort
# Score = dSPN_effect − iSPN_effect
# dSPN: hm4Di(inhibit)=-1, hm3Dq(excite)=+1, else=0
# iSPN: hm4Di(inhibit)=-1, hm3Dq(excite)=+1, else=0
CIRCUIT_SCORE = {
    'drd1_hm4di': -1,   # dSPN inhibited → less drive
    'drd1_hm3dq': +1,   # dSPN excited   → more drive
    'a2a_hm4di':  +1,   # iSPN inhibited → less brake → net more drive
    'a2a_hm3dq':  -1,   # iSPN excited   → more brake  → net less drive
    'controls':    0,   # no manipulation
    'a2a_opto':    0,   # acute laser (excluded from DREADD model)
}

COHORT_COLOR = {
    'drd1_hm4di': '#d73027', 'drd1_hm3dq': '#f4a582',
    'controls':   '#808080', 'a2a_hm4di': '#721515',
    'a2a_hm3dq':  '#d9a0a0', 'a2a_opto':  '#3399cc',
}

# Score → color (positive = red, zero = grey, negative = blue)
def score_color(s):
    if s > 0:
        return '#d73027'
    elif s < 0:
        return '#3399cc'
    return '#808080'


def surface_licking_frac(preds):
    n = len(preds)
    occ = np.bincount(preds.astype(int), minlength=n_behaviors) / n
    return occ[FL] + occ[WL]


def run_analysis(cohorts, suffix):
    print(f'\n=== Circuit Imbalance Model ({suffix}) ===')
    CMT = load_CMT('Dec24')
    MT, mouse_to_cohort = flatten_CMT(CMT, cohorts)
    del CMT

    # Exclude opto cohort from DREADD model — it uses a different design
    dreadd_cohorts = [c for c in cohorts if c != 'a2a_opto']

    # Per mouse: cocaine5 − saline3 surface licking (sensitization index)
    cohort_data = {c: [] for c in dreadd_cohorts}

    for m in MT:
        c = mouse_to_cohort[m]
        if c not in cohort_data:
            continue
        if 'saline3' not in MT[m] or 'cocaine5' not in MT[m]:
            continue
        sl_sal = surface_licking_frac(MT[m]['saline3']['merged'])
        sl_coc = surface_licking_frac(MT[m]['cocaine5']['merged'])
        if np.isnan(sl_sal) or np.isnan(sl_coc):
            continue
        cohort_data[c].append(sl_coc - sl_sal)

    # Summary statistics
    cohort_order = [c for c in dreadd_cohorts if len(cohort_data[c]) > 0]
    means  = np.array([np.mean(cohort_data[c]) for c in cohort_order])
    sems   = np.array([np.std(cohort_data[c]) / np.sqrt(len(cohort_data[c]))
                       for c in cohort_order])
    ns     = np.array([len(cohort_data[c]) for c in cohort_order])
    scores = np.array([CIRCUIT_SCORE[c] for c in cohort_order])

    print('\nCohort summary (N, mean sensitization, circuit score):')
    for c, m_, se, n_, s in zip(cohort_order, means, sems, ns, scores):
        print(f'  {c:<16}: N={n_:2d}  dLicking={m_:+.3f}+/-{se:.3f}  circuit={s:+d}')

    # Pearson correlation: circuit score vs observed sensitization (cohort means)
    r_grp, p_grp = stats.pearsonr(scores, means)
    print(f'\nCorrelation (cohort means): r={r_grp:.3f}, p={p_grp:.4f}')

    # One-sample t-tests: each cohort sensitization vs 0
    print('\nOne-sample t-tests (sensitization vs baseline):')
    pvals = []
    for c in cohort_order:
        d = cohort_data[c]
        t, p = stats.ttest_1samp(d, 0)
        pvals.append(p)
        print(f'  {c:<16}: t={t:.2f}, p={p:.4f}, N={len(d)}')

    # Also compute absolute cocaine5 licking per cohort
    cohort_coc5 = {c: [] for c in dreadd_cohorts}
    for m in MT:
        c = mouse_to_cohort[m]
        if c not in cohort_coc5:
            continue
        if 'cocaine5' not in MT[m]:
            continue
        sl = surface_licking_frac(MT[m]['cocaine5']['merged'])
        if not np.isnan(sl):
            cohort_coc5[c].append(sl)

    abs_means = np.array([np.mean(cohort_coc5[c]) if cohort_coc5[c] else np.nan
                          for c in cohort_order])
    abs_sems  = np.array([np.std(cohort_coc5[c]) / np.sqrt(len(cohort_coc5[c]))
                          if len(cohort_coc5[c]) > 1 else np.nan
                          for c in cohort_order])

    # ---------------------------------------------------------------------------
    # Figure — 3 panels
    # ---------------------------------------------------------------------------
    fig = plt.figure(figsize=(200 * mm, 80 * mm))
    gs = gridspec.GridSpec(1, 3, figure=fig,
                           left=0.1, right=0.96, top=0.82, bottom=0.24,
                           wspace=0.52)

    # Sort cohorts by circuit score for display
    sort_idx = np.argsort(scores)
    sorted_cohorts = [cohort_order[i] for i in sort_idx]
    sorted_scores  = scores[sort_idx]
    sorted_means   = means[sort_idx]
    sorted_sems    = sems[sort_idx]
    sorted_pvals   = [pvals[i] for i in sort_idx]

    x_pos = np.arange(len(cohort_order))
    from revision_utils import cohort_labels as cl

    # --- Panel A: Absolute surface licking at cocaine5 per cohort ---
    ax_a = fig.add_subplot(gs[0])
    for xi, i in enumerate(sort_idx):
        c = cohort_order[i]
        col = COHORT_COLOR[c]
        ax_a.bar(xi, abs_means[i], color=col, alpha=0.75, edgecolor='none', width=0.6)
        ax_a.errorbar(xi, abs_means[i], yerr=abs_sems[i],
                      fmt='none', color='k', capsize=3, lw=1)
        # Significance star
        p_ = pvals[i]
        sig = '***' if p_ < 0.001 else '**' if p_ < 0.01 else '*' if p_ < 0.05 else ''
        if sig:
            ax_a.text(xi, abs_means[i] + abs_sems[i] + 0.01, sig,
                      ha='center', fontsize=7)
    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels([cl[c] for c in sorted_cohorts],
                          fontsize=5.5, rotation=40, ha='right')
    ax_a.set_ylabel('Surface licking (cocaine 5)', fontsize=8)
    ax_a.set_title('Peak sensitization per cohort\n(sorted by circuit score)',
                   fontsize=8, fontweight='bold', pad=14)
    ax_a.text(-0.25, 1.15, 'A', transform=ax_a.transAxes,
              fontsize=12, fontweight='bold', va='top')

    # --- Panel B: Sensitization index (Δlicking = cocaine5 – saline3) ---
    ax_b = fig.add_subplot(gs[1])
    for xi, i in enumerate(sort_idx):
        c = cohort_order[i]
        col = COHORT_COLOR[c]
        ax_b.bar(xi, sorted_means[xi], color=col, alpha=0.75, edgecolor='none', width=0.6)
        ax_b.errorbar(xi, sorted_means[xi], yerr=sorted_sems[xi],
                      fmt='none', color='k', capsize=3, lw=1)
        sig = '***' if sorted_pvals[xi] < 0.001 else '**' if sorted_pvals[xi] < 0.01 \
              else '*' if sorted_pvals[xi] < 0.05 else ''
        if sig:
            ypos = sorted_means[xi] + sorted_sems[xi] + 0.01
            ax_b.text(xi, ypos, sig, ha='center', fontsize=7)
    ax_b.axhline(0, color='k', lw=0.7)
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels([cl[c] for c in sorted_cohorts],
                          fontsize=5.5, rotation=40, ha='right')
    ax_b.set_ylabel('Δ Surface licking (C5 − S3)', fontsize=8)
    ax_b.set_title('Sensitization index per cohort\n(cocaine 5 − saline 3)',
                   fontsize=8, fontweight='bold', pad=14)
    ax_b.text(-0.3, 1.15, 'B', transform=ax_b.transAxes,
              fontsize=12, fontweight='bold', va='top')

    # --- Panel C: Circuit score vs observed sensitization (cohort-level linear model) ---
    ax_c = fig.add_subplot(gs[2])
    for xi, (c, sc, m_, se) in enumerate(zip(sorted_cohorts, sorted_scores,
                                              sorted_means, sorted_sems)):
        col = COHORT_COLOR[c]
        ax_c.errorbar(sc, m_, yerr=se, fmt='o', color=col,
                      markersize=7, capsize=3, lw=1.2,
                      markeredgecolor='white', markeredgewidth=0.5, zorder=4)
        ax_c.text(sc + 0.05, m_ + 0.003, cl[c], fontsize=5.5, va='bottom')

    # Linear fit
    if len(scores) >= 3:
        slope, intercept, _, _, _ = stats.linregress(scores, means)
        x_fit = np.linspace(-1.2, 1.2, 100)
        ax_c.plot(x_fit, slope * x_fit + intercept, 'k--', lw=1, alpha=0.5)
    ax_c.axhline(0, color='k', lw=0.5, alpha=0.5)
    ax_c.axvline(0, color='k', lw=0.5, alpha=0.5)

    sig_c = '***' if p_grp < 0.001 else '**' if p_grp < 0.01 else '*' if p_grp < 0.05 else 'n.s.'
    ax_c.set_xlabel('Circuit imbalance score\n(dSPN − iSPN effect)', fontsize=8)
    ax_c.set_ylabel('Δ Surface licking (C5 − S3)', fontsize=8)
    ax_c.set_title(f'Circuit score predicts sensitization\nr={r_grp:.2f}, {sig_c}',
                   fontsize=8, fontweight='bold', pad=14)
    ax_c.set_xlim(-1.5, 1.5)
    ax_c.set_xticks([-1, 0, 1])
    ax_c.set_xticklabels(['iSPN\ndominant', 'Balanced', 'dSPN\ndominant'], fontsize=7)
    ax_c.text(-0.3, 1.15, 'C', transform=ax_c.transAxes,
              fontsize=12, fontweight='bold', va='top')

    save_fig(fig, output_folder, f'Int4_CircuitImbalanceModel_{suffix}')


run_analysis(cohorts_4, '4cohorts')
run_analysis(cohorts_6, '6cohorts')
print('\nDone — circuit imbalance model figures saved.')

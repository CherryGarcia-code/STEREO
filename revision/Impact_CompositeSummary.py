#%% Composite summary figure — key panels from existing analyses
"""
High-impact composite figure combining key results across all analyses:
  A. Surface licking escalation heatmap (all mice × sessions)
  B. Entropy reduction time course
  C. dSPN vs iSPN opto transition difference matrices (side-by-side)
  D. DREADDs grouped V-graphs (4 testable cohorts, compact)
Output: revision/output/
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats

from revision_utils import (
    mm, second, minute, n_behaviors, behaviors, colors, short_labels,
    trials_sal_coc as trials, trial_colors, FPS, ISI, bin_duration, lut,
    cohorts_6 as cohorts, cohort_labels,
    grouping_lut, category_names, category_colors,
    PATHO_LICKING, NATURAL_LICKING, NO_LICKING,
    gcamp_mice, subOptimal_infection,
    dSPN_color, iSPN_color, laser_color,
    load_CMT, load_CTM, flatten_CMT, setup_style, save_fig,
    calc_switch_entropy, transition_matrix
)

setup_style()
output_folder = 'revision/output/'
os.makedirs(output_folder, exist_ok=True)

# ---------------------------------------------------------------------------
# Load behavioral data
# ---------------------------------------------------------------------------
CMT = load_CMT('Dec24')
MT, mouse_to_cohort = flatten_CMT(CMT, cohorts)

valid_mice = sorted([m for m in MT if all(t in MT[m] for t in trials)])
n_mice = len(valid_mice)
n_trials = len(trials)
print(f'Behavioral data: {n_mice} mice')

# Panel A: Surface licking heatmap
surface_frac = np.full((n_mice, n_trials), np.nan)
for m_idx, mouse in enumerate(valid_mice):
    for t_idx, trial in enumerate(trials):
        preds = MT[mouse][trial]['merged']
        grouped = grouping_lut[preds]
        surface_frac[m_idx, t_idx] = np.count_nonzero(grouped == PATHO_LICKING) / preds.size

# Sort mice by cocaine5 surface licking
sort_idx = np.argsort(surface_frac[:, -1])[::-1]
surface_sorted = surface_frac[sort_idx, :]

# Panel B: Switch entropy (session-level, single value per mouse per session)
entropy_vals = np.full((n_mice, n_trials), np.nan)
for m_idx, mouse in enumerate(valid_mice):
    for t_idx, trial in enumerate(trials):
        preds = MT[mouse][trial]['merged']
        ent = calc_switch_entropy(preds, preds.size, preds.size, n_behaviors)
        entropy_vals[m_idx, t_idx] = ent[0] if len(ent) > 0 else np.nan

del MT

# ---------------------------------------------------------------------------
# Panel C: Opto transition difference matrices
# ---------------------------------------------------------------------------
from revision_utils import RNN_offset

# iSPN (new encoding) — uses FIXED timing
CTM_remap = load_CTM('Dec24')
a2a = CTM_remap.get('a2a_opto', {})
iSPN_stim = {}
for trial_name in ['cocaine6laserStim', 'cocaine8laserStim']:
    if trial_name in a2a:
        iSPN_stim.update(a2a[trial_name])

with open('data/opto_alignment_dict.pkl', 'rb') as f:
    opto_dict = pickle.load(f)

iSPN_first_stim = 10 * minute - RNN_offset
iSPN_stim_times = np.arange(iSPN_first_stim, 28 * minute, ISI)

# Filter >50% pre-stim licking
iSPN_mice = {}
for mouse in iSPN_stim:
    preds = iSPN_stim[mouse]['merged']
    pre = preds[:iSPN_first_stim]
    grouped = grouping_lut[pre]
    if np.count_nonzero(grouped == PATHO_LICKING) / pre.size > 0.50:
        iSPN_mice[mouse] = preds
del CTM_remap

def compute_opto_diff_fixed(mice_data, n_beh, stim_times_arr):
    """Compute mean ON-OFF transition diff matrix using fixed stim times."""
    all_on = []
    all_off = []
    for mouse, predictions in mice_data.items():
        on_set = set()
        for stim in stim_times_arr:
            on_set.update(range(int(stim), int(stim) + bin_duration))
        mat_on = np.zeros((n_beh, n_beh))
        mat_off = np.zeros((n_beh, n_beh))
        for i in range(len(predictions) - 1):
            if predictions[i] != predictions[i + 1]:
                if i in on_set:
                    mat_on[predictions[i], predictions[i + 1]] += 1
                else:
                    mat_off[predictions[i], predictions[i + 1]] += 1
        for b in range(n_beh):
            if mat_on[b, :].sum() > 0: mat_on[b, :] /= mat_on[b, :].sum()
            if mat_off[b, :].sum() > 0: mat_off[b, :] /= mat_off[b, :].sum()
        all_on.append(mat_on)
        all_off.append(mat_off)
    if len(all_on) == 0:
        return np.zeros((n_beh, n_beh))
    return np.mean(all_on, axis=0) - np.mean(all_off, axis=0)

def compute_opto_diff_perMouse(mice_data, n_beh, opto_dict_ref):
    """Compute mean ON-OFF transition diff matrix using per-mouse opto_dict offsets."""
    all_on = []
    all_off = []
    for mouse, predictions in mice_data.items():
        if mouse not in opto_dict_ref:
            continue
        offset = opto_dict_ref[mouse][1]
        first_stim = 3 * minute - int((offset / 1000.0) * FPS)
        stim_times = np.arange(first_stim, 14 * minute, ISI)
        on_set = set()
        for stim in stim_times:
            on_set.update(range(int(stim), int(stim) + bin_duration))

        mat_on = np.zeros((n_beh, n_beh))
        mat_off = np.zeros((n_beh, n_beh))
        for i in range(len(predictions) - 1):
            if predictions[i] != predictions[i + 1]:
                if i in on_set:
                    mat_on[predictions[i], predictions[i + 1]] += 1
                else:
                    mat_off[predictions[i], predictions[i + 1]] += 1
        # Normalize
        for b in range(n_beh):
            if mat_on[b, :].sum() > 0: mat_on[b, :] /= mat_on[b, :].sum()
            if mat_off[b, :].sum() > 0: mat_off[b, :] /= mat_off[b, :].sum()
        all_on.append(mat_on)
        all_off.append(mat_off)
    if len(all_on) == 0:
        return np.zeros((n_beh, n_beh))
    return np.mean(all_on, axis=0) - np.mean(all_off, axis=0)

iSPN_diff = compute_opto_diff_fixed(iSPN_mice, n_behaviors, iSPN_stim_times)
print(f'iSPN mice: {len(iSPN_mice)}')

# dSPN (Dec24 = canonical new encoding)
CTM_raw = load_CTM('Dec24')
dSPN_stim = CTM_raw['drd1_opto']['AloneStim']
del CTM_raw

dSPN_data = {m: dSPN_stim[m]['merged'] for m in dSPN_stim if m in opto_dict}
dSPN_diff = compute_opto_diff_perMouse(dSPN_data, n_behaviors, opto_dict)

# ---------------------------------------------------------------------------
# Panel D: DREADDs grouped V-graphs (4 testable cohorts)
# ---------------------------------------------------------------------------
CMT2 = load_CMT('Dec24')
dreadd_cohorts = ['drd1_hm4di', 'drd1_hm3dq', 'a2a_hm4di', 'a2a_hm3dq']
dreadd_labels_short = {
    'drd1_hm4di': r'Drd1$_{hm4Di}$', 'drd1_hm3dq': r'Drd1$_{hm3Dq}$',
    'a2a_hm4di': r'A2a$_{hm4Di}$', 'a2a_hm3dq': r'A2a$_{hm3Dq}$',
}
CNO2cocaineGap = {
    'drd1_hm4di': {'c512m3':30,'c512m4':30,'c512m7':30,'c526m2':32,'c526m3':35,'c528m5':31,'c528m10':38,'c548m1':31},
    'drd1_hm3dq': {'c514Bm2':35,'c514Bm8':35,'c514m1':35,'c514m3':38,'c514m5':32},
    'a2a_hm4di': {'cA154m4':35,'cA154m6':36,'cA156m1':35,'cA156m6':35,'cA156m7':35,'cA156m8':34},
    'a2a_hm3dq': {'cA156m2':33,'cA156m5':34,'cA158m2':30,'cA158m3':30,'cA158m4':30,
                   'cA184m4':33,'cA184m7':33,'cA242m5':30,'cA242m6':30,'cA242m8':30},
}
possible_trials = {
    'cocaineOnly': [f'cocaine{i}' for i in range(1, 11)],
    'cocaineCNO': ['cocaine6afterCNO','cocaine7afterCNO','cocaine8afterCNO','cocaine9afterCNO'],
}
CNO_MAX_TIME = 50
UNDEFINED = 1

dreadd_results = {}
for cohort in dreadd_cohorts:
    MT_c = CMT2[cohort]
    surface_pre = []
    surface_cno = []
    surface_post = []
    for m in MT_c:
        if m in gcamp_mice or m in subOptimal_infection:
            continue
        t_list = list(MT_c[m].keys())
        for t_idx in range(1, len(t_list) - 1):
            if (t_list[t_idx] in possible_trials['cocaineCNO'] and
                t_list[t_idx + 1] in possible_trials['cocaineOnly']):
                if t_list[t_idx - 1] in possible_trials['cocaineOnly']:
                    pre_tr = t_list[t_idx - 1]
                elif t_idx >= 2 and 'CNOonly' in t_list[t_idx - 1] and t_list[t_idx - 2] in possible_trials['cocaineOnly']:
                    pre_tr = t_list[t_idx - 2]
                else:
                    continue
                cutoff = (CNO_MAX_TIME - CNO2cocaineGap.get(cohort, {}).get(m, 30)) * minute
                for key, trial_name, arr in [('pre', pre_tr, surface_pre),
                                              ('cno', t_list[t_idx], surface_cno),
                                              ('post', t_list[t_idx + 1], surface_post)]:
                    preds = MT_c[m][trial_name]['merged'][:cutoff]
                    preds = preds[preds != UNDEFINED]
                    grouped = grouping_lut[preds]
                    arr.append(np.count_nonzero(grouped == PATHO_LICKING) / preds.size if preds.size > 0 else np.nan)
                break
    dreadd_results[cohort] = {
        'pre': np.array(surface_pre), 'cno': np.array(surface_cno),
        'post': np.array(surface_post), 'n': len(surface_pre)
    }
del CMT2

# ===========================================================================
# BUILD COMPOSITE FIGURE
# ===========================================================================
fig = plt.figure(figsize=(180 * mm, 170 * mm))
outer = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35,
                          height_ratios=[1, 0.9, 0.9])

# --- Panel A: Surface licking heatmap ---
ax_a = fig.add_subplot(outer[0, 0])
im = ax_a.imshow(surface_sorted, cmap='Reds', aspect='auto', vmin=0, vmax=0.7)
ax_a.set_xticks(range(n_trials))
ax_a.set_xticklabels(['S1','S2','S3','C1','C2','C3','C4','C5'], fontsize=6)
ax_a.set_ylabel(f'Mice (n={n_mice})', fontsize=8)
ax_a.set_title('Surface licking per mouse', fontsize=8, fontweight='bold', pad=12)
fig.colorbar(im, ax=ax_a, shrink=0.7, aspect=15, label='Fraction')
ax_a.text(-0.15, 1.15, 'A', transform=ax_a.transAxes, fontsize=12, fontweight='bold', va='top')

# --- Panel B: Entropy time course ---
ax_b = fig.add_subplot(outer[0, 1])
mean_ent = np.nanmean(entropy_vals, axis=0)
sem_ent = np.nanstd(entropy_vals, axis=0, ddof=0) / np.sqrt(n_mice)
x_ent = np.arange(n_trials)
ax_b.fill_between(x_ent, mean_ent - sem_ent, mean_ent + sem_ent, color='#1f4e79', alpha=0.2)
ax_b.plot(x_ent, mean_ent, color='#1f4e79', lw=2, marker='o', markersize=4)
for i in range(n_trials):
    ax_b.scatter(i, mean_ent[i], color=trial_colors[i], s=35, zorder=5, edgecolors='none')
ax_b.set_xticks(x_ent)
ax_b.set_xticklabels(['S1','S2','S3','C1','C2','C3','C4','C5'], fontsize=6)
ax_b.set_ylabel('Switch entropy', fontsize=8)
ax_b.set_title('Behavioral entropy across sessions', fontsize=8, fontweight='bold', pad=12)
ax_b.axvspan(-0.5, 2.5, color='gray', alpha=0.04)
ax_b.text(-0.15, 1.15, 'B', transform=ax_b.transAxes, fontsize=12, fontweight='bold', va='top')

# --- Panel C: Transition diff matrices (iSPN left, dSPN right) ---
gs_c = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1, :], wspace=0.4)

# iSPN diff
ax_c1 = fig.add_subplot(gs_c[0])
vmax_i = np.nanmax(np.abs(iSPN_diff)) if np.any(iSPN_diff != 0) else 0.1
im_c1 = ax_c1.imshow(iSPN_diff, cmap='RdBu_r', vmin=-vmax_i, vmax=vmax_i, aspect='equal')
ax_c1.set_xticks(range(n_behaviors))
ax_c1.set_xticklabels(short_labels, fontsize=4.5, rotation=45, ha='right')
ax_c1.set_yticks(range(n_behaviors))
ax_c1.set_yticklabels(short_labels, fontsize=4.5)
ax_c1.set_title(f'iSPN ON$-$OFF (N={len(iSPN_mice)})', fontsize=7, fontweight='bold', color=iSPN_color, pad=12)
fig.colorbar(im_c1, ax=ax_c1, shrink=0.7, aspect=15).ax.tick_params(labelsize=5)
ax_c1.set_xlabel('To', fontsize=6)
ax_c1.set_ylabel('From', fontsize=6)
ax_c1.text(-0.15, 1.15, 'C', transform=ax_c1.transAxes, fontsize=12, fontweight='bold', va='top')

# dSPN diff
ax_c2 = fig.add_subplot(gs_c[1])
vmax_d = np.nanmax(np.abs(dSPN_diff)) if np.any(dSPN_diff != 0) else 0.1
im_c2 = ax_c2.imshow(dSPN_diff, cmap='RdBu_r', vmin=-vmax_d, vmax=vmax_d, aspect='equal')
ax_c2.set_xticks(range(n_behaviors))
ax_c2.set_xticklabels(short_labels, fontsize=4.5, rotation=45, ha='right')
ax_c2.set_yticks(range(n_behaviors))
ax_c2.set_yticklabels(short_labels, fontsize=4.5)
ax_c2.set_title(f'dSPN ON$-$OFF (N={len(dSPN_data)})', fontsize=7, fontweight='bold', color=dSPN_color, pad=12)
fig.colorbar(im_c2, ax=ax_c2, shrink=0.7, aspect=15).ax.tick_params(labelsize=5)
ax_c2.set_xlabel('To', fontsize=6)
ax_c2.set_ylabel('From', fontsize=6)
ax_c2.text(-0.15, 1.15, 'D', transform=ax_c2.transAxes, fontsize=12, fontweight='bold', va='top')

# --- Panel D: DREADDs mini V-graphs ---
gs_d = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[2, :], wspace=0.3)

dreadd_colors = {
    'drd1_hm4di': dSPN_color, 'drd1_hm3dq': '#e57373',
    'a2a_hm4di': iSPN_color, 'a2a_hm3dq': '#b35a5a',
}

for c_idx, cohort in enumerate(dreadd_cohorts):
    ax = fig.add_subplot(gs_d[c_idx])
    r = dreadd_results[cohort]
    x_v = [0, 1, 2]

    # Individual lines
    for i in range(r['n']):
        ax.plot(x_v, [r['pre'][i], r['cno'][i], r['post'][i]],
                color='gray', alpha=0.25, lw=0.4)

    # Mean + SEM
    means = [np.nanmean(r['pre']), np.nanmean(r['cno']), np.nanmean(r['post'])]
    sems = [np.nanstd(r['pre'], ddof=0)/np.sqrt(r['n']),
            np.nanstd(r['cno'], ddof=0)/np.sqrt(r['n']),
            np.nanstd(r['post'], ddof=0)/np.sqrt(r['n'])]
    ax.errorbar(x_v, means, yerr=sems, color=dreadd_colors[cohort],
                marker='o', markersize=4, lw=2, capsize=3)

    # Stats
    t_stat, p_val = stats.ttest_rel(r['pre'], r['cno'])
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
    ymax = max(means) + max(sems) + 0.03
    ax.plot([0, 1], [ymax, ymax], 'k-', lw=0.5)
    ax.text(0.5, ymax + 0.005, sig, ha='center', fontsize=5)

    ax.set_xticks(x_v)
    ax.set_xticklabels(['Pre', 'CNO', 'Post'], fontsize=5)
    ax.set_title(f'{dreadd_labels_short[cohort]}\n(n={r["n"]})', fontsize=6, fontweight='bold',
                 color=dreadd_colors[cohort], pad=12)
    if c_idx == 0:
        ax.set_ylabel('Surface licking', fontsize=7)
    ax.tick_params(labelsize=5)

    panel = chr(ord('E') + c_idx)
    ax.text(-0.2, 1.15, panel, transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')

save_fig(fig, output_folder, 'Impact_CompositeSummary')
print(f'\nDone — Composite summary figure')

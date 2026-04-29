#%% SuppFig: Entropy During Opto + DREADDs Manipulations
"""
Addresses R4-Fig6 (temporal dynamics), R4-Fig3JK (behavioral quantification).

Panels:
  A-B: Switch entropy + transition rate: laser-ON vs laser-OFF (iSPN opto)
  C-D: Switch entropy + transition rate: Pre/During/Post CNO (DREADDs, per cohort)

Uses CTM_Aug24.pkl (opto), CMT_Aug24.pkl (DREADDs).
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
    mm, second, minute, n_behaviors, behaviors, lut,
    ISI, bin_duration, RNN_offset, FPS,
    load_CTM, load_CMT, flatten_CMT, setup_style, save_fig,
    calc_switch_entropy, transition_rate,
    gcamp_mice, subOptimal_infection,
    PATHO_LICKING, grouping_lut, category_colors, category_names
)

setup_style()
output_folder = 'revision/output/'
os.makedirs(output_folder, exist_ok=True)

# ==========================================================================
# PART 1: iSPN Opto — entropy ON vs OFF
# ==========================================================================
print('=== iSPN Opto ===')
CTM = load_CTM('Dec24')
stim_days_raw = CTM['a2a_opto']
stim_days = {key: stim_days_raw[key] for key in ['cocaine6laserStim', 'cocaine8laserStim']
             if key in stim_days_raw}
del CTM

num_of_stims_cocaine = 14
first_stim = 10 * minute - RNN_offset

# Select mice with >=50% licking pre-stim
selected = {}
for mouse in set().union(*[stim_days[d].keys() for d in stim_days]):
    for day in ['cocaine6laserStim', 'cocaine8laserStim']:
        if day in stim_days and mouse in stim_days[day]:
            preds = stim_days[day][mouse]['merged']
            pre = preds[:first_stim]
            if np.mean((pre == 2) | (pre == 3)) >= 0.5:
                selected[mouse] = preds
                break

mice_list = sorted(selected.keys())
n_mice_opto = len(mice_list)
print(f'Selected {n_mice_opto} mice')

stim_times = np.arange(first_stim, 28 * minute, ISI)

# Compute metrics for ON and OFF epochs
entropy_on = np.full(n_mice_opto, np.nan)
entropy_off = np.full(n_mice_opto, np.nan)
trans_rate_on = np.full(n_mice_opto, np.nan)
trans_rate_off = np.full(n_mice_opto, np.nan)

for m_idx, mouse in enumerate(mice_list):
    predictions = selected[mouse]
    on_frames = []
    off_frames = []

    laser_on_set = set()
    for stim in stim_times[:num_of_stims_cocaine]:
        laser_on_set.update(range(int(stim), int(stim) + bin_duration))

    # Collect ON and OFF segments within the stimulation period
    for s_idx, stim in enumerate(stim_times[:num_of_stims_cocaine]):
        s = int(stim)
        on_frames.extend(predictions[s : s + bin_duration].tolist())
        # OFF = post-stim epoch of same duration
        off_start = s + bin_duration
        off_end = off_start + bin_duration
        if off_end <= len(predictions):
            off_frames.extend(predictions[off_start : off_end].tolist())

    on_arr = np.array(on_frames, dtype=int)
    off_arr = np.array(off_frames, dtype=int)

    if len(on_arr) > 60:
        sw = calc_switch_entropy(on_arr, len(on_arr), len(on_arr), n_behaviors)
        entropy_on[m_idx] = sw[0] if len(sw) > 0 else np.nan
        trans_rate_on[m_idx] = transition_rate(on_arr, n_behaviors)
    if len(off_arr) > 60:
        sw = calc_switch_entropy(off_arr, len(off_arr), len(off_arr), n_behaviors)
        entropy_off[m_idx] = sw[0] if len(sw) > 0 else np.nan
        trans_rate_off[m_idx] = transition_rate(off_arr, n_behaviors)

print(f'  Entropy — OFF: {np.nanmean(entropy_off):.3f} ON: {np.nanmean(entropy_on):.3f}')
t, p = stats.ttest_rel(entropy_off[~np.isnan(entropy_on)], entropy_on[~np.isnan(entropy_on)])
print(f'  Paired t-test: t={t:.2f} p={p:.4f}')
print(f'  Trans rate — OFF: {np.nanmean(trans_rate_off):.1f} ON: {np.nanmean(trans_rate_on):.1f}')

# ==========================================================================
# PART 2: DREADDs — entropy Pre/During/Post CNO
# ==========================================================================
print('\n=== DREADDs ===')
CMT = load_CMT('Dec24')

dreadd_cohorts = ['drd1_hm4di', 'drd1_hm3dq', 'controls', 'a2a_hm4di', 'a2a_hm3dq']
dreadd_labels = {
    'drd1_hm4di': r'Drd1$_{hm4Di}$', 'drd1_hm3dq': r'Drd1$_{hm3Dq}$',
    'controls': 'Controls', 'a2a_hm4di': r'A2a$_{hm4Di}$', 'a2a_hm3dq': r'A2a$_{hm3Dq}$',
}
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

dreadd_entropy = {}
dreadd_trans_rate = {}

for cohort in dreadd_cohorts:
    MT_c = CMT[cohort]
    sandwich = {}
    for m in MT_c:
        if m in gcamp_mice or m in subOptimal_infection:
            continue
        trials = list(MT_c[m].keys())
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
                    'pre': MT_c[m][pre_tr]['merged'],
                    'CNO': MT_c[m][trials[t_idx]]['merged'],
                    'post': MT_c[m][trials[t_idx + 1]]['merged'],
                }
                break

    mice = sorted(sandwich.keys())
    n = len(mice)
    if n == 0:
        continue

    entropy_vals = np.full((n, 3), np.nan)
    tr_vals = np.full((n, 3), np.nan)

    for m_idx, m in enumerate(mice):
        cutoff_min = CNO2cocaineGap.get(cohort, {}).get(m, 30)
        cutoff = (CNO_MAX_TIME - cutoff_min) * minute
        for d_idx, key in enumerate(['pre', 'CNO', 'post']):
            preds = sandwich[m][key][:cutoff]
            if len(preds) > 60:
                sw = calc_switch_entropy(preds, len(preds), len(preds), n_behaviors)
                entropy_vals[m_idx, d_idx] = sw[0] if len(sw) > 0 else np.nan
                tr_vals[m_idx, d_idx] = transition_rate(preds, n_behaviors)

    dreadd_entropy[cohort] = entropy_vals
    dreadd_trans_rate[cohort] = tr_vals

    t1, p1 = stats.ttest_rel(entropy_vals[:, 0], entropy_vals[:, 1])
    print(f'  {cohort} (N={n}): entropy pre={np.nanmean(entropy_vals[:,0]):.3f} '
          f'CNO={np.nanmean(entropy_vals[:,1]):.3f} post={np.nanmean(entropy_vals[:,2]):.3f} '
          f'pre-CNO t={t1:.2f} p={p1:.4f}')

del CMT

# ===========================================================================
# BUILD THE FIGURE
# ===========================================================================
print('\nGenerating figure...')
fig = plt.figure(figsize=(180 * mm, 130 * mm))
outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[0.8, 1.2], hspace=0.45)

# --- Row 1: Opto (A: entropy, B: transition rate) ---
gs_top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.4)

for col_idx, (on_vals, off_vals, ylabel, title, panel) in enumerate([
    (entropy_on, entropy_off, 'Switch entropy (bits)', 'Entropy: laser ON vs OFF', 'A'),
    (trans_rate_on, trans_rate_off, 'Switches/min', 'Transition rate: laser ON vs OFF', 'B'),
]):
    ax = fig.add_subplot(gs_top[col_idx])
    for m in range(n_mice_opto):
        ax.plot([0, 1], [off_vals[m], on_vals[m]], color='gray', alpha=0.3, lw=0.5)
    ax.errorbar([0], [np.nanmean(off_vals)], yerr=[np.nanstd(off_vals, ddof=0)/np.sqrt(n_mice_opto)],
                color='#808080', marker='o', markersize=6, capsize=4, lw=2)
    ax.errorbar([1], [np.nanmean(on_vals)], yerr=[np.nanstd(on_vals, ddof=0)/np.sqrt(n_mice_opto)],
                color='#B2E2F6', marker='o', markersize=6, capsize=4, lw=2, markeredgecolor='#3399cc')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Laser OFF', 'Laser ON'], fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(f'iSPN opto (N={n_mice_opto}): {title}', fontsize=7, fontweight='bold', pad=12)

    valid = ~(np.isnan(on_vals) | np.isnan(off_vals))
    if valid.sum() >= 3:
        _, p = stats.ttest_rel(off_vals[valid], on_vals[valid])
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        ymax = max(np.nanmax(on_vals), np.nanmax(off_vals)) * 1.05
        ax.plot([0, 1], [ymax, ymax], 'k-', lw=0.8)
        ax.text(0.5, ymax * 1.01, sig, ha='center', fontsize=7)

    ax.text(-0.15, 1.15, panel, transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')

# --- Row 2: DREADDs (C: entropy, D: transition rate per cohort) ---
gs_bot = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.4)

cohort_colors = ['#d73027', '#ff6347', '#808080', '#6a5acd', '#c4a7e7']

for col_idx, (data_dict, ylabel, title, panel) in enumerate([
    (dreadd_entropy, 'Switch entropy (bits)', 'Entropy: Pre/CNO/Post', 'C'),
    (dreadd_trans_rate, 'Switches/min', 'Transition rate: Pre/CNO/Post', 'D'),
]):
    ax = fig.add_subplot(gs_bot[col_idx])
    x_offset = 0
    xtick_positions = []
    xtick_labels = []

    for c_idx, cohort in enumerate(dreadd_cohorts):
        if cohort not in data_dict:
            continue
        vals = data_dict[cohort]
        n = vals.shape[0]
        x_pos = np.array([0, 1, 2]) + x_offset

        # Individual lines
        for m in range(n):
            ax.plot(x_pos, vals[m, :], color='gray', alpha=0.15, lw=0.3)

        mean_v = np.nanmean(vals, axis=0)
        sem_v = np.nanstd(vals, axis=0, ddof=0) / np.sqrt(n)
        ax.errorbar(x_pos, mean_v, yerr=sem_v, color=cohort_colors[c_idx],
                    lw=1.5, capsize=3, capthick=1, marker='o', markersize=3, zorder=5)

        xtick_positions.extend(x_pos.tolist())
        xtick_labels.extend(['-', '+', '-'])

        # Label cohort
        ax.text(x_pos[1], np.nanmin(vals) * 0.85, dreadd_labels[cohort],
                ha='center', fontsize=4.5, fontweight='bold', color=cohort_colors[c_idx])

        x_offset += 4  # spacing between cohorts

    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, fontsize=5)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=7, fontweight='bold', pad=12)
    ax.tick_params(labelsize=6)
    ax.text(-0.1, 1.15, panel, transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')

save_fig(fig, output_folder, 'SuppFig7_Entropy_Manipulations')
print('\nDone.')

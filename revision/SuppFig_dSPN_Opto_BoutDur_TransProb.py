#%% SuppFig: dSPN Opto Bout Duration + Transition Probabilities
"""
Addresses R4-Fig5 (complement dSPN conclusions with bout duration/transitions).

Panels:
  A-C: Transition matrices — Laser OFF, Laser ON, ON−OFF difference
  D-H: Bout duration CDFs laser-ON vs laser-OFF for key behaviors

Uses CTM_Aug24.pkl, drd1_opto cohort + opto_alignment_dict.pkl.
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
import helper_functions as hf

from revision_utils import (
    mm, second, minute, FPS, ISI, bin_duration,
    n_behaviors, behaviors, colors, short_labels,
    load_CTM, setup_style, save_fig
)

setup_style()
output_folder = 'revision/output/'
os.makedirs(output_folder, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
CTM = load_CTM('Dec24')  # Dec24 uses canonical encoding
stim_day = CTM['drd1_opto']['AloneStim']
del CTM

with open('data/opto_alignment_dict.pkl', 'rb') as f:
    opto_dict = pickle.load(f)

# Canonical encoding (Dec24 = new encoding)
n_beh = n_behaviors
num_of_stims = 9

n_mice = len(stim_day)
print(f'dSPN opto data loaded: {n_mice} mice')

# ---------------------------------------------------------------------------
# Compute transition matrices and bout durations
# ---------------------------------------------------------------------------
trans_on_all = []
trans_off_all = []
bout_data = {b: {'on': [], 'off': []} for b in range(n_beh)}
mice_names = []

for mouse in stim_day:
    predictions = stim_day[mouse]['merged']
    offset = opto_dict[mouse][1]
    first_stim = 3 * minute - int((offset / 1000.0) * FPS)
    stim_times = np.arange(first_stim, 14 * minute, ISI)
    mice_names.append(mouse)

    # Build laser-on frame set
    laser_on_set = set()
    for stim in stim_times:
        laser_on_set.update(range(int(stim), int(stim) + bin_duration))

    # Transition matrices
    trans_on = np.zeros((n_beh, n_beh))
    trans_off = np.zeros((n_beh, n_beh))
    for i in range(len(predictions) - 1):
        if predictions[i] != predictions[i + 1]:
            if i in laser_on_set:
                trans_on[predictions[i], predictions[i + 1]] += 1
            else:
                trans_off[predictions[i], predictions[i + 1]] += 1

    for b in range(n_beh):
        if trans_on[b, :].sum() > 0:
            trans_on[b, :] /= trans_on[b, :].sum()
        if trans_off[b, :].sum() > 0:
            trans_off[b, :] /= trans_off[b, :].sum()

    trans_on_all.append(trans_on)
    trans_off_all.append(trans_off)

    # Bout durations
    laser_on_frames = set()
    for stim in stim_times:
        laser_on_frames.update(range(int(stim), int(stim) + bin_duration))

    for b in range(n_beh):
        bouts = hf.segment_bouts(predictions, b, 1)
        for bout_indices in bouts['indices']:
            overlap = sum(1 for idx in bout_indices if idx in laser_on_frames)
            if overlap / len(bout_indices) > 0.5:
                bout_data[b]['on'].append(len(bout_indices))
            else:
                bout_data[b]['off'].append(len(bout_indices))

trans_on_all = np.array(trans_on_all)
trans_off_all = np.array(trans_off_all)
n_m = len(mice_names)

mean_on = np.nanmean(trans_on_all, axis=0)
mean_off = np.nanmean(trans_off_all, axis=0)
diff_matrix = mean_on - mean_off

# Stats: paired t-test per cell
p_matrix = np.ones((n_beh, n_beh))
for i in range(n_beh):
    for j in range(n_beh):
        vals_on = trans_on_all[:, i, j]
        vals_off = trans_off_all[:, i, j]
        valid = ~(np.isnan(vals_on) | np.isnan(vals_off))
        if valid.sum() >= 3:
            _, p_matrix[i, j] = stats.ttest_rel(vals_on[valid], vals_off[valid])

print(f'\nSignificant transition changes (p<0.05): {np.sum(p_matrix < 0.05)}')
sig_cells = np.argwhere(p_matrix < 0.05)
for cell in sig_cells:
    i, j = cell
    print(f'  {short_labels[i]} -> {short_labels[j]}: '
          f'OFF={mean_off[i,j]:.3f} ON={mean_on[i,j]:.3f} diff={diff_matrix[i,j]:+.3f} p={p_matrix[i,j]:.4f}')

# Bout stats
print('\n--- Bout duration stats ---')
vdb_behaviors = [4, 5, 3, 2, 7]  # Grooming, Body lick, Wall lick, Floor lick, Locomotion
for bidx in vdb_behaviors:
    on_durs = np.array(bout_data[bidx]['on']) / FPS
    off_durs = np.array(bout_data[bidx]['off']) / FPS
    if len(on_durs) > 2 and len(off_durs) > 2:
        ks_stat, ks_p = stats.ks_2samp(on_durs, off_durs)
        print(f'  {behaviors[bidx]}: ON median={np.median(on_durs):.1f}s (n={len(on_durs)}), '
              f'OFF median={np.median(off_durs):.1f}s (n={len(off_durs)}), KS={ks_stat:.3f} p={ks_p:.4f}')

# ===========================================================================
# BUILD THE FIGURE
# ===========================================================================
print('\nGenerating figure...')
fig = plt.figure(figsize=(180 * mm, 120 * mm))
outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 0.8], hspace=0.45)

# --- Row 1: Transition matrices ---
gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], wspace=0.4)
vmax_abs = max(np.nanmax(mean_off), np.nanmax(mean_on))

for panel_idx, (mat, cmap, title, panel) in enumerate([
    (mean_off, 'Purples', 'Laser OFF', 'A'),
    (mean_on, 'Purples', 'Laser ON (dSPN)', 'B'),
    (diff_matrix, 'RdBu_r', 'ON $-$ OFF', 'C'),
]):
    ax = fig.add_subplot(gs_top[panel_idx])
    if panel_idx < 2:
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax_abs, aspect='equal')
    else:
        vmax_diff = np.nanmax(np.abs(diff_matrix))
        im = ax.imshow(mat, cmap=cmap, vmin=-vmax_diff, vmax=vmax_diff, aspect='equal')
        for i in range(n_beh):
            for j in range(n_beh):
                if p_matrix[i, j] < 0.05:
                    ax.text(j, i, '*', ha='center', va='center', fontsize=7, fontweight='bold')

    ax.set_xticks(range(n_beh))
    ax.set_xticklabels(short_labels, fontsize=5, rotation=45, ha='right')
    ax.set_yticks(range(n_beh))
    ax.set_yticklabels(short_labels, fontsize=5)
    ax.set_xlabel('To', fontsize=7)
    ax.set_title(title, fontsize=8, fontweight='bold', pad=12)
    fig.colorbar(im, ax=ax, shrink=0.7, aspect=15).ax.tick_params(labelsize=5)
    if panel_idx == 0:
        ax.set_ylabel('From', fontsize=7)
    ax.text(-0.15, 1.15, panel, transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')

# --- Row 2: Bout duration CDFs ---
gs_bot = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[1], wspace=0.35)

for b_idx, bidx in enumerate(vdb_behaviors):
    ax = fig.add_subplot(gs_bot[b_idx])
    on_durs = np.array(bout_data[bidx]['on']) / FPS
    off_durs = np.array(bout_data[bidx]['off']) / FPS

    if len(off_durs) > 0:
        sorted_off = np.sort(off_durs)
        cdf_off = np.arange(1, len(sorted_off) + 1) / len(sorted_off)
        ax.step(sorted_off, cdf_off, color='gray', lw=1.2, label=f'OFF (n={len(off_durs)})')
    if len(on_durs) > 0:
        sorted_on = np.sort(on_durs)
        cdf_on = np.arange(1, len(sorted_on) + 1) / len(sorted_on)
        ax.step(sorted_on, cdf_on, color='#730AFF', lw=1.2, label=f'ON (n={len(on_durs)})')

    ax.set_title(behaviors[bidx], fontsize=6, fontweight='bold', color=colors[bidx], pad=12)
    ax.set_xlabel('Duration (s)', fontsize=5)
    ax.legend(fontsize=4, loc='lower right')
    ax.tick_params(labelsize=5)
    ax.set_xlim(0, 30)
    if b_idx == 0:
        ax.set_ylabel('CDF', fontsize=6)

    if len(on_durs) > 2 and len(off_durs) > 2:
        ks_stat, ks_p = stats.ks_2samp(on_durs, off_durs)
        sig = '***' if ks_p < 0.001 else '**' if ks_p < 0.01 else '*' if ks_p < 0.05 else 'n.s.'
        ax.text(0.95, 0.15, f'KS {sig}\np={ks_p:.3f}', transform=ax.transAxes,
                fontsize=4, ha='right', va='bottom')

    panel = chr(ord('D') + b_idx)
    ax.text(-0.15, 1.15, panel, transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')

fig.suptitle(f'dSPN Opto: Transition Probabilities & Bout Durations (N={n_m})',
             fontsize=9, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95])
save_fig(fig, output_folder, 'SuppFig4_dSPN_Opto_BoutDur_TransProb')
print(f'\nDone — N={n_m} mice')

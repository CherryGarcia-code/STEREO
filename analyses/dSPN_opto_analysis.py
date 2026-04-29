#%% dSPN (Drd1) Optogenetic Analysis: Bout Duration & Transition Probabilities
# ============================================================================
# NOTE: This script requires May25/Data/CTM_May25.pkl and
#       May25/Data/opto_alignment_dict.pkl which are NOT present in this
#       workspace. Create the May25/Data/ directory and place the files there,
#       then run this script.
# ============================================================================

import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import bz2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import scipy.stats as stats
import os
import helper_functions as hf

root_folder = 'May25'
folder = 'data/'
mm = 1 / 25.4

# Check data availability
if not os.path.exists(folder + 'CTM_May25.pkl'):
    raise FileNotFoundError(
        f'{folder}CTM_May25.pkl not found. '
        'Copy the Drd1 opto data files into May25/Data/ and rerun.')
if not os.path.exists(folder + 'opto_alignment_dict.pkl'):
    raise FileNotFoundError(
        f'{folder}opto_alignment_dict.pkl not found. '
        'Copy the alignment dict into May25/Data/ and rerun.')

# --- Load data ---
ifile = bz2.BZ2File(folder + 'CTM_May25.pkl', 'rb')
stim_day = pickle.load(ifile)['drd1_opto']['AloneStim']
ifile.close()

temp_file = open(folder + 'opto_alignment_dict.pkl', 'rb')
opto_dict = pickle.load(temp_file)
temp_file.close()
print(f'***Drd1 opto data loaded: {len(stim_day)} mice***')

# --- Constants ---
behaviors = ['Grooming', 'Body licking', 'Wall licking', 'Floor licking',
             'Rearing', 'Back to camera', 'Stationary', 'Locomotion', 'Jump']
colors = ['#c21296', '#06d6a0', '#ee476f', '#1189b1', '#ffd169',
          '#783f04', '#596163', '#969595', '#b4a7d6']
FPS = 15
second = 15
minute = 60 * second
ISI = 1 * minute + 20 * second
bin_duration = 20 * second
num_of_behaviors = len(behaviors)
num_of_stims = 9
num_of_mice = len(stim_day)
VDB = 0
nVDB = 1

output_folder = root_folder + '/Figures/dSPN_opto_analysis/'
os.makedirs(output_folder, exist_ok=True)

#%% ========== ANALYSIS 3a: Transition Probability Matrices ==========
trans_on_all = []
trans_off_all = []
mice_names = []

for mouse in stim_day:
    predictions = stim_day[mouse]['merged']
    offset = opto_dict[mouse][1]
    first_stim = 3 * minute - int((offset / 1000.0) * 15)
    stim_times = np.arange(first_stim, 14 * minute, ISI)
    mice_names.append(mouse)

    # Build laser-on frame set
    laser_on_set = set()
    for stim in stim_times:
        laser_on_set.update(range(int(stim), int(stim) + bin_duration))

    trans_on = np.zeros((num_of_behaviors, num_of_behaviors))
    trans_off = np.zeros((num_of_behaviors, num_of_behaviors))

    for i in range(len(predictions) - 1):
        if predictions[i] != predictions[i + 1]:
            if i in laser_on_set:
                trans_on[predictions[i], predictions[i + 1]] += 1
            else:
                trans_off[predictions[i], predictions[i + 1]] += 1

    # Normalize
    for b in range(num_of_behaviors):
        if trans_on[b, :].sum() > 0:
            trans_on[b, :] /= trans_on[b, :].sum()
        if trans_off[b, :].sum() > 0:
            trans_off[b, :] /= trans_off[b, :].sum()

    trans_on_all.append(trans_on)
    trans_off_all.append(trans_off)

trans_on_all = np.array(trans_on_all)
trans_off_all = np.array(trans_off_all)
n_mice = len(mice_names)

mean_on = np.nanmean(trans_on_all, axis=0)
mean_off = np.nanmean(trans_off_all, axis=0)
diff_matrix = mean_on - mean_off

p_matrix = np.ones((num_of_behaviors, num_of_behaviors))
for i in range(num_of_behaviors):
    for j in range(num_of_behaviors):
        vals_on = trans_on_all[:, i, j]
        vals_off = trans_off_all[:, i, j]
        valid = ~(np.isnan(vals_on) | np.isnan(vals_off))
        if valid.sum() >= 3:
            _, p_matrix[i, j] = stats.ttest_rel(vals_on[valid], vals_off[valid])

# --- Figure: Transition matrices ---
short_labels = ['Grm', 'BdL', 'WlL', 'FlL', 'Rer', 'BtC', 'Stn', 'Loc', 'Jmp']
fig, axes = plt.subplots(1, 3, figsize=(180 * mm, 55 * mm))
vmax_abs = max(np.nanmax(mean_off), np.nanmax(mean_on))

im0 = axes[0].imshow(mean_off, cmap='Purples', vmin=0, vmax=vmax_abs, aspect='equal')
axes[0].set_title('Laser OFF', fontsize=8, fontweight='bold')
im1 = axes[1].imshow(mean_on, cmap='Purples', vmin=0, vmax=vmax_abs, aspect='equal')
axes[1].set_title('Laser ON (dSPN stim)', fontsize=8, fontweight='bold')
vmax_diff = np.nanmax(np.abs(diff_matrix))
im2 = axes[2].imshow(diff_matrix, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff, aspect='equal')
axes[2].set_title('ON $-$ OFF', fontsize=8, fontweight='bold')

for ax_i, im_i in zip(axes, [im0, im1, im2]):
    ax_i.set_xticks(range(num_of_behaviors))
    ax_i.set_xticklabels(short_labels, fontsize=5, rotation=45, ha='right')
    ax_i.set_yticks(range(num_of_behaviors))
    ax_i.set_yticklabels(short_labels, fontsize=5)
    ax_i.set_xlabel('To', fontsize=7)
    fig.colorbar(im_i, ax=ax_i, shrink=0.7, aspect=15).ax.tick_params(labelsize=5)
axes[0].set_ylabel('From', fontsize=7)

for i in range(num_of_behaviors):
    for j in range(num_of_behaviors):
        if p_matrix[i, j] < 0.05:
            axes[2].text(j, i, '*', ha='center', va='center', fontsize=7, fontweight='bold')

fig.suptitle(f'dSPN Opto: Transition Probabilities (N={n_mice})', fontsize=9, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(output_folder + 'dSPN_transition_matrices.png', dpi=300)
fig.savefig(output_folder + 'dSPN_transition_matrices.pdf', dpi=300)
plt.close(fig)
print('Analysis 3a figure saved.')

#%% ========== ANALYSIS 3b: Bout Duration Distributions ==========
bout_theta = 1  # minimum bout length in frames

# Collect bout durations for laser-on vs laser-off, per behavior
bout_data = {b: {'on': [], 'off': []} for b in range(num_of_behaviors)}

for mouse in stim_day:
    predictions = stim_day[mouse]['merged']
    offset = opto_dict[mouse][1]
    first_stim = 3 * minute - int((offset / 1000.0) * 15)
    stim_times = np.arange(first_stim, 14 * minute, ISI)

    laser_on_frames = set()
    for stim in stim_times:
        laser_on_frames.update(range(int(stim), int(stim) + bin_duration))

    for b in range(num_of_behaviors):
        bouts = hf.segment_bouts(predictions, b, bout_theta)
        for bout_indices in bouts['indices']:
            # Classify bout: >50% laser-on frames = laser-on bout
            overlap = sum(1 for idx in bout_indices if idx in laser_on_frames)
            if overlap / len(bout_indices) > 0.5:
                bout_data[b]['on'].append(len(bout_indices))
            else:
                bout_data[b]['off'].append(len(bout_indices))

# --- Figure: Bout duration CDFs for VDB behaviors ---
vdb_behaviors = [0, 1, 2, 3]  # Grooming, Body lick, Wall lick, Floor lick
fig, axes = plt.subplots(1, 4, figsize=(180 * mm, 50 * mm), sharey=True)

for ax_i, bidx in zip(axes, vdb_behaviors):
    on_durs = np.array(bout_data[bidx]['on']) / FPS  # convert to seconds
    off_durs = np.array(bout_data[bidx]['off']) / FPS

    if len(off_durs) > 0:
        sorted_off = np.sort(off_durs)
        cdf_off = np.arange(1, len(sorted_off) + 1) / len(sorted_off)
        ax_i.step(sorted_off, cdf_off, color='gray', lw=1.5, label=f'OFF (n={len(off_durs)})')

    if len(on_durs) > 0:
        sorted_on = np.sort(on_durs)
        cdf_on = np.arange(1, len(sorted_on) + 1) / len(sorted_on)
        ax_i.step(sorted_on, cdf_on, color='#730AFF', lw=1.5, label=f'ON (n={len(on_durs)})')

    ax_i.set_title(behaviors[bidx], fontsize=7, fontweight='bold', color=colors[bidx])
    ax_i.set_xlabel('Duration (s)', fontsize=6)
    ax_i.legend(fontsize=5)
    ax_i.spines['top'].set_visible(False)
    ax_i.spines['right'].set_visible(False)
    ax_i.tick_params(labelsize=5)
    ax_i.set_xlim(0, 30)

    # KS test
    if len(on_durs) > 2 and len(off_durs) > 2:
        ks_stat, ks_p = stats.ks_2samp(on_durs, off_durs)
        sig = '***' if ks_p < 0.001 else '**' if ks_p < 0.01 else '*' if ks_p < 0.05 else 'n.s.'
        ax_i.text(0.95, 0.15, f'KS {sig}\np={ks_p:.3f}', transform=ax_i.transAxes,
                  fontsize=5, ha='right', va='bottom')

axes[0].set_ylabel('CDF', fontsize=7)
fig.suptitle(f'dSPN Opto: Bout Duration CDFs (N={n_mice})', fontsize=9, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(output_folder + 'dSPN_bout_duration_CDFs.png', dpi=300)
fig.savefig(output_folder + 'dSPN_bout_duration_CDFs.pdf', dpi=300)
plt.close(fig)
print('Analysis 3b figure saved.')

#%% ========== COMPOSITE FIGURE ==========
fig = plt.figure(figsize=(180 * mm, 130 * mm))
outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 0.9], hspace=0.45)

# Row 1: Transition matrices
gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], wspace=0.4)
for panel_idx, (mat, cmap, title) in enumerate([
    (mean_off, 'Purples', 'Laser OFF'),
    (mean_on, 'Purples', 'Laser ON'),
    (diff_matrix, 'RdBu_r', 'ON $-$ OFF')]):
    ax = fig.add_subplot(gs_top[panel_idx])
    if panel_idx < 2:
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax_abs, aspect='equal')
    else:
        im = ax.imshow(mat, cmap=cmap, vmin=-vmax_diff, vmax=vmax_diff, aspect='equal')
        for i in range(num_of_behaviors):
            for j in range(num_of_behaviors):
                if p_matrix[i, j] < 0.05:
                    ax.text(j, i, '*', ha='center', va='center', fontsize=5, fontweight='bold')
    ax.set_title(title, fontsize=7, fontweight='bold')
    ax.set_xticks(range(num_of_behaviors))
    ax.set_xticklabels(short_labels, fontsize=4, rotation=45, ha='right')
    ax.set_yticks(range(num_of_behaviors))
    ax.set_yticklabels(short_labels, fontsize=4)
    ax.set_xlabel('To', fontsize=6)
    if panel_idx == 0:
        ax.set_ylabel('From', fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.6, aspect=12).ax.tick_params(labelsize=4)
    ax.text(-0.15, 1.1, chr(ord('A') + panel_idx), transform=ax.transAxes,
            fontsize=10, fontweight='bold')

# Row 2: Bout duration CDFs
gs_bot = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[1], wspace=0.3)
for panel_idx, bidx in enumerate(vdb_behaviors):
    ax = fig.add_subplot(gs_bot[panel_idx])
    on_durs = np.array(bout_data[bidx]['on']) / FPS
    off_durs = np.array(bout_data[bidx]['off']) / FPS

    if len(off_durs) > 0:
        sorted_off = np.sort(off_durs)
        cdf_off = np.arange(1, len(sorted_off) + 1) / len(sorted_off)
        ax.step(sorted_off, cdf_off, color='gray', lw=1, label='OFF')
    if len(on_durs) > 0:
        sorted_on = np.sort(on_durs)
        cdf_on = np.arange(1, len(sorted_on) + 1) / len(sorted_on)
        ax.step(sorted_on, cdf_on, color='#730AFF', lw=1, label='ON')

    ax.set_title(behaviors[bidx], fontsize=6, fontweight='bold', color=colors[bidx])
    ax.set_xlabel('Duration (s)', fontsize=5)
    ax.legend(fontsize=4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=4)
    ax.set_xlim(0, 30)
    if panel_idx == 0:
        ax.set_ylabel('CDF', fontsize=6)
    ax.text(-0.15, 1.1, chr(ord('D') + panel_idx), transform=ax.transAxes,
            fontsize=10, fontweight='bold')

fig.suptitle(f'dSPN Optogenetic Stimulation (N={n_mice})', fontsize=10, fontweight='bold', y=0.98)
fig.savefig(output_folder + 'dSPN_opto_composite.png', dpi=300)
fig.savefig(output_folder + 'dSPN_opto_composite.pdf', dpi=300)
plt.close(fig)
print('Composite figure saved.')

#%% Summary statistics
print('\n' + '='*60)
print('dSPN ANALYSIS SUMMARY')
print('='*60)
print(f'\nAnalysis 3a: Transition Matrices (N={n_mice})')
print(f'  Significant cells (p<0.05): {np.sum(p_matrix < 0.05)} / {num_of_behaviors**2}')
sig_cells = np.argwhere(p_matrix < 0.05)
for cell in sig_cells:
    i, j = cell
    print(f'    {short_labels[i]} -> {short_labels[j]}: '
          f'OFF={mean_off[i,j]:.3f}, ON={mean_on[i,j]:.3f}, '
          f'diff={diff_matrix[i,j]:+.3f}, p={p_matrix[i,j]:.4f}')

print(f'\nAnalysis 3b: Bout Duration (VDB behaviors)')
for bidx in vdb_behaviors:
    on_durs = np.array(bout_data[bidx]['on']) / FPS
    off_durs = np.array(bout_data[bidx]['off']) / FPS
    if len(on_durs) > 2 and len(off_durs) > 2:
        ks_stat, ks_p = stats.ks_2samp(on_durs, off_durs)
        print(f'  {behaviors[bidx]}: ON median={np.median(on_durs):.1f}s (n={len(on_durs)}), '
              f'OFF median={np.median(off_durs):.1f}s (n={len(off_durs)}), '
              f'KS={ks_stat:.3f}, p={ks_p:.4f}')
    else:
        print(f'  {behaviors[bidx]}: insufficient bouts (ON={len(on_durs)}, OFF={len(off_durs)})')
print('='*60)
print(f'\nAll figures saved to: {output_folder}')

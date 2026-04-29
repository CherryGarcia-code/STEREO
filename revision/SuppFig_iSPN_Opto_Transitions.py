#%% SuppFig: iSPN Opto Transition Probability Matrices
"""
Addresses R4-Fig3JK (transition probabilities during iSPN opto).

Panels:
  A. Transition matrix — Laser OFF
  B. Transition matrix — Laser ON
  C. Difference matrix (ON − OFF) with significance markers
  D–F. Pre/During/Post fraction V-graphs for surface licking, locomotion, grooming

Uses CTM_Aug24.pkl, a2a_opto cohort.
Reuses pattern from analyses/iSPN_opto_analysis.py.
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
    mm, second, minute, n_behaviors, behaviors, colors, short_labels, lut,
    ISI, bin_duration, RNN_offset, FPS,
    load_CTM, setup_style, save_fig
)

setup_style()
output_folder = 'revision/output/'
os.makedirs(output_folder, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
CTM = load_CTM('Dec24')
stim_days_raw = CTM['a2a_opto']
stim_days = {key: stim_days_raw[key] for key in ['cocaine6laserStim', 'cocaine8laserStim']
             if key in stim_days_raw}
del CTM
print(f'Available stim days: {list(stim_days.keys())}')

# ---------------------------------------------------------------------------
# Constants (from iSPN_opto_analysis.py)
# ---------------------------------------------------------------------------
num_of_stims_cocaine = 14
first_stim = 10 * minute - RNN_offset  # 8993 frames

# Mouse selection: >=50% licking in pre-stim period
# Try cocaine6laserStim first, then cocaine8laserStim
selected = {}
for mouse in set().union(*[stim_days[d].keys() for d in stim_days]):
    for day in ['cocaine6laserStim', 'cocaine8laserStim']:
        if day in stim_days and mouse in stim_days[day]:
            preds = stim_days[day][mouse]['merged']
            # Pre-stim licking fraction (behaviors 2,3 = floor/wall licking)
            pre = preds[:first_stim]
            lick_frac = np.mean((pre == 2) | (pre == 3))
            if lick_frac >= 0.5:
                selected[mouse] = (day, preds)
                break

mice_list = sorted(selected.keys())
n_mice = len(mice_list)
print(f'Selected {n_mice} mice with >=50% pre-stim licking')

# ---------------------------------------------------------------------------
# Compute transition matrices: laser ON vs OFF
# ---------------------------------------------------------------------------
stim_times = np.arange(first_stim, 28 * minute, ISI)

trans_on_all = []
trans_off_all = []

for mouse in mice_list:
    _, predictions = selected[mouse]

    laser_on_set = set()
    for stim in stim_times:
        laser_on_set.update(range(int(stim), int(stim) + bin_duration))

    trans_on = np.zeros((n_behaviors, n_behaviors))
    trans_off = np.zeros((n_behaviors, n_behaviors))

    for i in range(len(predictions) - 1):
        if predictions[i] != predictions[i + 1]:
            if i in laser_on_set:
                trans_on[predictions[i], predictions[i + 1]] += 1
            else:
                trans_off[predictions[i], predictions[i + 1]] += 1

    # Row-normalize
    for b in range(n_behaviors):
        if trans_on[b, :].sum() > 0:
            trans_on[b, :] /= trans_on[b, :].sum()
        if trans_off[b, :].sum() > 0:
            trans_off[b, :] /= trans_off[b, :].sum()

    trans_on_all.append(trans_on)
    trans_off_all.append(trans_off)

trans_on_all = np.array(trans_on_all)
trans_off_all = np.array(trans_off_all)

mean_on = np.nanmean(trans_on_all, axis=0)
mean_off = np.nanmean(trans_off_all, axis=0)
diff_matrix = mean_on - mean_off

# Paired t-test per cell
p_matrix = np.ones((n_behaviors, n_behaviors))
for i in range(n_behaviors):
    for j in range(n_behaviors):
        vals_on = trans_on_all[:, i, j]
        vals_off = trans_off_all[:, i, j]
        valid = ~(np.isnan(vals_on) | np.isnan(vals_off))
        if valid.sum() >= 3:
            _, p_matrix[i, j] = stats.ttest_rel(vals_on[valid], vals_off[valid])

# Print significant transitions
print('\n--- Significant transition changes (p<0.05) ---')
sig_cells = np.argwhere(p_matrix < 0.05)
for cell in sig_cells:
    i, j = cell
    print(f'  {short_labels[i]} -> {short_labels[j]}: '
          f'OFF={mean_off[i,j]:.3f}, ON={mean_on[i,j]:.3f}, '
          f'diff={diff_matrix[i,j]:+.3f}, p={p_matrix[i,j]:.4f}')

# ---------------------------------------------------------------------------
# Pre/During/Post epoch fractions for V-graphs
# ---------------------------------------------------------------------------
target_behaviors = {
    'Surface licking': [2, 3],
    'Locomotion': [7],
    'Grooming': [4],
}
target_colors_vgraph = {
    'Surface licking': '#d73027',
    'Locomotion': '#3399cc',
    'Grooming': '#c4a7e7',
}

epoch_fracs = {bname: np.full((n_mice, num_of_stims_cocaine, 3), np.nan)
               for bname in target_behaviors}

for m_idx, mouse in enumerate(mice_list):
    _, predictions = selected[mouse]
    for s_idx, stim in enumerate(stim_times[:num_of_stims_cocaine]):
        stim = int(stim)
        pre_epoch = predictions[stim - bin_duration : stim]
        dur_epoch = predictions[stim : stim + bin_duration]
        post_epoch = predictions[stim + bin_duration : stim + 2 * bin_duration]

        for bname, b_indices in target_behaviors.items():
            for e_idx, epoch in enumerate([pre_epoch, dur_epoch, post_epoch]):
                if len(epoch) > 0:
                    epoch_fracs[bname][m_idx, s_idx, e_idx] = np.mean(
                        np.isin(epoch, b_indices))

# Average across stims per mouse
mouse_fracs = {bname: np.nanmean(epoch_fracs[bname], axis=1)  # (n_mice, 3)
               for bname in target_behaviors}

# Stats: paired t-test pre vs during
print('\n--- V-graph stats (pre vs during laser) ---')
for bname in target_behaviors:
    pre = mouse_fracs[bname][:, 0]
    dur = mouse_fracs[bname][:, 1]
    t_stat, p_val = stats.ttest_rel(pre, dur)
    print(f'  {bname}: pre={np.mean(pre):.3f} dur={np.mean(dur):.3f} t={t_stat:.2f} p={p_val:.4f}')

# ===========================================================================
# BUILD THE FIGURE
# ===========================================================================
print('\nGenerating figure...')
fig = plt.figure(figsize=(180 * mm, 120 * mm))
outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 0.8], hspace=0.45)

# --- Row 1: Transition matrices ---
gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], wspace=0.4)

vmax_abs = max(np.nanmax(mean_off), np.nanmax(mean_on))

for panel_idx, (mat, cmap, title, panel_lbl) in enumerate([
    (mean_off, 'Blues', 'Laser OFF', 'A'),
    (mean_on, 'Blues', 'Laser ON (iSPN)', 'B'),
    (diff_matrix, 'RdBu_r', 'ON $-$ OFF', 'C'),
]):
    ax = fig.add_subplot(gs_top[panel_idx])
    if panel_idx < 2:
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax_abs, aspect='equal')
    else:
        vmax_diff = np.nanmax(np.abs(diff_matrix))
        im = ax.imshow(mat, cmap=cmap, vmin=-vmax_diff, vmax=vmax_diff, aspect='equal')
        for i in range(n_behaviors):
            for j in range(n_behaviors):
                if p_matrix[i, j] < 0.05:
                    ax.text(j, i, '*', ha='center', va='center', fontsize=7, fontweight='bold')

    ax.set_xticks(range(n_behaviors))
    ax.set_xticklabels(short_labels, fontsize=5, rotation=45, ha='right')
    ax.set_yticks(range(n_behaviors))
    ax.set_yticklabels(short_labels, fontsize=5)
    ax.set_xlabel('To', fontsize=7)
    ax.set_title(title, fontsize=8, fontweight='bold', pad=12)
    fig.colorbar(im, ax=ax, shrink=0.7, aspect=15).ax.tick_params(labelsize=5)
    if panel_idx == 0:
        ax.set_ylabel('From', fontsize=7)
    ax.text(-0.15, 1.15, panel_lbl, transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')

# --- Row 2: V-graphs ---
gs_bot = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], wspace=0.35)

epoch_labels = ['Pre', 'During', 'Post']
for v_idx, (bname, panel_lbl) in enumerate(zip(target_behaviors.keys(), ['D', 'E', 'F'])):
    ax = fig.add_subplot(gs_bot[v_idx])
    fracs = mouse_fracs[bname]  # (n_mice, 3)
    c = target_colors_vgraph[bname]

    # Individual mouse lines
    for m in range(n_mice):
        ax.plot(range(3), fracs[m, :], color='gray', alpha=0.25, lw=0.5, zorder=1)

    # Mean + SEM
    mean_v = np.nanmean(fracs, axis=0)
    sem_v = np.nanstd(fracs, axis=0, ddof=0) / np.sqrt(n_mice)
    ax.errorbar(range(3), mean_v, yerr=sem_v, color=c, lw=2, capsize=4, capthick=1.5,
                marker='o', markersize=5, markerfacecolor=c, zorder=5)

    ax.set_xticks(range(3))
    ax.set_xticklabels(epoch_labels, fontsize=7)
    ax.set_ylabel('Fraction of frames', fontsize=7)
    ax.set_title(bname, fontsize=8, fontweight='bold', color=c, pad=12)
    ax.tick_params(labelsize=6)

    # Significance bracket
    pre = fracs[:, 0]; dur = fracs[:, 1]
    _, p = stats.ttest_rel(pre, dur)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
    ymax = np.nanmax(fracs) * 1.1
    ax.plot([0, 1], [ymax, ymax], 'k-', lw=0.8)
    ax.text(0.5, ymax * 1.02, sig, ha='center', fontsize=7)
    ax.text(-0.15, 1.15, panel_lbl, transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')

fig.suptitle(f'iSPN Opto: Transition Probabilities & Behavior Fractions (N={n_mice})',
             fontsize=9, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95])
save_fig(fig, output_folder, 'SuppFig3_iSPN_Opto_Transitions')
print(f'\nDone — N={n_mice} mice')

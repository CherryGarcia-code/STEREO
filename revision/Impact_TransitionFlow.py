#%% Transition flow diagrams for optogenetic experiments
"""
Creates alluvial-style transition flow diagrams showing how behavioral
transitions are funneled (dSPN) or diversified (iSPN) during laser stimulation.
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
import matplotlib.patches as mpatches

from revision_utils import (
    mm, second, minute, FPS, ISI, bin_duration, lut,
    n_behaviors, behaviors, colors, short_labels,
    laser_color, dSPN_color, iSPN_color,
    load_CTM, setup_style, save_fig
)

setup_style()
output_folder = 'revision/output/'
os.makedirs(output_folder, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper: compute transition counts between behaviors (source -> dest)
# ---------------------------------------------------------------------------
def transitions_in_epochs(predictions, epoch_frames, n_beh):
    """Count transitions within specified frame sets."""
    mat = np.zeros((n_beh, n_beh))
    for i in range(len(predictions) - 1):
        if predictions[i] != predictions[i + 1]:
            if i in epoch_frames:
                mat[predictions[i], predictions[i + 1]] += 1
    return mat

# ---------------------------------------------------------------------------
# iSPN data (new encoding, remapped)
# ---------------------------------------------------------------------------
CTM_remap = load_CTM('Dec24')
a2a_opto = CTM_remap.get('a2a_opto', {})

# Find stim_day
iSPN_stim = {}
for trial_name in ['cocaine6laserStim', 'cocaine8laserStim']:
    if trial_name in a2a_opto:
        iSPN_stim.update(a2a_opto[trial_name])

# Filter to mice with >50% pre-stim licking
from revision_utils import PATHO_LICKING, grouping_lut, RNN_offset

with open('data/opto_alignment_dict.pkl', 'rb') as f:
    opto_dict = pickle.load(f)

# iSPN uses fixed timing (no per-mouse offset)
iSPN_first_stim = 10 * minute - RNN_offset
iSPN_stim_times = np.arange(iSPN_first_stim, 28 * minute, ISI)

iSPN_mice = {}
for mouse in iSPN_stim:
    preds = iSPN_stim[mouse]['merged']
    pre = preds[:iSPN_first_stim]
    grouped = grouping_lut[pre]
    if np.count_nonzero(grouped == PATHO_LICKING) / pre.size > 0.50:
        iSPN_mice[mouse] = preds
del CTM_remap

# iSPN transition matrices: laser-OFF vs laser-ON

iSPN_on = np.zeros((n_behaviors, n_behaviors))
iSPN_off = np.zeros((n_behaviors, n_behaviors))

for mouse, predictions in list(iSPN_mice.items()):
    on_set = set()
    off_set = set()
    for stim in iSPN_stim_times:
        on_set.update(range(int(stim), int(stim) + bin_duration))
    for i in range(len(predictions)):
        if i not in on_set:
            off_set.add(i)

    iSPN_on += transitions_in_epochs(predictions, on_set, n_behaviors)
    iSPN_off += transitions_in_epochs(predictions, off_set, n_behaviors)

# ---------------------------------------------------------------------------
# dSPN data (Dec24 = canonical new encoding)
# ---------------------------------------------------------------------------
CTM_raw = load_CTM('Dec24')
dSPN_stim_day = CTM_raw['drd1_opto']['AloneStim']
del CTM_raw

dSPN_on = np.zeros((n_behaviors, n_behaviors))
dSPN_off = np.zeros((n_behaviors, n_behaviors))

for mouse in dSPN_stim_day:
    if mouse not in opto_dict:
        continue
    predictions = dSPN_stim_day[mouse]['merged']
    offset = opto_dict[mouse][1]
    first_stim = 3 * minute - int((offset / 1000.0) * FPS)
    stim_times = np.arange(first_stim, 14 * minute, ISI)

    on_set = set()
    off_set = set()
    for stim in stim_times:
        on_set.update(range(int(stim), int(stim) + bin_duration))
    for i in range(len(predictions)):
        if i not in on_set:
            off_set.add(i)

    dSPN_on += transitions_in_epochs(predictions, on_set, n_behaviors)
    dSPN_off += transitions_in_epochs(predictions, off_set, n_behaviors)

# ---------------------------------------------------------------------------
# Flow diagram drawing function
# ---------------------------------------------------------------------------
def draw_flow(ax, trans_mat, beh_names, beh_colors, title, top_n=5, highlight_color=None):
    """Draw a simplified flow diagram showing top transitions as curved arrows."""
    n_beh = len(beh_names)

    # Normalize: row-wise probabilities
    row_sums = trans_mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    prob_mat = trans_mat / row_sums

    # Get top transitions by raw count
    flat_idx = np.argsort(trans_mat.ravel())[::-1]
    top_pairs = []
    for idx in flat_idx:
        i, j = divmod(idx, n_beh)
        if i != j and trans_mat[i, j] > 0:
            top_pairs.append((i, j, trans_mat[i, j], prob_mat[i, j]))
            if len(top_pairs) >= top_n:
                break

    # Layout: behaviors on left and right columns
    y_positions = np.linspace(0.9, 0.1, n_beh)

    # Draw behavior labels on both sides
    for idx in range(n_beh):
        y = y_positions[idx]
        # Left side (source)
        ax.text(0.05, y, beh_names[idx], ha='left', va='center', fontsize=5,
                color=beh_colors[idx], fontweight='bold')
        # Right side (destination)
        ax.text(0.95, y, beh_names[idx], ha='right', va='center', fontsize=5,
                color=beh_colors[idx], fontweight='bold')
        # Small markers
        ax.plot(0.25, y, 'o', color=beh_colors[idx], markersize=5, zorder=5)
        ax.plot(0.75, y, 'o', color=beh_colors[idx], markersize=5, zorder=5)

    # Draw arrows for top transitions
    max_count = max(t[2] for t in top_pairs) if top_pairs else 1
    for src, dst, count, prob in top_pairs:
        y_src = y_positions[src]
        y_dst = y_positions[dst]
        lw = 0.5 + 4.0 * (count / max_count)
        alpha = 0.3 + 0.5 * (count / max_count)

        # Curved arrow using FancyArrowPatch
        style = f"arc3,rad={0.15 if src != dst else 0.0}"
        arrow = mpatches.FancyArrowPatch(
            (0.28, y_src), (0.72, y_dst),
            connectionstyle=style,
            arrowstyle='->,head_width=3,head_length=2',
            lw=lw, color=beh_colors[src], alpha=alpha,
            mutation_scale=10
        )
        ax.add_patch(arrow)

        # Probability annotation at midpoint
        mid_y = (y_src + y_dst) / 2
        mid_x = 0.50
        ax.text(mid_x, mid_y, f'{prob:.0%}', fontsize=4, ha='center', va='center',
                color=beh_colors[src], alpha=0.8)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=8, fontweight='bold', color=highlight_color or 'k', pad=12)
    ax.axis('off')

# ---------------------------------------------------------------------------
# Figure: 2 rows x 2 columns (OFF vs ON for each pathway)
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(180 * mm, 160 * mm))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.2)

# Row 1: iSPN
ax1 = fig.add_subplot(gs[0, 0])
draw_flow(ax1, iSPN_off, behaviors, colors, f'iSPN Laser OFF (N={len(iSPN_mice)})',
          top_n=8, highlight_color='gray')
ax1.text(-0.05, 1.15, 'A', transform=ax1.transAxes, fontsize=12, fontweight='bold', va='top')

ax2 = fig.add_subplot(gs[0, 1])
draw_flow(ax2, iSPN_on, behaviors, colors, f'iSPN Laser ON',
          top_n=8, highlight_color=iSPN_color)
ax2.text(-0.05, 1.15, 'B', transform=ax2.transAxes, fontsize=12, fontweight='bold', va='top')

# Row 2: dSPN
ax3 = fig.add_subplot(gs[1, 0])
draw_flow(ax3, dSPN_off, behaviors, colors, f'dSPN Laser OFF (N={len(dSPN_stim_day)})',
          top_n=8, highlight_color='gray')
ax3.text(-0.05, 1.15, 'C', transform=ax3.transAxes, fontsize=12, fontweight='bold', va='top')

ax4 = fig.add_subplot(gs[1, 1])
draw_flow(ax4, dSPN_on, behaviors, colors, f'dSPN Laser ON',
          top_n=8, highlight_color=dSPN_color)
ax4.text(-0.05, 1.15, 'D', transform=ax4.transAxes, fontsize=12, fontweight='bold', va='top')

# Add transition diversity summary
iSPN_off_entropy = -np.sum((iSPN_off / max(iSPN_off.sum(), 1)) *
                            np.log2(np.maximum(iSPN_off / max(iSPN_off.sum(), 1), 1e-10)))
iSPN_on_entropy = -np.sum((iSPN_on / max(iSPN_on.sum(), 1)) *
                           np.log2(np.maximum(iSPN_on / max(iSPN_on.sum(), 1), 1e-10)))
dSPN_off_entropy = -np.sum((dSPN_off / max(dSPN_off.sum(), 1)) *
                            np.log2(np.maximum(dSPN_off / max(dSPN_off.sum(), 1), 1e-10)))
dSPN_on_entropy = -np.sum((dSPN_on / max(dSPN_on.sum(), 1)) *
                           np.log2(np.maximum(dSPN_on / max(dSPN_on.sum(), 1), 1e-10)))

print(f'iSPN transition entropy: OFF={iSPN_off_entropy:.2f}, ON={iSPN_on_entropy:.2f}')
print(f'dSPN transition entropy: OFF={dSPN_off_entropy:.2f}, ON={dSPN_on_entropy:.2f}')

save_fig(fig, output_folder, 'Impact_TransitionFlow')
print(f'\nDone — iSPN: {len(iSPN_mice)}, dSPN: {len(dSPN_stim_day)} mice')

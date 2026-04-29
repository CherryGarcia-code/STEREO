#%% SuppFig: Time-Warped Splash Test Photometry
"""
Addresses R4-Fig7: Why quartiles instead of time-warp?

Compares time-warp interpolation vs quartile-based bout separation for
body licking (behavior 5) photometry during splashTest.

Panels (2 rows x 3 cols):
  Row 1 (dSPN): A. Time-warped mean+SEM, B. Quartile traces overlaid,
                 C. Time-warped heatmap sorted by original duration
  Row 2 (iSPN): D-F same layout

Uses CMT_Dec24.pkl with photom_exclusions applied.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from revision_utils import (
    mm, second, FPS, n_behaviors,
    load_CMT, get_pathway, get_photom_signal, photom_exclusions,
    segment_bouts_for_photom, segment_signal_window,
    dSPN_photom_color, iSPN_photom_color, quartile_colors,
    setup_style, save_fig
)
import helper_functions as hf

setup_style()
output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(output_folder, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BODY_LICKING = 5
TRIAL = 'splashTest'
sig_type = 'z'
smooth_window = 3

pre_window = 3 * second      # 45 frames before bout onset
post_window = 5 * second      # 75 frames after bout onset

# Time-warp parameters
target_bout_length = 2 * second   # 30 frames: interpolated bout segment
tail_length = 2 * second          # 30 frames: post-bout tail kept as-is
warped_total = pre_window + target_bout_length + tail_length  # 105 frames

# Photometry cohorts (those with photom data)
photom_cohorts = ['drd1_hm4di', 'drd1_hm3dq', 'controls', 'a2a_hm4di', 'a2a_hm3dq']

# ---------------------------------------------------------------------------
# Load data and apply exclusions
# ---------------------------------------------------------------------------
print('Loading CMT_Dec24...')
CMT = load_CMT('Dec24')

# Apply photom_exclusions
trials_all = ['saline1', 'saline2', 'saline3',
              'cocaine1', 'cocaine2', 'cocaine3', 'cocaine4', 'cocaine5',
              'splashTest']

for cohort, mouse in photom_exclusions['remove_mouse']:
    if cohort in CMT and mouse in CMT[cohort]:
        for tr in trials_all:
            if tr in CMT[cohort][mouse] and 'photom' in CMT[cohort][mouse][tr]:
                del CMT[cohort][mouse][tr]['photom']

for cohort, mouse, side in photom_exclusions['remove_hemisphere']:
    if cohort in CMT and mouse in CMT[cohort]:
        for tr in trials_all:
            if (tr in CMT[cohort][mouse] and 'photom' in CMT[cohort][mouse][tr]
                    and side in CMT[cohort][mouse][tr]['photom']):
                del CMT[cohort][mouse][tr]['photom'][side]

for cohort, mouse, trial in photom_exclusions['remove_trial']:
    if (cohort in CMT and mouse in CMT[cohort]
            and trial in CMT[cohort][mouse]
            and 'photom' in CMT[cohort][mouse][trial]):
        del CMT[cohort][mouse][trial]['photom']

print('Exclusions applied.')

# ---------------------------------------------------------------------------
# Extract body licking bouts with photometry during splashTest
# ---------------------------------------------------------------------------
data = {'dSPN': {'traces': [], 'durations': []},
        'iSPN': {'traces': [], 'durations': []}}

for cohort in photom_cohorts:
    if cohort not in CMT:
        continue
    for mouse in CMT[cohort]:
        pathway = get_pathway(cohort, mouse)
        if pathway is None:
            continue
        spn = 'dSPN' if pathway == 'drd1' else 'iSPN'

        if TRIAL not in CMT[cohort][mouse]:
            continue
        entry = CMT[cohort][mouse][TRIAL]
        if 'photom' not in entry:
            continue

        # Get predictions (Dec24 format already flattened by load_CMT)
        preds = entry['merged']

        # Get photometry signals
        signals = get_photom_signal(entry, side='both', sig_type=sig_type, smooth=True)
        if not signals:
            continue

        # Find body licking bouts (no preceding-behavior constraint, min 1 frame)
        bouts = segment_bouts_for_photom(preds, BODY_LICKING, min_bout_dur=1)

        for onset, dur in bouts:
            # Need enough signal before onset and after bout end
            bout_end = onset + dur
            for sig, side_name in signals:
                # Check boundaries: need pre_window before onset, post_window after onset
                # For quartile approach: pre + post around onset
                if onset - pre_window < 0 or onset + post_window > len(sig):
                    continue

                # --- Quartile trace: fixed window around onset ---
                quartile_seg = sig[onset - pre_window: onset + post_window]
                if np.all(np.isnan(quartile_seg)):
                    continue

                # --- Time-warped trace ---
                # Pre-onset segment (keep as-is)
                pre_seg = sig[onset - pre_window: onset]

                # Bout segment (interpolate to target_bout_length)
                if bout_end > len(sig):
                    continue
                bout_seg = sig[onset: bout_end]
                if dur > 0:
                    x_orig = np.linspace(0, 1, dur)
                    x_new = np.linspace(0, 1, target_bout_length)
                    bout_warped = np.interp(x_new, x_orig, bout_seg)
                else:
                    continue

                # Post-bout tail (keep as-is, take tail_length frames after bout end)
                tail_start = bout_end
                tail_end = tail_start + tail_length
                if tail_end > len(sig):
                    continue
                tail_seg = sig[tail_start: tail_end]

                if np.all(np.isnan(pre_seg)) or np.all(np.isnan(tail_seg)):
                    continue

                warped_trace = np.concatenate([pre_seg, bout_warped, tail_seg])
                assert warped_trace.shape[0] == warped_total, \
                    f"Warped trace length {warped_trace.shape[0]} != {warped_total}"

                data[spn]['traces'].append(warped_trace)
                data[spn]['durations'].append(dur)

# Also collect raw (non-warped) traces for quartile panel
quartile_data = {'dSPN': {'traces': [], 'durations': []},
                 'iSPN': {'traces': [], 'durations': []}}

for cohort in photom_cohorts:
    if cohort not in CMT:
        continue
    for mouse in CMT[cohort]:
        pathway = get_pathway(cohort, mouse)
        if pathway is None:
            continue
        spn = 'dSPN' if pathway == 'drd1' else 'iSPN'

        if TRIAL not in CMT[cohort][mouse]:
            continue
        entry = CMT[cohort][mouse][TRIAL]
        if 'photom' not in entry:
            continue

        preds = entry['merged']
        signals = get_photom_signal(entry, side='both', sig_type=sig_type, smooth=True)
        if not signals:
            continue

        bouts = segment_bouts_for_photom(preds, BODY_LICKING, min_bout_dur=1)

        for onset, dur in bouts:
            for sig, side_name in signals:
                if onset - pre_window < 0 or onset + post_window > len(sig):
                    continue
                seg = sig[onset - pre_window: onset + post_window]
                if np.all(np.isnan(seg)):
                    continue
                quartile_data[spn]['traces'].append(seg)
                quartile_data[spn]['durations'].append(dur)

del CMT

# ---------------------------------------------------------------------------
# Print summary stats
# ---------------------------------------------------------------------------
for spn in ['dSPN', 'iSPN']:
    n_bouts = len(data[spn]['traces'])
    durs = np.array(data[spn]['durations'])
    if n_bouts > 0:
        print(f'{spn}: {n_bouts} bouts, duration median={np.median(durs)/FPS:.2f}s, '
              f'mean={np.mean(durs)/FPS:.2f}s, range=[{np.min(durs)/FPS:.2f}, {np.max(durs)/FPS:.2f}]s')
    else:
        print(f'{spn}: 0 bouts')

    n_q = len(quartile_data[spn]['traces'])
    print(f'  Quartile data: {n_q} bouts')

# ---------------------------------------------------------------------------
# Helper: split into quartiles
# ---------------------------------------------------------------------------
def split_quartiles(traces, durations):
    """Split traces into 4 quartiles by duration. Returns list of 4 arrays."""
    traces = np.array(traces, dtype=np.float32)
    durations = np.array(durations)
    q25, q50, q75 = np.percentile(durations, [25, 50, 75])
    masks = [
        durations <= q25,
        (durations > q25) & (durations <= q50),
        (durations > q50) & (durations <= q75),
        durations > q75,
    ]
    return [traces[m] for m in masks], [q25, q50, q75]


# ===========================================================================
# BUILD THE FIGURE (2 rows x 3 cols)
# ===========================================================================
print('\nGenerating figure...')
fig = plt.figure(figsize=(180 * mm, 120 * mm))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

panel_labels = ['A', 'B', 'C', 'D', 'E', 'F']
spn_list = ['dSPN', 'iSPN']
spn_colors = [dSPN_photom_color, iSPN_photom_color]

# Time-warp x-axis: ticks every 15 frames
warp_xtick_positions = np.arange(0, warped_total, second)  # 0, 15, 30, 45, 60, 75, 90
warp_xtick_labels = ['S-3', 'S-2', 'S-1', 'S', '', 'E', 'E+1']
# Ensure we don't exceed the tick count
n_ticks = min(len(warp_xtick_positions), len(warp_xtick_labels))
warp_xtick_positions = warp_xtick_positions[:n_ticks]
warp_xtick_labels = warp_xtick_labels[:n_ticks]

for row_idx, (spn, color) in enumerate(zip(spn_list, spn_colors)):
    # --- Panel A/D: Time-warped mean + SEM ---
    ax_warp = fig.add_subplot(gs[row_idx, 0])
    traces_w = np.array(data[spn]['traces'], dtype=np.float32)
    if traces_w.shape[0] > 0:
        mean_w = np.nanmean(traces_w, axis=0)
        sem_w = np.nanstd(traces_w, axis=0, ddof=0) / np.sqrt(traces_w.shape[0])
        x = np.arange(warped_total)
        ax_warp.plot(x, mean_w, color=color, lw=1.0)
        ax_warp.fill_between(x, mean_w - sem_w, mean_w + sem_w, color=color, alpha=0.3)
        # Mark bout start and end
        ax_warp.axvline(x=pre_window, color='k', ls='--', lw=0.5)
        ax_warp.axvline(x=pre_window + target_bout_length, color='k', ls='--', lw=0.5)
    ax_warp.set_xticks(warp_xtick_positions)
    ax_warp.set_xticklabels(warp_xtick_labels, fontsize=6)
    ax_warp.set_ylabel(r'$\Delta$F/F (z)', fontsize=8)
    ax_warp.set_title(f'{spn} — Time-warped (N={traces_w.shape[0]})', fontsize=8, fontweight='bold', pad=12)
    panel = panel_labels[row_idx * 3]
    ax_warp.text(-0.15, 1.15, panel, transform=ax_warp.transAxes, fontsize=10,
                 fontweight='bold', va='top')

    # --- Panel B/E: Quartile traces overlaid ---
    ax_q = fig.add_subplot(gs[row_idx, 1])
    q_traces = quartile_data[spn]['traces']
    q_durs = quartile_data[spn]['durations']
    if len(q_traces) > 0:
        quartiles, q_bounds = split_quartiles(q_traces, q_durs)
        q_labels = ['Q1 (shortest)', 'Q2', 'Q3', 'Q4 (longest)']
        x_q = np.arange(pre_window + post_window)
        for qi, (q_arr, q_col, q_lab) in enumerate(zip(quartiles, quartile_colors, q_labels)):
            if q_arr.shape[0] == 0:
                continue
            q_mean = np.nanmean(q_arr, axis=0)
            q_sem = np.nanstd(q_arr, axis=0, ddof=0) / np.sqrt(q_arr.shape[0])
            ax_q.plot(x_q, q_mean, color=q_col, lw=0.8, label=f'{q_lab} (n={q_arr.shape[0]})')
            ax_q.fill_between(x_q, q_mean - q_sem, q_mean + q_sem, color=q_col, alpha=0.2)
        ax_q.axvline(x=pre_window, color='k', ls='--', lw=0.5)
    # X-axis: seconds relative to onset
    q_xtick_pos = np.arange(0, pre_window + post_window + 1, 2 * second)
    q_xtick_labels_arr = ((q_xtick_pos - pre_window) / second).astype(int)
    ax_q.set_xticks(q_xtick_pos)
    ax_q.set_xticklabels(q_xtick_labels_arr, fontsize=6)
    ax_q.set_xlabel('Time from onset (s)', fontsize=8)
    ax_q.set_ylabel(r'$\Delta$F/F (z)', fontsize=8)
    ax_q.set_title(f'{spn} — Quartile split', fontsize=8, fontweight='bold', pad=12)
    ax_q.legend(fontsize=5, loc='upper right')
    panel = panel_labels[row_idx * 3 + 1]
    ax_q.text(-0.15, 1.15, panel, transform=ax_q.transAxes, fontsize=10,
              fontweight='bold', va='top')

    # --- Panel C/F: Time-warped heatmap sorted by original duration ---
    ax_hm = fig.add_subplot(gs[row_idx, 2])
    if traces_w.shape[0] > 0:
        durs_w = np.array(data[spn]['durations'])
        sort_idx = np.argsort(durs_w)
        sorted_traces = traces_w[sort_idx, :]
        im = ax_hm.imshow(sorted_traces, aspect='auto', cmap='magma',
                          vmin=-0.5, vmax=1.5, interpolation='nearest',
                          extent=[0, warped_total, sorted_traces.shape[0], 0])
        ax_hm.axvline(x=pre_window, color='w', ls='--', lw=0.5)
        ax_hm.axvline(x=pre_window + target_bout_length, color='w', ls='--', lw=0.5)
        ax_hm.set_xticks(warp_xtick_positions)
        ax_hm.set_xticklabels(warp_xtick_labels, fontsize=6)
        ax_hm.set_ylabel('Bout (sorted by duration)', fontsize=8)
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=6)
    ax_hm.set_title(f'{spn} — Heatmap (N={traces_w.shape[0]})', fontsize=8, fontweight='bold', pad=12)
    panel = panel_labels[row_idx * 3 + 2]
    ax_hm.text(-0.05, 1.15, panel, transform=ax_hm.transAxes, fontsize=10,
               fontweight='bold', va='top')

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
save_fig(fig, output_folder, 'SuppFig14_TimeWarped_SplashTest')
print('\nDone.')

#%% SuppFig_Grooming_Photometry.py
"""
Supplementary Figure: Grooming & Body-Licking Photometry During Splash Test
Addresses R4-Fig7: Show grooming neuronal activity during splash test.

Panels (2 rows x 3 cols):
  Row 1 (dSPN): A. Grooming mean trace, B. Body licking mean trace, C. Grooming BRT heatmap
  Row 2 (iSPN): D-F same
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
    dSPN_photom_color, iSPN_photom_color,
    setup_style, save_fig
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
GROOMING = 4
BODY_LICKING = 5

PRE_FRAMES = 3 * second    # 45 frames (3 s)
POST_FRAMES = 5 * second   # 75 frames (5 s)
WINDOW = PRE_FRAMES + POST_FRAMES  # 120 frames total
BASELINE_FRAMES = 30       # first 2 seconds for baseline subtraction

MIN_BOUT_DUR = 3           # minimum bout duration in frames

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def is_excluded_mouse(cohort, mouse):
    for c, m in photom_exclusions['remove_mouse']:
        if c == cohort and m == mouse:
            return True
    return False

def is_excluded_hemisphere(cohort, mouse, side):
    for c, m, s in photom_exclusions['remove_hemisphere']:
        if c == cohort and m == mouse and s == side:
            return True
    return False

def is_excluded_trial(cohort, mouse, trial):
    for c, m, t in photom_exclusions['remove_trial']:
        if c == cohort and m == mouse and t == trial:
            return True
    return False


def collect_bouts(CMT, target_beh):
    """Collect bout-triggered photometry traces for a given behavior during splashTest.
    Returns dict: {'drd1': {'traces': [...], 'durations': [...], 'mice': set()},
                   'a2a':  {...}}
    """
    result = {
        'drd1': {'traces': [], 'durations': [], 'mice': set()},
        'a2a':  {'traces': [], 'durations': [], 'mice': set()},
    }

    for cohort in CMT:
        for mouse in CMT[cohort]:
            if is_excluded_mouse(cohort, mouse):
                continue

            pathway = get_pathway(cohort, mouse)
            if pathway is None:
                continue

            if 'splashTest' not in CMT[cohort][mouse]:
                continue

            entry = CMT[cohort][mouse]['splashTest']

            if is_excluded_trial(cohort, mouse, 'splashTest'):
                continue

            preds = entry['merged']
            if not isinstance(preds, np.ndarray):
                continue

            # Find bouts — no from_beh requirement for splash test
            bouts = segment_bouts_for_photom(preds, target_beh,
                                             from_beh=None, min_bout_dur=MIN_BOUT_DUR)
            if len(bouts) == 0:
                continue

            # Get photometry signals for both hemispheres
            signals = get_photom_signal(entry, side='both', sig_type='z')
            if len(signals) == 0:
                continue

            for sig, side_name in signals:
                if is_excluded_hemisphere(cohort, mouse, side_name):
                    continue

                for onset, dur in bouts:
                    trace = segment_signal_window(sig, onset, PRE_FRAMES, POST_FRAMES)
                    if np.all(np.isnan(trace)):
                        continue
                    # Baseline subtract: mean of first 2 seconds (30 frames)
                    bl = np.nanmean(trace[:BASELINE_FRAMES])
                    trace = trace - bl
                    result[pathway]['traces'].append(trace)
                    result[pathway]['durations'].append(dur)
                    result[pathway]['mice'].add(mouse)

    # Convert to arrays
    for pw in result:
        if len(result[pw]['traces']) > 0:
            result[pw]['traces'] = np.array(result[pw]['traces'])
            result[pw]['durations'] = np.array(result[pw]['durations'])
        else:
            result[pw]['traces'] = np.empty((0, WINDOW))
            result[pw]['durations'] = np.array([])

    return result


def plot_mean_trace(ax, traces, color, label, title):
    """Plot mean +/- SEM trace with vertical line at bout onset."""
    time_s = (np.arange(WINDOW) - PRE_FRAMES) / FPS
    if traces.shape[0] == 0:
        ax.set_title(title, pad=12)
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
        return
    mean = np.nanmean(traces, axis=0)
    sem = np.nanstd(traces, axis=0) / np.sqrt(traces.shape[0])
    ax.fill_between(time_s, mean - sem, mean + sem, color=color, alpha=0.3)
    ax.plot(time_s, mean, color=color, lw=1.5, label=f'n={traces.shape[0]} bouts')
    ax.axvline(0, color='k', ls='--', lw=0.8, alpha=0.6)
    ax.set_title(title, fontsize=10, pad=12)
    ax.set_xlabel('Time from bout onset (s)', fontsize=10)
    ax.set_ylabel('z-scored dF/F', fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc='upper right')


def plot_heatmap(ax, traces, durations, title, cmap='magma', vmin=-0.5, vmax=1.5):
    """Plot bout-response-triggered (BRT) heatmap, sorted by bout duration."""
    if traces.shape[0] == 0:
        ax.set_title(title, pad=12)
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
        return
    # Sort by duration (longest on top)
    sort_idx = np.argsort(-durations)
    sorted_traces = traces[sort_idx]

    time_s = (np.arange(WINDOW) - PRE_FRAMES) / FPS
    im = ax.imshow(sorted_traces, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[time_s[0], time_s[-1], sorted_traces.shape[0], 0])
    ax.axvline(0, color='w', ls='--', lw=0.8, alpha=0.8)
    ax.set_title(title, fontsize=10, pad=12)
    ax.set_xlabel('Time from bout onset (s)', fontsize=10)
    ax.set_ylabel('Bout (sorted by dur.)', fontsize=10)
    plt.colorbar(im, ax=ax, label='z-scored dF/F', shrink=0.8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    setup_style()

    print('Loading CMT_Dec24...')
    CMT = load_CMT('Dec24')

    print('Collecting grooming bouts...')
    grooming_data = collect_bouts(CMT, GROOMING)

    print('Collecting body licking bouts...')
    bodylick_data = collect_bouts(CMT, BODY_LICKING)

    # Print summary
    for pw_name, pw_key in [('dSPN', 'drd1'), ('iSPN', 'a2a')]:
        n_grm = grooming_data[pw_key]['traces'].shape[0]
        n_bl = bodylick_data[pw_key]['traces'].shape[0]
        mice_grm = grooming_data[pw_key]['mice']
        mice_bl = bodylick_data[pw_key]['mice']
        all_mice = mice_grm | mice_bl
        print(f'  {pw_name}: {n_grm} grooming bouts ({len(mice_grm)} mice), '
              f'{n_bl} body licking bouts ({len(mice_bl)} mice), '
              f'{len(all_mice)} total mice')

    # Build figure: 2 rows x 3 cols
    fig = plt.figure(figsize=(180 * mm, 120 * mm))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.45)

    panel_labels = ['A', 'B', 'C', 'D', 'E', 'F']

    # Row 1: dSPN
    ax_a = fig.add_subplot(gs[0, 0])
    plot_mean_trace(ax_a, grooming_data['drd1']['traces'],
                    dSPN_photom_color, 'dSPN', 'dSPN — Grooming')

    ax_b = fig.add_subplot(gs[0, 1])
    plot_mean_trace(ax_b, bodylick_data['drd1']['traces'],
                    dSPN_photom_color, 'dSPN', 'dSPN — Body Licking')

    ax_c = fig.add_subplot(gs[0, 2])
    plot_heatmap(ax_c, grooming_data['drd1']['traces'],
                 grooming_data['drd1']['durations'], 'dSPN — Grooming BRT')

    # Row 2: iSPN
    ax_d = fig.add_subplot(gs[1, 0])
    plot_mean_trace(ax_d, grooming_data['a2a']['traces'],
                    iSPN_photom_color, 'iSPN', 'iSPN — Grooming')

    ax_e = fig.add_subplot(gs[1, 1])
    plot_mean_trace(ax_e, bodylick_data['a2a']['traces'],
                    iSPN_photom_color, 'iSPN', 'iSPN — Body Licking')

    ax_f = fig.add_subplot(gs[1, 2])
    plot_heatmap(ax_f, grooming_data['a2a']['traces'],
                 grooming_data['a2a']['durations'], 'iSPN — Grooming BRT')

    # Panel labels
    axes = [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f]
    for ax, lbl in zip(axes, panel_labels):
        ax.text(-0.15, 1.15, lbl, transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top', ha='left')

    save_fig(fig, OUTPUT_DIR, 'SuppFig13_Grooming_Photometry')
    print('Done.')


if __name__ == '__main__':
    main()

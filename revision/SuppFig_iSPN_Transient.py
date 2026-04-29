#%% SuppFig: Transient iSPN Photometry Effects (Early vs Late Cocaine)
"""
Addresses R4-FigS2HM: Show transient iSPN photometry effects in early vs late
cocaine sessions.

Panels:
  A. iSPN peak prominence CDFs: saline vs C1-C2 vs C3-C5
  B. iSPN IEI CDFs: saline vs C1-C2 vs C3-C5
  C. dSPN peak prominence CDFs: saline vs C1-C2 vs C3-C5
  D. dSPN IEI CDFs: saline vs C1-C2 vs C3-C5
  E. Prominence per-session means (line plot, both pathways)
  F. IEI per-session means (line plot, both pathways)

Uses CMT_Dec24.pkl with z-scored photometry signals.
Peak detection via scipy.signal.find_peaks(prominence=2, width=5).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from revision_utils import (
    mm, second, FPS, n_behaviors,
    dSPN_color, iSPN_color,
    load_CMT, get_pathway, get_photom_signal, photom_exclusions,
    setup_style, save_fig
)

setup_style()
output_folder = 'revision/output/'
os.makedirs(output_folder, exist_ok=True)

# ---------------------------------------------------------------------------
# Photometry mice by pathway
# ---------------------------------------------------------------------------
iSPN_photom_mice = {
    'a2a_hm3dq': ['cA184m4', 'cA184m7', 'cA242m5', 'cA242m6'],  # cA242m8 excluded (corrupted)
    'controls': ['cA242m4', 'cA242m9'],                           # cA-prefix controls -> a2a
}

dSPN_photom_mice = {
    'drd1_hm4di': ['c528m5', 'c528m10', 'c548m1'],
    'controls': ['c548m8', 'c548m10', 'c548m11'],                 # non-cA controls -> drd1
}

# Trial groupings
saline_trials = ['saline1', 'saline2', 'saline3']
early_cocaine = ['cocaine1', 'cocaine2']
late_cocaine = ['cocaine3', 'cocaine4', 'cocaine5']
all_trials = saline_trials + ['cocaine1', 'cocaine2', 'cocaine3', 'cocaine4', 'cocaine5']

# Colors for CDF conditions
saline_color = '#808080'
early_color = '#FFA500'
late_color = '#990000'

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print('Loading CMT_Dec24...')
CMT = load_CMT('Dec24')

# ---------------------------------------------------------------------------
# Build exclusion lookup sets for fast checking
# ---------------------------------------------------------------------------
excl_mouse = set((c, m) for c, m in photom_exclusions['remove_mouse'])
excl_hemi = set((c, m, h) for c, m, h in photom_exclusions['remove_hemisphere'])
excl_trial = set((c, m, t) for c, m, t in photom_exclusions['remove_trial'])

def is_excluded_mouse(cohort, mouse):
    return (cohort, mouse) in excl_mouse

def is_excluded_trial(cohort, mouse, trial):
    return (cohort, mouse, trial) in excl_trial

def is_excluded_hemi(cohort, mouse, hemi):
    return (cohort, mouse, hemi) in excl_hemi

# ---------------------------------------------------------------------------
# Extract peaks from photometry signals
# ---------------------------------------------------------------------------
def extract_peak_features(signal):
    """Detect peaks and return (prominences, widths, IEIs) arrays."""
    peaks, props = scipy.signal.find_peaks(signal, prominence=2, width=5)
    if len(peaks) == 0:
        return np.array([]), np.array([]), np.array([])
    prominences = props['prominences']
    widths = props['widths'] / FPS  # convert to seconds
    if len(peaks) >= 2:
        ieis = np.diff(peaks) / FPS  # inter-event intervals in seconds
    else:
        ieis = np.array([])
    return prominences, widths, ieis


def collect_features(mice_dict, pathway_name):
    """Collect peak features across mice/trials for a given pathway.
    Returns:
      by_condition: dict condition_name -> {'prom': [], 'iei': []}
      by_session: dict trial_name -> {'prom_means': [], 'iei_means': []}
                  where each list has one value per mouse-hemisphere
    """
    by_condition = {
        'saline': {'prom': [], 'iei': []},
        'early': {'prom': [], 'iei': []},
        'late': {'prom': [], 'iei': []},
    }
    by_session = {t: {'prom_means': [], 'iei_means': []} for t in all_trials}

    n_mice = 0
    n_peaks_total = 0

    for cohort, mice_list in mice_dict.items():
        if cohort not in CMT:
            print(f'  WARNING: cohort {cohort} not found in CMT')
            continue
        for mouse in mice_list:
            if mouse not in CMT[cohort]:
                print(f'  WARNING: mouse {mouse} not found in {cohort}')
                continue
            if is_excluded_mouse(cohort, mouse):
                print(f'  Excluding mouse {mouse} (corrupted)')
                continue

            n_mice += 1

            for trial in all_trials:
                if trial not in CMT[cohort][mouse]:
                    continue
                if is_excluded_trial(cohort, mouse, trial):
                    continue

                entry = CMT[cohort][mouse][trial]
                signals = get_photom_signal(entry, side='both', sig_type='z')

                for sig, side_name in signals:
                    if is_excluded_hemi(cohort, mouse, side_name):
                        continue

                    prom, wid, iei = extract_peak_features(sig)
                    n_peaks_total += len(prom)

                    # Determine condition
                    if trial in saline_trials:
                        cond = 'saline'
                    elif trial in early_cocaine:
                        cond = 'early'
                    else:
                        cond = 'late'

                    by_condition[cond]['prom'].extend(prom)
                    by_condition[cond]['iei'].extend(iei)

                    # Per-session means
                    if len(prom) > 0:
                        by_session[trial]['prom_means'].append(np.mean(prom))
                    else:
                        by_session[trial]['prom_means'].append(np.nan)
                    if len(iei) > 0:
                        by_session[trial]['iei_means'].append(np.mean(iei))
                    else:
                        by_session[trial]['iei_means'].append(np.nan)

    print(f'  {pathway_name}: N={n_mice} mice, {n_peaks_total} total peaks')
    return by_condition, by_session


print('\nCollecting iSPN features...')
iSPN_cond, iSPN_sess = collect_features(iSPN_photom_mice, 'iSPN')

print('Collecting dSPN features...')
dSPN_cond, dSPN_sess = collect_features(dSPN_photom_mice, 'dSPN')

# ---------------------------------------------------------------------------
# Print summary statistics
# ---------------------------------------------------------------------------
for pathway, cond_data in [('iSPN', iSPN_cond), ('dSPN', dSPN_cond)]:
    print(f'\n--- {pathway} Peak Features ---')
    for cond_name in ['saline', 'early', 'late']:
        proms = np.array(cond_data[cond_name]['prom'])
        ieis = np.array(cond_data[cond_name]['iei'])
        print(f'  {cond_name}: {len(proms)} peaks, '
              f'prom={np.mean(proms):.2f}+/-{np.std(proms)/max(np.sqrt(len(proms)),1):.2f} '
              f'IEI={np.nanmean(ieis):.2f}+/-{np.nanstd(ieis)/max(np.sqrt(len(ieis)),1):.2f}s'
              if len(proms) > 0 else f'  {cond_name}: no peaks')

# ===========================================================================
# BUILD THE FIGURE
# ===========================================================================
print('\nGenerating figure...')
fig = plt.figure(figsize=(180 * mm, 170 * mm))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.65, wspace=0.35)

# ---------------------------------------------------------------------------
# Helper: plot CDF
# ---------------------------------------------------------------------------
def plot_cdf(ax, data_dict, metric, title, xlabel, xlim=None):
    """Plot step CDFs for saline/early/late conditions."""
    cond_info = [
        ('saline', 'Saline', saline_color, '-'),
        ('early', 'C1-C2', early_color, '-'),
        ('late', 'C3-C5', late_color, '-'),
    ]
    for cond_key, label, color, ls in cond_info:
        vals = np.array(data_dict[cond_key][metric])
        if len(vals) == 0:
            continue
        sorted_v = np.sort(vals)
        cdf_y = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
        ax.step(sorted_v, cdf_y, color=color, lw=1.2, ls=ls,
                label=f'{label} (n={len(vals)})')
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel('CDF', fontsize=8)
    ax.set_title(title, fontsize=8, fontweight='bold', pad=12)
    ax.legend(fontsize=6, loc='lower right')
    if xlim is not None:
        ax.set_xlim(xlim)

# --- Panel A: iSPN prominence CDFs ---
ax_a = fig.add_subplot(gs[0, 0])
plot_cdf(ax_a, iSPN_cond, 'prom', 'iSPN Peak Prominence', 'Prominence (z)', xlim=(0, 15))
ax_a.text(-0.15, 1.15, 'A', transform=ax_a.transAxes, fontsize=12, fontweight='bold', va='top')

# --- Panel B: iSPN IEI CDFs ---
ax_b = fig.add_subplot(gs[0, 1])
plot_cdf(ax_b, iSPN_cond, 'iei', 'iSPN Inter-Event Interval', 'IEI (s)', xlim=(0, 60))
ax_b.text(-0.15, 1.15, 'B', transform=ax_b.transAxes, fontsize=12, fontweight='bold', va='top')

# --- Panel C: dSPN prominence CDFs ---
ax_c = fig.add_subplot(gs[1, 0])
plot_cdf(ax_c, dSPN_cond, 'prom', 'dSPN Peak Prominence', 'Prominence (z)', xlim=(0, 15))
ax_c.text(-0.15, 1.15, 'C', transform=ax_c.transAxes, fontsize=12, fontweight='bold', va='top')

# --- Panel D: dSPN IEI CDFs ---
ax_d = fig.add_subplot(gs[1, 1])
plot_cdf(ax_d, dSPN_cond, 'iei', 'dSPN Inter-Event Interval', 'IEI (s)', xlim=(0, 60))
ax_d.text(-0.15, 1.15, 'D', transform=ax_d.transAxes, fontsize=12, fontweight='bold', va='top')

# ---------------------------------------------------------------------------
# Helper: session line plot
# ---------------------------------------------------------------------------
def plot_session_line(ax, iSPN_sess, dSPN_sess, metric_key, ylabel, title):
    """Line plot of per-session means with SEM for both pathways."""
    x = np.arange(len(all_trials))
    x_labels = ['S1', 'S2', 'S3', 'C1', 'C2', 'C3', 'C4', 'C5']

    for sess_data, color, label, ls in [
        (iSPN_sess, iSPN_color, 'iSPN', '--'),
        (dSPN_sess, dSPN_color, 'dSPN', '-'),
    ]:
        means = []
        sems = []
        ns = []
        for trial in all_trials:
            vals = np.array(sess_data[trial][metric_key], dtype=float)
            valid = vals[~np.isnan(vals)]
            n = len(valid)
            ns.append(n)
            if n > 0:
                means.append(np.mean(valid))
                sems.append(np.sqrt(np.nanvar(valid)) / max(np.sqrt(n), 1))
            else:
                means.append(np.nan)
                sems.append(0)
        means = np.array(means)
        sems = np.array(sems)
        n_display = max(ns) if ns else 0

        ax.errorbar(x, means, yerr=sems, color=color, lw=1.2,
                    capsize=2, marker='o', markersize=3, ls=ls,
                    label=f'{label} (N={n_display})')

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=6)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=8, fontweight='bold', pad=12)
    ax.legend(fontsize=6, loc='lower right')

    # Shade cocaine region
    ax.axvspan(2.5, 7.5, alpha=0.05, color='red', zorder=0)

# --- Panel E: Prominence per-session means ---
ax_e = fig.add_subplot(gs[2, 0])
plot_session_line(ax_e, iSPN_sess, dSPN_sess, 'prom_means',
                  'Mean prominence (z)', 'Peak Prominence Across Sessions')
ax_e.text(-0.15, 1.15, 'E', transform=ax_e.transAxes, fontsize=12, fontweight='bold', va='top')

# --- Panel F: IEI per-session means ---
ax_f = fig.add_subplot(gs[2, 1])
plot_session_line(ax_f, iSPN_sess, dSPN_sess, 'iei_means',
                  'Mean IEI (s)', 'Inter-Event Interval Across Sessions')
ax_f.text(-0.15, 1.15, 'F', transform=ax_f.transAxes, fontsize=12, fontweight='bold', va='top')

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
save_fig(fig, output_folder, 'SuppFig8_iSPN_Transient')
print('\nDone.')

#%% SuppFig: Photometry Amplitude vs Bout Duration Correlation
"""
Addresses R4-Fig2L: Is the dSPN vs iSPN activation difference explained by
bout duration?

Uses master_table.pkl which contains pre-segmented photometry traces aligned
to behavior bout onsets. For each bout, peak amplitude is measured in the
post-onset period (frames 45+, i.e. after the 3s pre-switch window).

Panels:
  A: Scatter — bout duration vs peak amplitude for dSPN (up_regulated bouts)
  B: Scatter — bout duration vs peak amplitude for iSPN (up_regulated bouts)
  C: Duration-matched comparison — bin bouts by duration, compare mean amplitude
  D: Box plot — peak amplitude dSPN vs iSPN across all cocaine bouts

Pools cocaine3, cocaine4, cocaine5 trials.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from revision_utils import (
    mm, FPS, second,
    dSPN_color, iSPN_color,
    load_master_table, setup_style, save_fig
)

setup_style()
output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(output_folder, exist_ok=True)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
pre_switch_window = 45  # frames (3s at 15 FPS) — onset is at this index
target_trials = ['cocaine3', 'cocaine4', 'cocaine5']
regulation = 'up_regulated'

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print('Loading master_table...')
master_table = load_master_table()

# ---------------------------------------------------------------------------
# Extract bout duration and peak amplitude per pathway
# ---------------------------------------------------------------------------
def extract_bout_data(master_table, pathway, trials, regulation):
    """Extract (duration_s, peak_amplitude, mouse) for each bout."""
    durations = []
    amplitudes = []
    mice = []
    for trial in trials:
        if trial not in master_table:
            continue
        if pathway not in master_table[trial]:
            continue
        if regulation not in master_table[trial][pathway]:
            continue
        entry = master_table[trial][pathway][regulation]
        bout_durations = entry['duration']
        signals = entry['signal']
        mouse_ids = entry['mouse']
        for i in range(len(signals)):
            trace = np.asarray(signals[i], dtype=float)
            dur_frames = bout_durations[i]
            dur_s = dur_frames / FPS
            # Peak amplitude in post-onset period
            if len(trace) > pre_switch_window:
                peak = np.nanmax(trace[pre_switch_window:])
            else:
                peak = np.nanmax(trace)
            if np.isfinite(peak) and np.isfinite(dur_s):
                durations.append(dur_s)
                amplitudes.append(peak)
                mice.append(mouse_ids[i])
    return np.array(durations), np.array(amplitudes), mice

print('Extracting dSPN bouts...')
dSPN_dur, dSPN_amp, dSPN_mice = extract_bout_data(master_table, 'dSPN', target_trials, regulation)
print(f'  dSPN: {len(dSPN_dur)} bouts from {len(set(dSPN_mice))} mice')

print('Extracting iSPN bouts...')
iSPN_dur, iSPN_amp, iSPN_mice = extract_bout_data(master_table, 'iSPN', target_trials, regulation)
print(f'  iSPN: {len(iSPN_dur)} bouts from {len(set(iSPN_mice))} mice')

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
print('\n--- Pearson correlation: duration vs amplitude ---')
if len(dSPN_dur) > 2:
    r_d, p_d = stats.pearsonr(dSPN_dur, dSPN_amp)
    print(f'  dSPN: r={r_d:.3f}, p={p_d:.4f} (n={len(dSPN_dur)} bouts)')
else:
    r_d, p_d = np.nan, np.nan
    print(f'  dSPN: insufficient data (n={len(dSPN_dur)})')

if len(iSPN_dur) > 2:
    r_i, p_i = stats.pearsonr(iSPN_dur, iSPN_amp)
    print(f'  iSPN: r={r_i:.3f}, p={p_i:.4f} (n={len(iSPN_dur)} bouts)')
else:
    r_i, p_i = np.nan, np.nan
    print(f'  iSPN: insufficient data (n={len(iSPN_dur)})')

print('\n--- Mann-Whitney U: peak amplitude dSPN vs iSPN ---')
if len(dSPN_amp) > 2 and len(iSPN_amp) > 2:
    u_stat, p_mw = stats.mannwhitneyu(dSPN_amp, iSPN_amp, alternative='two-sided')
    print(f'  dSPN median={np.median(dSPN_amp):.3f}, iSPN median={np.median(iSPN_amp):.3f}')
    print(f'  U={u_stat:.0f}, p={p_mw:.4f}')
else:
    u_stat, p_mw = np.nan, np.nan
    print('  Insufficient data for comparison')

# ---------------------------------------------------------------------------
# Duration-matched binning
# ---------------------------------------------------------------------------
# Use shared bin edges across both pathways
all_durs = np.concatenate([dSPN_dur, iSPN_dur]) if len(dSPN_dur) > 0 and len(iSPN_dur) > 0 else np.array([])
if len(all_durs) > 0:
    bin_edges = np.percentile(all_durs, [0, 25, 50, 75, 100])
    # Ensure unique edges
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 3:
        bin_edges = np.linspace(all_durs.min(), all_durs.max(), 5)
else:
    bin_edges = np.array([0, 1, 2, 3, 4])

n_bins = len(bin_edges) - 1

def bin_amplitudes(durations, amplitudes, edges):
    """Return mean amplitude per duration bin."""
    means = []
    sems = []
    ns = []
    for i in range(len(edges) - 1):
        mask = (durations >= edges[i]) & (durations < edges[i + 1])
        if i == len(edges) - 2:  # include right edge for last bin
            mask = (durations >= edges[i]) & (durations <= edges[i + 1])
        vals = amplitudes[mask]
        if len(vals) > 0:
            means.append(np.nanmean(vals))
            sems.append(np.nanstd(vals, ddof=0) / np.sqrt(len(vals)))
            ns.append(len(vals))
        else:
            means.append(np.nan)
            sems.append(np.nan)
            ns.append(0)
    return np.array(means), np.array(sems), ns

dSPN_bin_mean, dSPN_bin_sem, dSPN_bin_n = bin_amplitudes(dSPN_dur, dSPN_amp, bin_edges)
iSPN_bin_mean, iSPN_bin_sem, iSPN_bin_n = bin_amplitudes(iSPN_dur, iSPN_amp, bin_edges)

bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(n_bins)]
bin_labels = [f'{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}s' for i in range(n_bins)]

print('\n--- Duration-matched amplitude comparison ---')
for i in range(n_bins):
    print(f'  Bin {bin_labels[i]}: dSPN={dSPN_bin_mean[i]:.3f} (n={dSPN_bin_n[i]}), '
          f'iSPN={iSPN_bin_mean[i]:.3f} (n={iSPN_bin_n[i]})')

# ===========================================================================
# BUILD THE FIGURE
# ===========================================================================
print('\nGenerating figure...')
fig = plt.figure(figsize=(180 * mm, 100 * mm))
gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.45)

# --- Panel A: dSPN scatter ---
ax_a = fig.add_subplot(gs[0])
if len(dSPN_dur) > 0:
    ax_a.scatter(dSPN_dur, dSPN_amp, s=8, alpha=0.4, color=dSPN_color, edgecolors='none')
    # Regression line
    if len(dSPN_dur) > 2:
        z = np.polyfit(dSPN_dur, dSPN_amp, 1)
        x_fit = np.linspace(dSPN_dur.min(), dSPN_dur.max(), 100)
        ax_a.plot(x_fit, np.polyval(z, x_fit), color=dSPN_color, lw=1.2, ls='--')
ax_a.set_xlabel('Bout duration (s)')
ax_a.set_ylabel('Peak amplitude (z)')
ax_a.set_title(f'dSPN (n={len(dSPN_dur)})', fontsize=8, fontweight='bold', pad=12)
if np.isfinite(r_d):
    ax_a.text(0.95, 0.95, f'r={r_d:.2f}\np={p_d:.3f}',
              transform=ax_a.transAxes, fontsize=7, ha='right', va='top')
ax_a.text(-0.2, 1.15, 'A', transform=ax_a.transAxes, fontsize=10, fontweight='bold', va='top')

# --- Panel B: iSPN scatter ---
ax_b = fig.add_subplot(gs[1])
if len(iSPN_dur) > 0:
    ax_b.scatter(iSPN_dur, iSPN_amp, s=8, alpha=0.4, color=iSPN_color, edgecolors='none')
    if len(iSPN_dur) > 2:
        z = np.polyfit(iSPN_dur, iSPN_amp, 1)
        x_fit = np.linspace(iSPN_dur.min(), iSPN_dur.max(), 100)
        ax_b.plot(x_fit, np.polyval(z, x_fit), color=iSPN_color, lw=1.2, ls='--')
ax_b.set_xlabel('Bout duration (s)')
ax_b.set_ylabel('Peak amplitude (z)')
ax_b.set_title(f'iSPN (n={len(iSPN_dur)})', fontsize=8, fontweight='bold', pad=12)
if np.isfinite(r_i):
    ax_b.text(0.95, 0.95, f'r={r_i:.2f}\np={p_i:.3f}',
              transform=ax_b.transAxes, fontsize=7, ha='right', va='top')
ax_b.text(-0.2, 1.15, 'B', transform=ax_b.transAxes, fontsize=10, fontweight='bold', va='top')

# --- Panel C: Duration-matched comparison ---
ax_c = fig.add_subplot(gs[2])
bar_width = 0.35
x_pos = np.arange(n_bins)
ax_c.bar(x_pos - bar_width / 2, dSPN_bin_mean, bar_width, yerr=dSPN_bin_sem,
         color=dSPN_color, alpha=0.8, capsize=3, label='dSPN')
ax_c.bar(x_pos + bar_width / 2, iSPN_bin_mean, bar_width, yerr=iSPN_bin_sem,
         color=iSPN_color, alpha=0.8, capsize=3, label='iSPN')
ax_c.set_xticks(x_pos)
ax_c.set_xticklabels(bin_labels, fontsize=6, rotation=30, ha='right')
ax_c.set_xlabel('Duration bin')
ax_c.set_ylabel('Peak amplitude (z)')
ax_c.set_title('Duration-matched', fontsize=8, fontweight='bold', pad=12)
ax_c.legend(fontsize=6, loc='lower right')
ax_c.text(-0.2, 1.15, 'C', transform=ax_c.transAxes, fontsize=10, fontweight='bold', va='top')

# --- Panel D: Box plot comparison ---
ax_d = fig.add_subplot(gs[3])
data_box = []
positions = []
box_colors = []
labels_box = []
if len(dSPN_amp) > 0:
    data_box.append(dSPN_amp)
    positions.append(1)
    box_colors.append(dSPN_color)
    labels_box.append('dSPN')
if len(iSPN_amp) > 0:
    data_box.append(iSPN_amp)
    positions.append(2)
    box_colors.append(iSPN_color)
    labels_box.append('iSPN')

if len(data_box) > 0:
    bp = ax_d.boxplot(data_box, positions=positions, widths=0.5, patch_artist=True,
                      showfliers=False, medianprops=dict(color='black', lw=1.2))
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    # Overlay individual points with jitter
    for i, (d, pos) in enumerate(zip(data_box, positions)):
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(d))
        ax_d.scatter(np.full(len(d), pos) + jitter, d, s=4, alpha=0.3,
                     color=box_colors[i], edgecolors='none', zorder=3)

ax_d.set_xticks(positions)
ax_d.set_xticklabels(labels_box)
ax_d.set_ylabel('Peak amplitude (z)')
ax_d.set_title('All cocaine bouts', fontsize=8, fontweight='bold', pad=12)
if np.isfinite(p_mw):
    sig_str = f'p={p_mw:.3f}' if p_mw >= 0.001 else f'p={p_mw:.1e}'
    y_max = max(np.percentile(dSPN_amp, 95) if len(dSPN_amp) > 0 else 0,
                np.percentile(iSPN_amp, 95) if len(iSPN_amp) > 0 else 0) * 1.1
    ax_d.plot([1, 1, 2, 2], [y_max, y_max * 1.03, y_max * 1.03, y_max], color='black', lw=0.8)
    ax_d.text(1.5, y_max * 1.05, sig_str, fontsize=7, ha='center', va='bottom')
ax_d.text(-0.2, 1.15, 'D', transform=ax_d.transAxes, fontsize=10, fontweight='bold', va='top')

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
save_fig(fig, output_folder, 'SuppFig10_Photom_BoutCorrelation')
print('\nDone.')

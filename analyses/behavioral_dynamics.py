#%% Imports and constants
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import copy
import numpy as np
import bz2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import scipy.stats as stats
import os

root_folder = '.'
folder = 'data/'

# Load data
ifile = bz2.BZ2File(folder + 'CMT_Aug24.pkl', 'rb')
CMT = pickle.load(ifile)
ifile.close()
print('***All dictionaries were loaded***')

behaviors = ['Jump', 'Undefined', 'Floor licking', 'Wall licking', 'Grooming',
             'Body licking', 'Rearing', 'Locomotion', 'Stationary']
colors = ["#696969", "#d3d3d3", "#d73027", "#e57373", "#c4a7e7",
          "#8e63b8", "#b3e5fc", "#3399cc", "#1f4e79"]
lut = np.array([4, 5, 3, 2, 6, 1, 8, 7, 0])

FPS = 15
second = 15
minute = 60 * second
mm = 1 / 25.4
num_of_behaviors = len(behaviors)

trials = ['saline1', 'saline2', 'saline3', 'cocaine1', 'cocaine2',
          'cocaine3', 'cocaine4', 'cocaine5']
cohorts = ['drd1_hm4di', 'drd1_hm3dq', 'controls', 'a2a_hm4di',
           'a2a_hm3dq', 'a2a_opto']
num_of_trials = len(trials)

# Remap predictions via lut, keeping cohort→mouse mapping
MT = {}
mouse_to_cohort = {}
for c in cohorts:
    for m in CMT[c]:
        MT[m] = copy.deepcopy(CMT[c][m])
        mouse_to_cohort[m] = c
        for t in MT[m]:
            MT[m][t]['merged'] = lut[MT[m][t]['merged']]
del CMT

#%% Metric functions

def switch_entropy(preds, num_beh):
    """Shannon entropy of the destination-behavior distribution at switches."""
    switches = np.where(np.diff(preds.astype(int)) != 0)[0] + 1
    if len(switches) < 2:
        return np.nan
    destinations = preds[switches]
    counts = np.bincount(destinations.astype(int), minlength=num_beh)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def n_distinct_behaviors(preds, num_beh):
    """Number of distinct behaviors observed in the window."""
    return len(np.unique(preds.astype(int)))


def transition_rate(preds, num_beh):
    """Number of behavioral switches per minute."""
    n_switches = np.sum(np.diff(preds.astype(int)) != 0)
    duration_min = len(preds) / (FPS * 60)
    if duration_min == 0:
        return np.nan
    return n_switches / duration_min


def transition_matrix(preds, num_beh):
    """Row-normalised transition probability matrix (9×9).
    Entry [i,j] = P(switch to j | currently in i), computed only at switch frames.
    """
    preds = preds.astype(int)
    mat = np.zeros((num_beh, num_beh))
    for k in range(len(preds) - 1):
        if preds[k] != preds[k + 1]:
            mat[preds[k], preds[k + 1]] += 1
    # Row-normalise
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    return mat / row_sums


#%% Identify valid mice (complete data across all 8 sessions)
mice_list = list(MT.keys())
valid_mask = np.array([
    all(trial in MT[m] for trial in trials)
    for m in mice_list
])
valid_mice = [mice_list[i] for i in range(len(mice_list)) if valid_mask[i]]
n_mice = len(valid_mice)
print(f'N = {n_mice} mice with complete data')

#%% Within-session time courses for three metrics
window_frames = 1 * minute
step_frames = 30 * second
max_duration = 27 * minute

trace_trials = ['saline3', 'cocaine1', 'cocaine2', 'cocaine3', 'cocaine4', 'cocaine5']
trace_colors = ['#808080', '#FFA500', '#FF8C00', '#FF6347', '#E60000', '#990000']

n_windows = int((max_duration - window_frames) / step_frames) + 1
window_centers_min = np.array([(w * step_frames + window_frames / 2) / minute
                                for w in range(n_windows)])

metrics = {
    'switch_entropy': switch_entropy,
    'n_distinct': n_distinct_behaviors,
    'transition_rate': transition_rate,
}

# Pre-compute all traces: metric → trial → (n_mice, n_windows)
all_traces = {}
for metric_name, metric_fn in metrics.items():
    all_traces[metric_name] = {}
    for trial in trace_trials:
        traces = np.full((n_mice, n_windows), np.nan)
        for m_idx, mouse in enumerate(valid_mice):
            if trial in MT[mouse]:
                preds = MT[mouse][trial]['merged']
                for w in range(n_windows):
                    start = w * step_frames
                    end = start + window_frames
                    if end <= len(preds):
                        traces[m_idx, w] = metric_fn(preds[start:end],
                                                      num_of_behaviors)
        all_traces[metric_name][trial] = traces

print('Within-session traces computed')

#%% Transition probability matrices: baseline vs late cocaine
# Baseline = saline1-3 pooled; Late cocaine = cocaine3-5 pooled
baseline_trials = ['saline1', 'saline2', 'saline3']
late_cocaine_trials = ['cocaine3', 'cocaine4', 'cocaine5']

baseline_matrices = []
late_matrices = []

for mouse in valid_mice:
    for trial in baseline_trials:
        if trial in MT[mouse]:
            preds = MT[mouse][trial]['merged']
            baseline_matrices.append(transition_matrix(preds, num_of_behaviors))
    for trial in late_cocaine_trials:
        if trial in MT[mouse]:
            preds = MT[mouse][trial]['merged']
            late_matrices.append(transition_matrix(preds, num_of_behaviors))

mean_baseline_mat = np.mean(baseline_matrices, axis=0)
mean_late_mat = np.mean(late_matrices, axis=0)
diff_mat = mean_late_mat - mean_baseline_mat

print('Transition matrices computed')

#%% Across-session summary stats for each metric
print('\n--- Across-session summary (whole-session values) ---')
for metric_name, metric_fn in metrics.items():
    values = np.full((n_mice, num_of_trials), np.nan)
    for m_idx, mouse in enumerate(valid_mice):
        for t_idx, trial in enumerate(trials):
            if trial in MT[mouse]:
                preds = MT[mouse][trial]['merged']
                values[m_idx, t_idx] = metric_fn(preds, num_of_behaviors)

    session_means = np.nanmean(values, axis=0)
    rho, p_rho = stats.spearmanr(np.arange(num_of_trials), session_means)

    saline_vals = np.nanmean(values[:, :3], axis=1)
    late_vals = np.nanmean(values[:, 5:8], axis=1)
    t_stat, p_tt = stats.ttest_rel(saline_vals, late_vals)
    d = (np.mean(saline_vals - late_vals) /
         np.std(saline_vals - late_vals, ddof=1))

    print(f'\n{metric_name}:')
    print(f'  Saline mean: {np.nanmean(saline_vals):.3f} ± '
          f'{np.sqrt(np.nanvar(saline_vals) / n_mice):.3f}')
    print(f'  Late cocaine mean: {np.nanmean(late_vals):.3f} ± '
          f'{np.sqrt(np.nanvar(late_vals) / n_mice):.3f}')
    print(f'  Spearman: rho={rho:.3f}, p={p_rho:.4f}')
    print(f'  Paired t-test: t({n_mice-1})={t_stat:.3f}, p={p_tt:.4f}, d={d:.2f}')

#%% Build multi-panel figure
output_folder = root_folder + '/Figures/Behavioral_dynamics/'
os.makedirs(output_folder, exist_ok=True)

fig = plt.figure(figsize=(180 * mm, 155 * mm))
gs = gridspec.GridSpec(2, 3, figure=fig,
                       height_ratios=[1, 1.1],
                       hspace=0.45, wspace=0.35)

# ---- Helper to plot within-session time course ----
def plot_time_course(ax, metric_name, ylabel, panel_label):
    for t_idx, trial in enumerate(trace_trials):
        traces = all_traces[metric_name][trial]
        valid_n = np.sum(~np.isnan(traces), axis=0)
        valid_n = np.maximum(valid_n, 1)
        mean_trace = np.nanmean(traces, axis=0)
        sem_trace = np.sqrt(np.nanvar(traces, axis=0, ddof=0) / valid_n)

        ax.plot(window_centers_min, mean_trace,
                color=trace_colors[t_idx], lw=1, label=trial, zorder=5)
        ax.fill_between(window_centers_min,
                        mean_trace - sem_trace, mean_trace + sem_trace,
                        color=trace_colors[t_idx], alpha=0.2, zorder=2)

    ax.set_xlabel('Time (min)', fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.text(-0.15, 1.05, panel_label, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')

# ---- Panel A: Switch entropy ----
ax_a = fig.add_subplot(gs[0, 0])
plot_time_course(ax_a, 'switch_entropy', 'Switch entropy (bits)', 'A')
ax_a.legend(fontsize=6, frameon=False, loc='upper right')

# ---- Panel B: Distinct behaviors ----
ax_b = fig.add_subplot(gs[0, 1])
plot_time_course(ax_b, 'n_distinct', 'Distinct behaviors', 'B')

# ---- Panel C: Transition rate ----
ax_c = fig.add_subplot(gs[0, 2])
plot_time_course(ax_c, 'transition_rate', 'Switches / min', 'C')

# ---- Panel D & E: Transition matrices ----
# Short behavior labels for matrix display
beh_short = ['Jmp', 'Undef', 'FlrLk', 'WllLk', 'Groom',
             'BdyLk', 'Rear', 'Loco', 'Stat']

# D: Baseline
ax_d = fig.add_subplot(gs[1, 0])
im_d = ax_d.imshow(mean_baseline_mat, cmap='YlOrRd', vmin=0, vmax=0.5,
                    aspect='equal')
ax_d.set_xticks(range(num_of_behaviors))
ax_d.set_xticklabels(beh_short, fontsize=5.5, rotation=45, ha='right')
ax_d.set_yticks(range(num_of_behaviors))
ax_d.set_yticklabels(beh_short, fontsize=5.5)
ax_d.set_title('Baseline (saline)', fontsize=8)
ax_d.text(-0.15, 1.05, 'D', transform=ax_d.transAxes,
          fontsize=12, fontweight='bold', va='top')
# Annotate cells
for i in range(num_of_behaviors):
    for j in range(num_of_behaviors):
        val = mean_baseline_mat[i, j]
        if val > 0.01:
            txt_color = 'white' if val > 0.3 else 'black'
            ax_d.text(j, i, f'{val:.2f}', ha='center', va='center',
                      fontsize=4.5, color=txt_color)

# E: Late cocaine
ax_e = fig.add_subplot(gs[1, 1])
im_e = ax_e.imshow(mean_late_mat, cmap='YlOrRd', vmin=0, vmax=0.5,
                    aspect='equal')
ax_e.set_xticks(range(num_of_behaviors))
ax_e.set_xticklabels(beh_short, fontsize=5.5, rotation=45, ha='right')
ax_e.set_yticks(range(num_of_behaviors))
ax_e.set_yticklabels(beh_short, fontsize=5.5)
ax_e.set_title('Late cocaine (C3–C5)', fontsize=8)
ax_e.text(-0.15, 1.05, 'E', transform=ax_e.transAxes,
          fontsize=12, fontweight='bold', va='top')
for i in range(num_of_behaviors):
    for j in range(num_of_behaviors):
        val = mean_late_mat[i, j]
        if val > 0.01:
            txt_color = 'white' if val > 0.3 else 'black'
            ax_e.text(j, i, f'{val:.2f}', ha='center', va='center',
                      fontsize=4.5, color=txt_color)

# F: Difference matrix (late cocaine − baseline)
ax_f = fig.add_subplot(gs[1, 2])
max_abs = np.max(np.abs(diff_mat))
im_f = ax_f.imshow(diff_mat, cmap='RdBu_r', vmin=-max_abs, vmax=max_abs,
                    aspect='equal')
ax_f.set_xticks(range(num_of_behaviors))
ax_f.set_xticklabels(beh_short, fontsize=5.5, rotation=45, ha='right')
ax_f.set_yticks(range(num_of_behaviors))
ax_f.set_yticklabels(beh_short, fontsize=5.5)
ax_f.set_title('Difference (cocaine − baseline)', fontsize=8)
ax_f.text(-0.15, 1.05, 'F', transform=ax_f.transAxes,
          fontsize=12, fontweight='bold', va='top')
for i in range(num_of_behaviors):
    for j in range(num_of_behaviors):
        val = diff_mat[i, j]
        if abs(val) > 0.01:
            txt_color = 'white' if abs(val) > 0.15 else 'black'
            ax_f.text(j, i, f'{val:+.2f}', ha='center', va='center',
                      fontsize=4.5, color=txt_color)

# Shared colorbar for D and E
cbar_ax = fig.add_axes([0.02, 0.02, 0.38, 0.015])
fig.colorbar(im_d, cax=cbar_ax, orientation='horizontal')
cbar_ax.tick_params(labelsize=6)
cbar_ax.set_xlabel('P(transition)', fontsize=7)

# Colorbar for F
cbar_ax_f = fig.add_axes([0.68, 0.02, 0.25, 0.015])
fig.colorbar(im_f, cax=cbar_ax_f, orientation='horizontal')
cbar_ax_f.tick_params(labelsize=6)
cbar_ax_f.set_xlabel('ΔP', fontsize=7)

plt.savefig(output_folder + 'behavioral_dynamics_composite.png',
            dpi=300, bbox_inches='tight')
plt.savefig(output_folder + 'behavioral_dynamics_composite.pdf',
            dpi=300, bbox_inches='tight')
plt.close()
print(f'\nComposite figure saved to {output_folder}')

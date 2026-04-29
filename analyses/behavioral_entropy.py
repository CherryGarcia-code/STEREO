#%% Imports and constants
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import copy
import numpy as np
import bz2
import matplotlib.pyplot as plt
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

trials = ['saline1', 'saline2', 'saline3', 'cocaine1', 'cocaine2', 'cocaine3', 'cocaine4', 'cocaine5']
cohorts = ['drd1_hm4di', 'drd1_hm3dq', 'controls', 'a2a_hm4di', 'a2a_hm3dq', 'a2a_opto']
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

#%% Compute Shannon entropy per mouse per session
def shannon_entropy(predictions, num_behaviors):
    """Compute Shannon entropy (bits) of the behavioral distribution."""
    counts = np.bincount(predictions.astype(int), minlength=num_behaviors)
    probs = counts / counts.sum()
    # Remove zero-probability bins to avoid log(0)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

max_entropy = np.log2(num_of_behaviors)  # ~3.17 bits for 9 behaviors

# Collect entropy per mouse per trial
mice_list = list(MT.keys())
entropy_matrix = np.full((len(mice_list), num_of_trials), np.nan)

for m_idx, mouse in enumerate(mice_list):
    for t_idx, trial in enumerate(trials):
        if trial in MT[mouse]:
            preds = MT[mouse][trial]['merged']
            entropy_matrix[m_idx, t_idx] = shannon_entropy(preds, num_of_behaviors)

# Remove mice with any missing sessions
valid_mask = ~np.any(np.isnan(entropy_matrix), axis=1)
entropy_valid = entropy_matrix[valid_mask, :]
valid_mice = [mice_list[i] for i in range(len(mice_list)) if valid_mask[i]]
n_mice = entropy_valid.shape[0]
print(f'N = {n_mice} mice with complete data across all {num_of_trials} sessions')
print(f'Max possible entropy: {max_entropy:.3f} bits')
print(f'Mean entropy saline: {np.nanmean(entropy_valid[:, :3]):.3f} ± {np.sqrt(np.nanvar(np.nanmean(entropy_valid[:, :3], axis=1)) / n_mice):.3f}')
print(f'Mean entropy cocaine5: {np.nanmean(entropy_valid[:, 7]):.3f} ± {np.sqrt(np.nanvar(entropy_valid[:, 7]) / n_mice):.3f}')

#%% Statistics: Spearman correlation of entropy across sessions
# Use per-mouse mean across trials as a trajectory
session_means = np.nanmean(entropy_valid, axis=0)
session_sems = np.sqrt(np.nanvar(entropy_valid, axis=0) / n_mice)

# Spearman on session index vs per-mouse entropy (using all mouse×session data points, paired)
# More appropriate: compute per-mouse slope, then test
per_mouse_rho = []
for m_idx in range(n_mice):
    rho, _ = stats.spearmanr(np.arange(num_of_trials), entropy_valid[m_idx, :])
    per_mouse_rho.append(rho)

# Group-level Spearman on session means
rho_group, p_group = stats.spearmanr(np.arange(num_of_trials), session_means)
print(f'\nGroup-level Spearman: rho = {rho_group:.3f}, p = {p_group:.4f}')

# Paired t-test: mean saline entropy vs mean late cocaine entropy (cocaine3-5)
saline_entropy = np.nanmean(entropy_valid[:, :3], axis=1)
late_cocaine_entropy = np.nanmean(entropy_valid[:, 5:8], axis=1)
t_stat, p_ttest = stats.ttest_rel(saline_entropy, late_cocaine_entropy)
cohens_d = np.mean(saline_entropy - late_cocaine_entropy) / np.std(saline_entropy - late_cocaine_entropy, ddof=1)
print(f'Paired t-test (saline vs late cocaine): t({n_mice-1}) = {t_stat:.3f}, p = {p_ttest:.4f}, d = {cohens_d:.2f}')

#%% Figure: Entropy trajectory across sessions
output_folder = root_folder + '/Figures/Behavioral_entropy/'
os.makedirs(output_folder, exist_ok=True)

# Session gradient colors (saline = gray, cocaine = warm gradient)
session_colors = ['#808080', '#808080', '#808080',
                  '#FFA500', '#FF8C00', '#FF6347', '#E60000', '#990000']

x = np.arange(num_of_trials)

plt.figure(figsize=(85 * mm, 60 * mm), frameon=False)

# Individual mice
for m_idx in range(n_mice):
    plt.plot(x, entropy_valid[m_idx, :], color='gray', alpha=0.15, lw=0.5)

# Group mean ± SEM with colored segments
for t_idx in range(num_of_trials - 1):
    plt.plot(x[t_idx:t_idx + 2], session_means[t_idx:t_idx + 2],
             color=session_colors[t_idx + 1], lw=1.5, zorder=5)

# Error bars at each session
for t_idx in range(num_of_trials):
    plt.errorbar(x[t_idx], session_means[t_idx], yerr=session_sems[t_idx],
                 color=session_colors[t_idx], capsize=2, capthick=1, lw=1, zorder=6,
                 fmt='o', markersize=3, markerfacecolor=session_colors[t_idx],
                 markeredgecolor='white', markeredgewidth=0.5)

# Max entropy reference line
plt.axhline(y=max_entropy, color='gray', ls=':', lw=0.8, alpha=0.5)
plt.text(num_of_trials - 0.5, max_entropy + 0.02, 'max', fontsize=7, color='gray',
         ha='right', va='bottom')

# Spearman annotation
if p_group < 0.001:
    p_str = 'p < 0.001'
else:
    p_str = f'p = {p_group:.3f}'
plt.text(0.02, 0.02, f'ρ = {rho_group:.2f}, {p_str}\nn = {n_mice} mice',
         transform=plt.gca().transAxes, fontsize=7, va='bottom',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='lightgray', alpha=0.8))

# Axes
plt.ylabel('Behavioral entropy (bits)', fontsize=10)
plt.xlabel('Session', fontsize=10)
plt.xticks(x, ['S1', 'S2', 'S3', 'C1', 'C2', 'C3', 'C4', 'C5'], fontsize=8)
plt.yticks(fontsize=8)
plt.ylim([0.5, max_entropy + 0.15])
plt.xlim([-0.3, num_of_trials - 0.7])

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.99], pad=0.1)
plt.savefig(output_folder + 'entropy_across_sessions.png', dpi=300, bbox_inches='tight')
plt.savefig(output_folder + 'entropy_across_sessions.pdf', dpi=300, bbox_inches='tight')
plt.close()
print(f'\nFigure saved to {output_folder}')

#%% Supplementary: per-cohort breakdown
fig, axes = plt.subplots(2, 3, figsize=(140 * mm, 90 * mm), frameon=False, sharex=True, sharey=True)
axes = axes.flatten()

cohort_labels = {'drd1_hm4di': 'Drd1$_{hm4Di}$', 'drd1_hm3dq': 'Drd1$_{hm3Dq}$',
                 'controls': 'Controls', 'a2a_hm4di': 'A2a$_{hm4Di}$',
                 'a2a_hm3dq': 'A2a$_{hm3Dq}$', 'a2a_opto': 'A2a$_{opto}$'}

for c_idx, cohort in enumerate(cohorts):
    ax = axes[c_idx]
    mice_in_cohort = [m for m in mice_list if mouse_to_cohort[m] == cohort]
    c_entropy = np.full((len(mice_in_cohort), num_of_trials), np.nan)
    for m_idx, mouse in enumerate(mice_in_cohort):
        for t_idx, trial in enumerate(trials):
            if trial in MT[mouse]:
                preds = MT[mouse][trial]['merged']
                c_entropy[m_idx, t_idx] = shannon_entropy(preds, num_of_behaviors)

    valid = ~np.any(np.isnan(c_entropy), axis=1)
    c_valid = c_entropy[valid, :]
    n_c = c_valid.shape[0]

    for m_idx in range(n_c):
        ax.plot(x, c_valid[m_idx, :], color='gray', alpha=0.2, lw=0.4)

    if n_c > 0:
        c_mean = np.nanmean(c_valid, axis=0)
        c_sem = np.sqrt(np.nanvar(c_valid, axis=0) / n_c)
        for t_idx in range(num_of_trials - 1):
            ax.plot(x[t_idx:t_idx + 2], c_mean[t_idx:t_idx + 2],
                    color=session_colors[t_idx + 1], lw=1.2, zorder=5)
        for t_idx in range(num_of_trials):
            ax.errorbar(x[t_idx], c_mean[t_idx], yerr=c_sem[t_idx],
                        color=session_colors[t_idx], capsize=1.5, capthick=0.7,
                        lw=0.7, zorder=6, fmt='o', markersize=2,
                        markerfacecolor=session_colors[t_idx],
                        markeredgecolor='white', markeredgewidth=0.3)

    ax.axhline(y=max_entropy, color='gray', ls=':', lw=0.6, alpha=0.4)
    ax.set_title(f'{cohort_labels[cohort]} (n={n_c})', fontsize=8)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(labelsize=7)

    if c_idx >= 3:
        ax.set_xlabel('Session', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(['S1', 'S2', 'S3', 'C1', 'C2', 'C3', 'C4', 'C5'], fontsize=7)
    if c_idx % 3 == 0:
        ax.set_ylabel('Entropy (bits)', fontsize=8)

fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.97], pad=0.3)
fig.suptitle('Behavioral entropy by cohort', fontsize=10, y=0.99)
plt.savefig(output_folder + 'entropy_by_cohort.png', dpi=300, bbox_inches='tight')
plt.savefig(output_folder + 'entropy_by_cohort.pdf', dpi=300, bbox_inches='tight')
plt.close()
print('Per-cohort figure saved')

#%% Within-session entropy time course
# Compute Shannon entropy in sliding 1-minute windows across session duration
window_frames = 1 * minute   # 900 frames = 1 min
step_frames = 30 * second    # 450 frames = 30 sec
max_duration = 27 * minute   # Truncate at 27 min (most sessions are ~30 min)

# Sessions to plot: saline3 as baseline reference + all cocaine
trace_trials = ['saline3', 'cocaine1', 'cocaine2', 'cocaine3', 'cocaine4', 'cocaine5']
trace_colors = ['#808080', '#FFA500', '#FF8C00', '#FF6347', '#E60000', '#990000']

# Window centers in minutes
n_windows = int((max_duration - window_frames) / step_frames) + 1
window_centers_min = np.array([(w * step_frames + window_frames / 2) / minute
                                for w in range(n_windows)])

# Compute entropy traces: per mouse per window per session
entropy_traces = {}
for trial in trace_trials:
    traces = np.full((len(valid_mice), n_windows), np.nan)
    for m_idx, mouse in enumerate(valid_mice):
        if trial in MT[mouse]:
            preds = MT[mouse][trial]['merged']
            for w in range(n_windows):
                start = w * step_frames
                end = start + window_frames
                if end <= len(preds):
                    traces[m_idx, w] = shannon_entropy(preds[start:end],
                                                       num_of_behaviors)
    entropy_traces[trial] = traces

# Figure: within-session entropy
plt.figure(figsize=(100 * mm, 70 * mm), frameon=False)

for t_idx, trial in enumerate(trace_trials):
    traces = entropy_traces[trial]
    valid_n = np.sum(~np.isnan(traces), axis=0)
    mean_trace = np.nanmean(traces, axis=0)
    sem_trace = np.sqrt(np.nanvar(traces, axis=0, ddof=0) / valid_n)

    plt.plot(window_centers_min, mean_trace,
             color=trace_colors[t_idx], lw=1.2, label=trial, zorder=5)
    plt.fill_between(window_centers_min,
                     mean_trace - sem_trace, mean_trace + sem_trace,
                     color=trace_colors[t_idx], alpha=0.2, zorder=2)

plt.xlabel('Time (min)', fontsize=10)
plt.ylabel('Behavioral entropy (bits)', fontsize=10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.legend(fontsize=8, frameon=False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.tight_layout(pad=0.3)
plt.savefig(output_folder + 'entropy_within_sessions.png', dpi=300, bbox_inches='tight')
plt.savefig(output_folder + 'entropy_within_sessions.pdf', dpi=300, bbox_inches='tight')
plt.close()
print('Within-session entropy figure saved')

# Print within-session summary stats
for trial in trace_trials:
    traces = entropy_traces[trial]
    valid_n = np.sum(~np.isnan(traces), axis=0)
    mean_trace = np.nanmean(traces, axis=0)
    # Early (0-5 min) vs late (20-27 min) within-session comparison
    early_windows = window_centers_min < 5
    late_windows = window_centers_min > 20
    early_mean = np.nanmean(mean_trace[early_windows])
    late_mean = np.nanmean(mean_trace[late_windows])
    print(f'{trial}: n={int(valid_n[0])}, early(0-5min)={early_mean:.3f}, '
          f'late(20-27min)={late_mean:.3f}, delta={late_mean - early_mean:+.3f}')

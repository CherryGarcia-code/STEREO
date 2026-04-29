#%% Imports and setup
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats
import bz2, pickle, copy
from matplotlib.lines import Line2D

behaviors = ['Jump', 'Undefined', 'Floor licking', 'Wall licking',
             'Grooming', 'Body licking', 'Rearing', 'Locomotion', 'Stationary']
n_behaviors = len(behaviors)
mm = 1/25.4
sec = 15
minute = 60 * sec

trials = ['saline1','saline2','saline3','cocaine1','cocaine2','cocaine3','cocaine4','cocaine5']
cohorts = ['drd1_hm4di', 'controls', 'a2a_hm4di', 'a2a_opto']

trial_colors = ['#808080','#808080','#808080',
                '#FFA500','#FF8C00','#FF6347','#E60000','#990000']
cohort_labels = {'drd1_hm4di': r'Drd1$_{hm4Di}$',
                 'controls': 'Controls',
                 'a2a_hm4di': r'A2a$_{hm4Di}$',
                 'a2a_opto': r'A2a$_{opto}$'}

output_folder = 'output/Entropy_analyses/'
os.makedirs(output_folder, exist_ok=True)

# ---------- Switch entropy (weighted transition entropy) ----------
def calc_switch_entropy(behavior, window_size, stride, n_behaviors, base=2):
    def calc_entropy_in_bin(b):
        b = np.asarray(b, dtype=int)
        counts = np.bincount(b, minlength=n_behaviors)
        p_state = counts / counts.sum()
        mask = np.concatenate(([True], b[1:] != b[:-1]))
        seq = b[mask]
        T = np.zeros((n_behaviors, n_behaviors), dtype=float)
        if seq.size >= 2:
            for t in range(seq.size - 1):
                T[seq[t], seq[t+1]] += 1
        row_sums = T.sum(axis=1, keepdims=True)
        P = np.divide(T, row_sums, out=np.zeros_like(T), where=row_sums != 0)
        H_switch = np.zeros(n_behaviors, dtype=float)
        for i in range(n_behaviors):
            p = P[i, :].copy()
            p[i] = 0.0
            s = p.sum()
            if s > 0:
                p /= s
                H_switch[i] = scipy.stats.entropy(p, base=base)
            else:
                H_switch[i] = np.nan
        valid = ~np.isnan(H_switch)
        if valid.any():
            w = p_state.copy()
            w[~valid] = 0
            if w.sum() > 0:
                w /= w.sum()
            return float(np.nansum(w * H_switch))
        return np.nan
    out = []
    for t0 in range(0, behavior.size, stride):
        t1 = t0 + window_size
        if t1 <= behavior.size:
            out.append(calc_entropy_in_bin(behavior[t0:t1]))
    return np.array(out, dtype=float)

# ---------- Behavioral entropy (normalized occupancy) ----------
def calc_behavioral_entropy(behavior, window_size, stride, n_behaviors):
    def behavioral_entropy_bin(b):
        b = np.asarray(b, dtype=int)
        counts = np.array([(b == i).sum() for i in range(n_behaviors)], dtype=float)
        p = counts / counts.sum()
        p = p[p > 0]
        H = -np.sum(p * np.log2(p))
        Hmax = np.log2(n_behaviors) if n_behaviors > 1 else 1.0
        return H / Hmax
    out = []
    for t0 in range(0, behavior.size, stride):
        t1 = t0 + window_size
        if t1 <= behavior.size:
            out.append(behavioral_entropy_bin(behavior[t0:t1]))
    return np.array(out, dtype=float)

#%% Load data
ifile = bz2.BZ2File('data/CMT_Aug24.pkl', 'rb')
CMT = pickle.load(ifile)
ifile.close()
print('Data loaded')

TM = {}
mouse_to_cohort = {}
for cohort in cohorts:
    if cohort not in CMT:
        continue
    for mouse in CMT[cohort]:
        mouse_to_cohort[mouse] = cohort
        for t in CMT[cohort][mouse]:
            if t not in TM:
                TM[t] = {}
            TM[t][mouse] = CMT[cohort][mouse][t]
del CMT

# Windowing: 3 min window, 30 sec stride
window_frames = 180 * sec
step_frames = 30 * sec
max_duration = 27 * minute
n_windows = int((max_duration - window_frames) / step_frames) + 1
window_centers_min = np.array([(w * step_frames + window_frames / 2) / minute
                                for w in range(n_windows)])

#%% Compute entropies
print('Computing entropies...')
switch_traces = {}
behav_traces = {}
mouse_lists = {}

for trial in trials:
    if trial not in TM:
        print(f'  {trial}: not in data')
        continue
    sw_traces, bh_traces, m_list = [], [], []
    for mouse in TM[trial]:
        preds = TM[trial][mouse]['merged']
        if preds.size // (60 * sec) < 29:
            continue
        sw = calc_switch_entropy(preds, window_frames, step_frames, n_behaviors)
        bh = calc_behavioral_entropy(preds, window_frames, step_frames, n_behaviors)
        sw_traces.append(sw[:n_windows])
        bh_traces.append(bh[:n_windows])
        m_list.append(mouse)
    switch_traces[trial] = np.array(sw_traces, dtype=float) if sw_traces else np.array([]).reshape(0, n_windows)
    behav_traces[trial] = np.array(bh_traces, dtype=float) if bh_traces else np.array([]).reshape(0, n_windows)
    mouse_lists[trial] = m_list
    print(f'  {trial}: {len(m_list)} mice')

#%% Plotting helpers
def plot_within_session(traces_dict, ylabel, save_name):
    fig = plt.figure(figsize=(100 * mm, 70 * mm))
    for t_idx, trial in enumerate(trials):
        if trial not in traces_dict or traces_dict[trial].shape[0] == 0:
            continue
        tr = traces_dict[trial]
        n_mice = np.maximum(np.sum(~np.isnan(tr), axis=0), 1)
        mean_tr = np.nanmean(tr, axis=0)
        sem_tr = np.sqrt(np.nanvar(tr, axis=0, ddof=0) / n_mice)
        plt.plot(window_centers_min, mean_tr, color=trial_colors[t_idx], lw=1.2, label=trial, zorder=5)
        plt.fill_between(window_centers_min, mean_tr - sem_tr, mean_tr + sem_tr,
                         color=trial_colors[t_idx], alpha=0.2, zorder=2)
    plt.xlabel('Time (min)', fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.xticks(fontsize=8); plt.yticks(fontsize=8)
    plt.legend(fontsize=7, frameon=False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.tight_layout(pad=0.3)
    plt.savefig(output_folder + save_name + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_folder + save_name + '.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved {save_name}')

def plot_by_cohort(traces_dict, mouse_lists_dict, ylabel, save_name):
    fig, axes = plt.subplots(1, 4, figsize=(180 * mm, 55 * mm), sharex=True, sharey=True)
    for c_idx, cohort in enumerate(cohorts):
        ax = axes[c_idx]
        for t_idx, trial in enumerate(trials):
            if trial not in traces_dict or traces_dict[trial].shape[0] == 0:
                continue
            m_indices = [i for i, m in enumerate(mouse_lists_dict[trial])
                         if mouse_to_cohort.get(m) == cohort]
            if len(m_indices) == 0:
                continue
            c_traces = traces_dict[trial][m_indices, :]
            n_mice = np.maximum(np.sum(~np.isnan(c_traces), axis=0), 1)
            mean_tr = np.nanmean(c_traces, axis=0)
            sem_tr = np.sqrt(np.nanvar(c_traces, axis=0, ddof=0) / n_mice)
            ax.plot(window_centers_min, mean_tr, color=trial_colors[t_idx], lw=0.9, zorder=5)
            ax.fill_between(window_centers_min, mean_tr - sem_tr, mean_tr + sem_tr,
                            color=trial_colors[t_idx], alpha=0.15, zorder=2)
        n_c = len([m for m in mouse_to_cohort if mouse_to_cohort[m] == cohort])
        ax.set_title(f'{cohort_labels[cohort]} (n={n_c})', fontsize=8)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(labelsize=7)
        ax.set_xlabel('Time (min)', fontsize=8)
        if c_idx == 0:
            ax.set_ylabel(ylabel, fontsize=8)
    legend_handles = [Line2D([0], [0], color=c, lw=1) for c in trial_colors]
    fig.legend(legend_handles, trials, loc='upper right', fontsize=6,
               frameon=False, bbox_to_anchor=(0.99, 0.99))
    fig.tight_layout(rect=[0.01, 0.01, 0.92, 0.97], pad=0.3)
    plt.savefig(output_folder + save_name + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_folder + save_name + '.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved {save_name}')

#%% Generate all 4 figures
print('\nGenerating figures...')
plot_within_session(switch_traces, 'Switch entropy (bits)', 'switch_entropy_within_sessions')
plot_by_cohort(switch_traces, mouse_lists, 'Switch entropy (bits)', 'switch_entropy_by_cohort')
plot_within_session(behav_traces, 'Behavioral entropy (norm.)', 'behavioral_entropy_within_sessions')
plot_by_cohort(behav_traces, mouse_lists, 'Behavioral entropy (norm.)', 'behavioral_entropy_by_cohort')
print(f'\nDone! All figures saved to {output_folder}')

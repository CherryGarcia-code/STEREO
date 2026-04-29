#%% Imports and constants
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
import helper_functions as hf
import bz2
import pickle
import copy
import sys
import os
import stats_helper_file as shf
import matplotlib as mpl
# ---- add this ANOVA effect-size section ----
import statsmodels.api as sm
from scipy import stats as spstats
import os
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
progressive_color_pallete = ['#808080','#FFA500','#FF8C00','#FF6347','#E60000','#990000']
behaviors = ['Jump', 'Undefined', 'Floor licking', 'Wall licking', 'Grooming', 'Body licking', 'Rearing', 'Locomotion', 'Stationary']
colors = ["#696969", "#d3d3d3", "#d73027", "#e57373", "#c4a7e7", "#8e63b8", "#b3e5fc", "#3399cc", "#1f4e79"]
lut = np.array([4,5,3,2,6,1,8,7,0])
FPS = 15
sec = 15
minute = 60*sec
LICKING_BEHAVIORS = [1,2,3,4]
def calc_entropy(behavior, window_size, stride, n_behaviors, base=2):
    def calc_entropy_in_bin(b):
        # b must be int labels 0..n_behaviors-1
        b = np.asarray(b, dtype=int)

        # --- occupancy (full-length, aligned) ---
        counts = np.bincount(b, minlength=n_behaviors)
        p_state = counts / counts.sum()

        # --- collapse consecutive identical labels into bouts ---
        mask = np.concatenate(([True], b[1:] != b[:-1]))
        seq = b[mask]  # bout sequence

        # --- transition counts on bout sequence ---
        T = np.zeros((n_behaviors, n_behaviors), dtype=float)
        if seq.size >= 2:
            for t in range(seq.size - 1):
                T[seq[t], seq[t+1]] += 1

        # --- row-normalize safely ---
        row_sums = T.sum(axis=1, keepdims=True)
        P = np.divide(T, row_sums, out=np.zeros_like(T), where=row_sums != 0)

        # --- switch entropy per state (exclude self, renormalize) ---
        H_switch = np.zeros(n_behaviors, dtype=float)
        for i in range(n_behaviors):
            p = P[i, :].copy()
            p[i] = 0.0
            s = p.sum()
            if s > 0:
                p /= s
                H_switch[i] = scipy.stats.entropy(p, base=base)
            else:
                H_switch[i] = np.nan  # no exits from i in this window

        # --- weighted average entropy across states ---
        valid = ~np.isnan(H_switch)
        if valid.any():
            # renormalize weights over valid states only
            w = p_state.copy()
            w[~valid] = 0
            if w.sum() > 0:
                w /= w.sum()
            return float(np.nansum(w * H_switch))
        return np.nan

    all_session_entropy = []
    for t0 in range(0, behavior.size, stride):
        t1 = t0 + window_size
        if t1 <= behavior.size:
            all_session_entropy.append(calc_entropy_in_bin(behavior[t0:t1]))
    return np.array(all_session_entropy, dtype=float)

#%% Load Saline->cocaine5 data
root_folder = 'May24'
folder = root_folder+'/Data/'
ifile = bz2.BZ2File(folder + 'CMT_Dec24.pkl', 'rb')
CMT = pickle.load(ifile)
ifile.close()
print('***All dictionaries were loaded***')
trials  = ['saline1','saline2','saline3','splashTest']
cohorts = ['drd1_hm4di','drd1_hm3dq','controls','a2a_hm4di','a2a_hm3dq','a2a_opto']
num_of_trials = len(trials)
n_behaviors = len(behaviors)
MT = {}
TM = {}
for c in cohorts:
    for m in CMT[c]:
        MT[m] = copy.deepcopy(CMT[c][m])
for cohort in cohorts:
    for mouse in CMT[cohort].keys():
        for t in CMT[cohort][mouse].keys():
            if t not in TM.keys():
                TM[t] = {}
            TM[t][mouse] = copy.deepcopy(CMT[cohort][mouse][t])
del CMT
#%% Within and between session entropy - saline3-> cocaine5
s = 15
output_folder = 'May24/Figures/Story_revision/'
os.makedirs(output_folder, exist_ok=True)
plt.figure()
for t_idx,t in enumerate(trials):
    all_mice_entropy = []
    for m in TM[t].keys():
        predictions = TM[t][m]['merged']['predictions']['smartMerge']
        if predictions.size//(60*s) < 29: continue
        # make sure you define n_behaviors correctly:
        # n_behaviors = int(np.max(predictions)) + 1  # or your fixed known number
        all_session_entropy = calc_entropy(
            predictions,
            window_size=180*s,
            stride=30*s,
            n_behaviors=n_behaviors,
            base=2
        )
        all_mice_entropy.append(all_session_entropy)

    min_len = np.min([len(m) for m in all_mice_entropy])
    trimmed_all_mice_entropy = [mouse[:min_len] for mouse in all_mice_entropy]
    trimmed_all_mice_entropy = np.array(trimmed_all_mice_entropy, dtype=float)
    all_mice_entropy_mean = np.nanmean(trimmed_all_mice_entropy,axis=0)
    all_mice_entropy_stderr = np.sqrt(np.nanvar(trimmed_all_mice_entropy,axis=0)/trimmed_all_mice_entropy.shape[0])
    plt.plot(all_mice_entropy_mean,label=t, color= colors[t_idx])
    plt.fill_between(np.arange(min_len),y1 = all_mice_entropy_mean-all_mice_entropy_stderr, y2 = all_mice_entropy_mean+all_mice_entropy_stderr,alpha=.2,color= colors[t_idx])

plt.title('Switch entropy within sessions')
plt.xlabel('Time (min)')
plt.xticks(np.arange(0, 60, 4),np.arange(0,30,2))
plt.ylabel('Switch entropy (bits)')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend(frameon=False)
plt.savefig(f'{output_folder}cocaineExposure_entropy_all_mice.png', dpi=200, bbox_inches='tight')
plt.close()

#%% Load D1 opto
root_folder = 'May24'
folder = root_folder+'/Data/'
ifile = bz2.BZ2File(folder + 'CMT_Dec24.pkl', 'rb')
CMT = pickle.load(ifile)
ifile.close()
print('***All dictionaries were loaded***')
trials  = ['BaselineDay1','BaselineDay2','BaselineDay3','AloneStim']
cohorts = ['drd1_opto']
num_of_trials = len(trials)
n_behaviors = len(behaviors)
temp_file = open(folder+'opto_alignment_dict.pkl', 'rb')
opto_dict = pickle.load(temp_file)
temp_file.close()
ISI = 80 * sec
stim_dur = 20*sec
#%% Within and between session entropy - D1 opto

from helper_functions import *
s = 15
output_folder = 'May24/Figures/Story_revision/'
os.makedirs(output_folder, exist_ok=True)
plt.figure()
for t_idx,t in enumerate(trials):
    all_mice_entropy = []
    for m in TM[t].keys():
        offset = opto_dict[m][1]
        predictions = CMT[c][m][t]['merged']['predictions']['smartMerge']
        # make sure you define n_behaviors correctly:
        # n_behaviors = int(np.max(predictions)) + 1  # or your fixed known number
        all_session_entropy = calc_entropy(
            predictions,
            window_size=20*s,
            stride=10*s,
            n_behaviors=n_behaviors,
            base=2
        )
        all_mice_entropy.append(all_session_entropy)

    min_len = np.min([len(m) for m in all_mice_entropy])
    trimmed_all_mice_entropy = [mouse[:min_len] for mouse in all_mice_entropy]

    trimmed_all_mice_entropy = np.array(trimmed_all_mice_entropy, dtype=float)
    all_mice_entropy_mean = np.nanmean(trimmed_all_mice_entropy,axis=0)
    all_mice_entropy_stderr = np.sqrt(np.nanvar(trimmed_all_mice_entropy,axis=0)/trimmed_all_mice_entropy.shape[0])
    plt.plot(all_mice_entropy_mean,label=t, color= colors[t_idx])
    plt.fill_between(np.arange(min_len),y1 = all_mice_entropy_mean-all_mice_entropy_stderr, y2 = all_mice_entropy_mean+all_mice_entropy_stderr,alpha=.2,color= colors[t_idx])

plt.title('Switch entropy within sessions')
plt.xlabel('Time (min)')
plt.xticks(np.arange(0, trimmed_all_mice_entropy.shape[1], 6),np.arange(16))
plt.vlines([18,26,34,42,50,58,66,74,82],ymin=.3,ymax=1,ls='--',lw=.7)
plt.ylabel('Switch entropy (bits)')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend(frameon=False)
plt.savefig(f'{output_folder}גd1Opto_entropy_all_mice.png', dpi=200, bbox_inches='tight')
plt.close()

plt.figure()
laser_on = [18,26,34,42,50,58,66,74,82]
sandwich_entropy = []
for mouse in np.arange(trimmed_all_mice_entropy.shape[0]):
    mouse_sandwich_entropy = []
    for t in laser_on:
        mouse_sandwich_entropy.append([trimmed_all_mice_entropy[mouse,t-2],trimmed_all_mice_entropy[mouse,t],trimmed_all_mice_entropy[mouse,t+2]])
    mouse_sandwich_entropy = np.array(mouse_sandwich_entropy, dtype=float)
    plt.plot(np.nanmean(mouse_sandwich_entropy,axis=0), color= 'gray',alpha=.3)
    sandwich_entropy.append(np.mean(mouse_sandwich_entropy,axis=0))

sandwich_entropy = np.array(sandwich_entropy, dtype=float)
print(sandwich_entropy)
plt.plot(np.nanmean(sandwich_entropy,axis=0), color= 'k')
plt.xticks([0,1,2],['Before','During','After'])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.ylabel('Switch entropy (bits)')
plt.savefig(f'{output_folder}גd1Opto_entropy_around_laser.png', dpi=200, bbox_inches='tight')
plt.close()

#%% Number of behavioral switches - baseline days VS stimulation day
c = 'drd1_opto'
n_mice = len(CMT[c])
plt.figure()

for t in trials[2:]:
    all_mice_prob_of_change = np.empty((n_mice, 1 * min - 1))
    all_mice_prob_of_change[:] = np.nan
    for m_idx,m in enumerate(CMT[c]):
        if m in ['c494m2']: continue
        per_mouse_n_switches = []
        baseline_n_switches = []
        predictions = CMT[c][m][t]['merged']['predictions']['smartMerge']
        offset = opto_dict[m][1]
        first_stim = 3 * min - int((offset / 1000.0) * 15)
        stim_times = np.arange(first_stim, 14 * min, ISI)
        n_stims = len(stim_times)
        mouse_prob_of_change = np.empty((n_stims, 1 * min-1))
        mouse_prob_of_change[:] = np.nan
        for time_idx, time in enumerate(stim_times[:5]):
            stim_predictions = np.copy(predictions[time-stim_dur:time+2*stim_dur])
            switch_mat = np.where(stim_predictions[1:]!=stim_predictions[:-1], 1, 0)
            mouse_prob_of_change[time_idx]= switch_mat

        mouse_prob_of_change = np.nanmean(mouse_prob_of_change, axis=0)
        mouse_prob_of_change = hf.smoothing(mouse_prob_of_change, 15)
        # plt.plot(mouse_prob_of_change, c='gray')
        all_mice_prob_of_change[m_idx] = mouse_prob_of_change

    plt.plot(np.nanmean(all_mice_prob_of_change,axis=0),label=t)
    plt.vlines([300,600],ymin=0.01,ymax = 0.08,ls='--')
# plt.xticks(np.arange(2),['Baseline','Stimulation Day'])
# plt.ylabel('# of behavioral switches per second')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend(frameon=False)
plt.show()


#%%
def behavioral_entropy(labels):
    labels = np.asarray(labels)
    counts = np.array([(labels == b).sum() for b in range(n_behaviors)], dtype=float)
    p = counts / counts.sum()

    p = p[p > 0]
    H = -np.sum(p * np.log2(p))

    K = n_behaviors
    Hmax = np.log2(K) if K > 1 else 1.0
    # return H
    return H / Hmax
c = 'drd1_opto'
if 'c494m2' in CMT[c]: CMT[c].pop('c494m2')
if 'c488Bm8' in CMT[c]: CMT[c].pop('c488Bm8')
if 'c488Bm10' in CMT[c]: CMT[c].pop('c488Bm10')
if 'c488Bm12' in CMT[c]: CMT[c].pop('c488Bm12')
if 'c480m2' in CMT[c]: CMT[c].pop('c480m2')
n_mice = len(CMT[c])
plt.figure()
n_trials = 2
n_stims = 9
between_session_entropy_comparison = np.zeros((3 , n_mice))

GRACE_PERIOD = 10
for m_idx,m in enumerate(CMT[c]):
    before = []
    during = []
    after = []
    predictions = CMT[c][m]['AloneStim']['merged']['predictions']['smartMerge']
    offset = opto_dict[m][1]
    first_stim = 3 * minute - int((offset / 1000.0) * 15)
    stim_times = np.arange(first_stim, 14 * minute, ISI)
    for time_idx, time in enumerate(stim_times):
        before.append(behavioral_entropy(predictions[time -stim_dur:time]))
        during.append(behavioral_entropy(predictions[time:time + stim_dur]))
        after.append(behavioral_entropy(predictions[time+stim_dur:time+2*stim_dur]))
    between_session_entropy_comparison[0,m_idx] = np.mean(before)
    between_session_entropy_comparison[1, m_idx] = np.mean(during)
    between_session_entropy_comparison[2, m_idx] = np.mean(after)

plt.figure()
plt.bar([0,1,2],np.mean(between_session_entropy_comparison,axis=1),color='white',edgecolor='k')
plt.plot(between_session_entropy_comparison,c='gray')
# plt.scatter(np.zeros(n_mice),between_session_entropy_comparison[0])
# plt.scatter(np.ones(n_mice),between_session_entropy_comparison[1])
# plt.scatter(np.ones(n_mice)+1,between_session_entropy_comparison[2])
# plt.bar([0,1,2],np.mean(between_session_entropy_comparison,axis=1),color='white',edgecolor='k')
plt.xticks([0,1,2],['Before','During','After'])
plt.show()


#%% Hazard analysis
import numpy as np

FPS = 15

def _is_stable_run(preds, start, min_run, end):
    """Return True if preds[start:start+min_run] are all equal (within bounds)."""
    stop = min(end, start + min_run)
    if stop - start < min_run:
        return False
    return np.all(preds[start:stop] == preds[start])

def first_stable_switch(preds, onset, end, min_run=3):
    """
    Find first stable switch away from behavior at onset.
    Returns (t1_frames, b1) where t1 is frames after onset (>=1), b1 is new behavior label.
    If none: (None, None)
    """
    b0 = preds[onset]
    for k in range(onset + 1, end):
        if preds[k] == b0:
            continue
        if min_run <= 1 or _is_stable_run(preds, k, min_run, end):
            return (k - onset), preds[k]
    return (None, None)

def first_escape_from_b1(preds, onset, t1, b1, end, min_run=3):
    """
    After switching into b1 at time t1, find first stable switch away from b1.
    Returns t2_frames (frames after onset) or None if no escape by end.
    """
    start = onset + t1 + 1
    for k in range(start, end):
        if preds[k] == b1:
            continue
        if min_run <= 1 or _is_stable_run(preds, k, min_run, end):
            return (k - onset)
    return None
def epoch_metrics(preds, onset, win_frames, early_frames=15, min_run=3):
    """
    Returns:
      early_switched (0/1): did a stable switch occur within first early_frames?
      escape_time (frames): t2 - t1 if early switch happened and escape observed; else None
      rigid (0/1): early switch happened but no escape within window
    """
    end = min(len(preds), onset + win_frames)
    if end - onset < win_frames:
        return None  # incomplete epoch

    # t1: first stable switch from onset behavior
    t1, b1 = first_stable_switch(preds, onset, end, min_run=min_run)
    if t1 is None or t1 > early_frames:
        return dict(early_switched=0, escape_time=None, rigid=0)

    # t2: first stable escape away from b1
    t2 = first_escape_from_b1(preds, onset, t1, b1, end, min_run=min_run)
    if t2 is None:
        return dict(early_switched=1, escape_time=None, rigid=1)

    return dict(early_switched=1, escape_time=(t2 - t1), rigid=0)
def mouse_summary(preds, stim_times, win_frames, early_frames=15, min_run=3):
    rows = []
    for onset in stim_times:
        out = epoch_metrics(preds, onset, win_frames, early_frames, min_run)
        if out is None:
            continue
        rows.append(out)

    if len(rows) == 0:
        return None

    early = np.array([r["early_switched"] for r in rows], dtype=float)
    rigid = np.array([r["rigid"] for r in rows], dtype=float)
    escape_times = [r["escape_time"] for r in rows if r["escape_time"] is not None]

    return dict(
        p_early=np.mean(early),          # Panel 1 per mouse
        p_rigid=np.mean(rigid),          # Panel 3 per mouse
        escape_times=np.array(escape_times, dtype=int)  # Panel 2 per mouse pooled times (frames)
    )
def cdf_from_escape_times(escape_times, win_frames):
    """
    escape_times: array of event times in frames (>=1), already t2-t1
    Censoring: epochs with no escape are not included here; we'll handle censoring separately below if needed.
    Returns CDF over 1..win_frames-1.
    """
    F = np.zeros(win_frames - 1, dtype=float)
    if len(escape_times) == 0:
        return np.full(win_frames - 1, np.nan)
    for t in escape_times:
        if 1 <= t < win_frames:
            F[t-1:] += 1
    F /= len(escape_times)
    return F
def km_survival(event_times, event_observed, win_frames):
    """
    event_times: list of times in frames (1..win-1) for events; censored can be win_frames (or None mapped to win_frames)
    event_observed: list of 0/1
    Returns survival S over frames 1..win-1.
    """
    d = np.zeros(win_frames, dtype=int)
    c = np.zeros(win_frames, dtype=int)
    n = len(event_times)

    for t, e in zip(event_times, event_observed):
        if t is None:
            t = win_frames
            e = 0
        t = int(t)
        if t < 1: t = 1
        if t > win_frames: t = win_frames
        if e == 1 and t < win_frames:
            d[t] += 1
        else:
            c[t] += 1  # censored at t (including t==win_frames)

    S = np.ones(win_frames - 1, dtype=float)
    at_risk = n
    surv = 1.0
    for k in range(1, win_frames):
        if at_risk > 0:
            hazard = d[k] / at_risk
            surv *= (1 - hazard)
        S[k-1] = surv
        at_risk -= (d[k] + c[k])
    return S  # survival; CDF = 1 - S
import matplotlib.pyplot as plt

def sem(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) <= 1:
        return np.nan
    return np.nanstd(x, ddof=1) / np.sqrt(len(x))

def paired_bar(ax, y_baseline, y_stim, ylabel, xticklabels=("Baseline", "Stim")):
    yb = np.asarray(y_baseline, float)
    ys = np.asarray(y_stim, float)
    # keep paired mice only
    mask = ~np.isnan(yb) & ~np.isnan(ys)
    yb, ys = yb[mask], ys[mask]

    means = [np.mean(yb), np.mean(ys)]
    errors = [sem(yb), sem(ys)]

    ax.bar([0], means[0], yerr=errors[0], capsize=3,color = 'white',edgecolor='gray')
    ax.bar([1], means[1], yerr=errors[1], capsize=3, color='white', edgecolor='blue')
    for i in range(len(yb)):
        ax.plot([0,1], [yb[i], ys[i]], linewidth=.7,c='gray')

    ax.set_xticks([0,1])
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel(ylabel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
def plot_cdf_with_sem(ax, cdfs, label,color):
    # cdfs: list of arrays (win-1,) per mouse
    M = np.vstack([c for c in cdfs if c is not None])
    mean = np.nanmean(M, axis=0)
    se = np.nanstd(M, axis=0, ddof=1) / np.sqrt(M.shape[0])
    x = np.arange(1, len(mean)+1) / FPS
    ax.plot(x, mean, c=color,label=label)
    ax.fill_between(x, mean - se, mean + se, alpha=0.2,color=color)

    ax.set_xlabel("Time since early switch (s)")
    ax.set_ylabel("P(escaped by time t)")
    ax.set_ylim(0, 1.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
EARLY_FRAMES = 15        # 1 s
WIN_FRAMES   = 300       # total window after onset (e.g. 20 s)
WIN_ESCAPE   = 285       # max allowed t2-t1 window (e.g. 19 s)
MIN_RUN      = 3

# store per mouse results
p_early_base, p_early_stim = [], []
p_rigid_base, p_rigid_stim = [], []
cdf_base_list, cdf_stim_list = [], []

for m in CMT[c]:
    if m=='c494m2':continue
    preds_base = CMT[c][m]['BaselineDay3']['merged']['predictions']['smartMerge']
    preds_stim = CMT[c][m]['AloneStim']['merged']['predictions']['smartMerge']
    offset = opto_dict[m][1]
    first_stim = 3 * minute - int((offset / 1000.0) * 15)
    stim_times = np.arange(first_stim, 14 * minute, ISI)

    # compute epoch-wise metrics, but keep only epochs with early switch for “escape”
    def collect_escape_events(preds):
        event_times = []
        event_obs = []
        for onset in stim_times:
            end = onset + WIN_FRAMES
            t1, b1 = first_stable_switch(preds, onset, end, min_run=MIN_RUN)
            if t1 is None or t1 > EARLY_FRAMES:
                continue  # not part of rigidity claim
            t2 = first_escape_from_b1(preds, onset, t1, b1, end, min_run=MIN_RUN)
            if t2 is None:
                event_times.append(WIN_ESCAPE)  # censored at max
                event_obs.append(0)
            else:
                dt = t2 - t1
                dt = min(dt, WIN_ESCAPE)
                event_times.append(dt)
                event_obs.append(1)
        return event_times, event_obs

    # Panel 1 + 3 per mouse:
    ms_base = mouse_summary(preds_base, stim_times, WIN_FRAMES, EARLY_FRAMES, MIN_RUN)
    ms_stim = mouse_summary(preds_stim, stim_times, WIN_FRAMES, EARLY_FRAMES, MIN_RUN)

    p_early_base.append(ms_base["p_early"] if ms_base else np.nan)
    p_early_stim.append(ms_stim["p_early"] if ms_stim else np.nan)
    p_rigid_base.append(ms_base["p_rigid"] if ms_base else np.nan)
    p_rigid_stim.append(ms_stim["p_rigid"] if ms_stim else np.nan)

    # Panel 2 per mouse KM CDF:
    et_b, eo_b = collect_escape_events(preds_base)
    et_s, eo_s = collect_escape_events(preds_stim)

    if len(et_b) > 0:
        S_b = km_survival(et_b, eo_b, win_frames=WIN_ESCAPE+1)
        cdf_base_list.append(1 - S_b)
    if len(et_s) > 0:
        S_s = km_survival(et_s, eo_s, win_frames=WIN_ESCAPE+1)
        cdf_stim_list.append(1 - S_s)

# ---- Plot panels ----
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

paired_bar(axes[0], p_early_base, p_early_stim,
           ylabel="P(early switch within 1 s)")

plot_cdf_with_sem(axes[1], cdf_base_list, label="Baseline",color='gray')
plot_cdf_with_sem(axes[1], cdf_stim_list, label="Stim",color='blue')
axes[1].legend(frameon=False)

paired_bar(axes[2], p_rigid_base, p_rigid_stim,
           ylabel="P(rigid | early switch)",
           xticklabels=("Baseline", "Stim"))

plt.tight_layout()
plt.show()

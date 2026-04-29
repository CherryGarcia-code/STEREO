#%% Imports and data loading
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import bz2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import scipy.stats as stats
import os
import helper_functions as hf

root_folder = '.'
folder = 'data/'
mm = 1 / 25.4

# --- A2a opto cocaine stim data ---
ifile = bz2.BZ2File(folder + 'CTM_Aug24.pkl', 'rb')
stim_days = pickle.load(ifile)['a2a_opto']
stim_days = {key: stim_days[key] for key in ['cocaine6laserStim', 'cocaine8laserStim']}
ifile.close()

# LUT remap (same as A2a_opto.py)
behaviors_a2a = ['Jump', 'Undefined', 'Floor licking', 'Wall licking',
                 'Grooming', 'Body licking', 'Rearing', 'Locomotion', 'Stationary']
colors_a2a = ["#696969", "#d3d3d3", "#d73027", "#e57373", "#c4a7e7",
              "#8e63b8", "#b3e5fc", "#3399cc", "#1f4e79"]
lut = np.array([4, 5, 3, 2, 6, 1, 8, 7, 0])
for stim_day in stim_days:
    for m in stim_days[stim_day]:
        stim_days[stim_day][m]['merged'] = lut[stim_days[stim_day][m]['merged']]

# --- A2a splash test opto data ---
ifile = bz2.BZ2File(folder + 'CTM_May24.pkl', 'rb')
CTM_May24 = pickle.load(ifile)
ifile.close()
splash_stim_days = {key: CTM_May24['a2a_opto'][key]
                    for key in ['splashTest1p2mW', 'splashTest2p2mW']
                    if key in CTM_May24['a2a_opto']}

# --- Splash test baseline data (all cohorts) ---
ifile = bz2.BZ2File(folder + 'CMT_May24.pkl', 'rb')
CMT_May24 = pickle.load(ifile)
ifile.close()

behaviors_splash = ['Grooming', 'Body licking', 'Wall licking', 'Floor licking',
                    'Rearing', 'Back to camera', 'Stationary', 'Locomotion', 'Jump']
colors_splash = ['#c21296', '#06d6a0', '#ee476f', '#1189b1', '#ffd169',
                 '#783f04', '#596163', '#969595', '#b4a7d6']
cohorts = ['drd1_hm4di', 'drd1_hm3dq', 'controls', 'a2a_hm4di', 'a2a_hm3dq', 'a2a_opto']

# Build splash test dict (same logic as splashTest_behaviorOnly.py)
splashTest_days = {}
for c in cohorts:
    for m in CMT_May24[c]:
        if 'splashTest' in CMT_May24[c][m]:
            if c == 'a2a_opto':
                try:
                    splashTest_days[m] = {}
                    splashTest_days[m]['pre1'] = CMT_May24[c][m]['BaselineDay1']
                    if m == 'cA180m6':
                        splashTest_days[m]['pre2'] = CMT_May24[c][m]['BaselineDay2A2ACreChR2']
                    else:
                        splashTest_days[m]['pre2'] = CMT_May24[c][m]['BaselineDay2']
                    splashTest_days[m]['pre3'] = CMT_May24[c][m]['BaselineDay3']
                    splashTest_days[m]['splashTest'] = CMT_May24[c][m]['splashTest']
                except KeyError:
                    del splashTest_days[m]
            else:
                try:
                    splashTest_days[m] = {}
                    splashTest_days[m]['pre1'] = CMT_May24[c][m]['saline1']
                    splashTest_days[m]['pre2'] = CMT_May24[c][m]['saline2']
                    splashTest_days[m]['pre3'] = CMT_May24[c][m]['saline3']
                    splashTest_days[m]['splashTest'] = CMT_May24[c][m]['splashTest']
                except KeyError:
                    if m in splashTest_days:
                        del splashTest_days[m]
del CMT_May24, CTM_May24

print(f'***Data loaded: {len(stim_days["cocaine6laserStim"])} A2a opto mice, '
      f'{len(splashTest_days)} splash test mice***')

# --- Constants ---
FPS = 15
second = 15
minute = 60 * second
ISI = 1 * minute + 20 * second
bin_duration = 20 * second
num_of_behaviors = len(behaviors_a2a)
num_of_stims_cocaine = 14
RNN_offset = 7

PATHO_LICKING = 0
NATURAL_LICKING = 1
NO_LICKING = 2
grouping_lut = np.array([NO_LICKING, NO_LICKING, PATHO_LICKING, PATHO_LICKING,
                         NATURAL_LICKING, NATURAL_LICKING, NO_LICKING, NO_LICKING, NO_LICKING])

#%% ========== ANALYSIS 1: iSPN Opto Transition Probability Matrices ==========
output_folder = root_folder + '/Figures/iSPN_opto_analysis/'
os.makedirs(output_folder, exist_ok=True)

first_stim = 10 * minute - RNN_offset
stim_times = np.arange(first_stim, 28 * minute, ISI)

# Collect transition counts across all mice, separating laser-on vs laser-off
trans_on_all = []   # per-mouse transition probability matrices (laser-on)
trans_off_all = []  # per-mouse transition probability matrices (laser-off)
mice_used = []
outliers = []

for mouse in stim_days['cocaine6laserStim']:
    # Select the stim day where mouse was stereotyping (>50% licking pre-stim)
    if np.count_nonzero(stim_days['cocaine6laserStim'][mouse]['merged'][:first_stim] <= 3) / first_stim >= 0.5:
        t = 'cocaine6laserStim'
    elif np.count_nonzero(stim_days['cocaine8laserStim'][mouse]['merged'][:first_stim] <= 3) / first_stim >= 0.5:
        t = 'cocaine8laserStim'
    else:
        outliers.append(mouse)
        continue

    predictions = stim_days[t][mouse]['merged']
    mice_used.append(mouse)

    # Build set of laser-on frame indices
    laser_on_set = set()
    for stim in stim_times:
        laser_on_set.update(range(int(stim), int(stim) + bin_duration))

    # Count transitions separately
    trans_on = np.zeros((num_of_behaviors, num_of_behaviors))
    trans_off = np.zeros((num_of_behaviors, num_of_behaviors))

    for i in range(len(predictions) - 1):
        if predictions[i] != predictions[i + 1]:  # only count actual transitions
            if i in laser_on_set:
                trans_on[predictions[i], predictions[i + 1]] += 1
            else:
                trans_off[predictions[i], predictions[i + 1]] += 1

    # Normalize rows to probabilities
    trans_on_prob = np.copy(trans_on)
    trans_off_prob = np.copy(trans_off)
    for b in range(num_of_behaviors):
        row_sum_on = trans_on_prob[b, :].sum()
        row_sum_off = trans_off_prob[b, :].sum()
        if row_sum_on > 0:
            trans_on_prob[b, :] /= row_sum_on
        if row_sum_off > 0:
            trans_off_prob[b, :] /= row_sum_off

    trans_on_all.append(trans_on_prob)
    trans_off_all.append(trans_off_prob)

trans_on_all = np.array(trans_on_all)
trans_off_all = np.array(trans_off_all)
n_mice = len(mice_used)
print(f'Analysis 1: N={n_mice} mice (outliers: {outliers})')

# Mean matrices
mean_on = np.nanmean(trans_on_all, axis=0)
mean_off = np.nanmean(trans_off_all, axis=0)
diff_matrix = mean_on - mean_off

# Stats: paired t-test per cell
p_matrix = np.ones((num_of_behaviors, num_of_behaviors))
for i in range(num_of_behaviors):
    for j in range(num_of_behaviors):
        vals_on = trans_on_all[:, i, j]
        vals_off = trans_off_all[:, i, j]
        valid = ~(np.isnan(vals_on) | np.isnan(vals_off))
        if valid.sum() >= 3:
            _, p_matrix[i, j] = stats.ttest_rel(vals_on[valid], vals_off[valid])

# --- Figure: 3-panel transition matrices ---
fig, axes = plt.subplots(1, 3, figsize=(180 * mm, 55 * mm))
short_labels = ['Jmp', 'Udf', 'FlL', 'WlL', 'Grm', 'BdL', 'Rer', 'Loc', 'Stn']

vmax_abs = max(np.nanmax(mean_off), np.nanmax(mean_on))
im0 = axes[0].imshow(mean_off, cmap='Blues', vmin=0, vmax=vmax_abs, aspect='equal')
axes[0].set_title('Laser OFF', fontsize=8, fontweight='bold')
axes[0].set_xticks(range(num_of_behaviors))
axes[0].set_xticklabels(short_labels, fontsize=5, rotation=45, ha='right')
axes[0].set_yticks(range(num_of_behaviors))
axes[0].set_yticklabels(short_labels, fontsize=5)
axes[0].set_ylabel('From', fontsize=7)
axes[0].set_xlabel('To', fontsize=7)

im1 = axes[1].imshow(mean_on, cmap='Blues', vmin=0, vmax=vmax_abs, aspect='equal')
axes[1].set_title('Laser ON (iSPN stim)', fontsize=8, fontweight='bold')
axes[1].set_xticks(range(num_of_behaviors))
axes[1].set_xticklabels(short_labels, fontsize=5, rotation=45, ha='right')
axes[1].set_yticks(range(num_of_behaviors))
axes[1].set_yticklabels(short_labels, fontsize=5)
axes[1].set_xlabel('To', fontsize=7)

vmax_diff = np.nanmax(np.abs(diff_matrix))
im2 = axes[2].imshow(diff_matrix, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff, aspect='equal')
axes[2].set_title('ON $-$ OFF', fontsize=8, fontweight='bold')
axes[2].set_xticks(range(num_of_behaviors))
axes[2].set_xticklabels(short_labels, fontsize=5, rotation=45, ha='right')
axes[2].set_yticks(range(num_of_behaviors))
axes[2].set_yticklabels(short_labels, fontsize=5)
axes[2].set_xlabel('To', fontsize=7)

# Mark significant cells
for i in range(num_of_behaviors):
    for j in range(num_of_behaviors):
        if p_matrix[i, j] < 0.05:
            axes[2].text(j, i, '*', ha='center', va='center', fontsize=7, fontweight='bold')

# Colorbars
for ax_i, im_i in [(axes[0], im0), (axes[1], im1)]:
    cb = fig.colorbar(im_i, ax=ax_i, shrink=0.7, aspect=15)
    cb.ax.tick_params(labelsize=5)
cb2 = fig.colorbar(im2, ax=axes[2], shrink=0.7, aspect=15)
cb2.ax.tick_params(labelsize=5)

fig.suptitle(f'iSPN Opto: Transition Probabilities (N={n_mice})', fontsize=9, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(output_folder + 'iSPN_transition_matrices.png', dpi=300)
fig.savefig(output_folder + 'iSPN_transition_matrices.pdf', dpi=300)
plt.close(fig)
print('Analysis 1 figure saved.')

#%% ========== ANALYSIS 2a: iSPN Effects on Grooming, Rearing, Locomotion ==========
# Using same data loaded above — compute pre/during/post fractions for key behaviors
# Grooming=4, Rearing=6, Locomotion=7 in the a2a (lut-remapped) encoding

target_behaviors = {'Grooming': 4, 'Rearing': 6, 'Locomotion': 7}
target_colors = {'Grooming': '#c4a7e7', 'Rearing': '#b3e5fc', 'Locomotion': '#3399cc'}

epoch_data = {}  # {behavior_name: (3, n_stims, n_mice)}
for bname, bidx in target_behaviors.items():
    epoch_data[bname] = np.zeros((3, num_of_stims_cocaine, n_mice))

m_idx = 0
for mouse in mice_used:
    if np.count_nonzero(stim_days['cocaine6laserStim'][mouse]['merged'][:first_stim] <= 3) / first_stim >= 0.5:
        t = 'cocaine6laserStim'
    else:
        t = 'cocaine8laserStim'
    predictions = stim_days[t][mouse]['merged']

    for stim_idx, laser_onset in enumerate(stim_times):
        for bname, bidx in target_behaviors.items():
            epoch_data[bname][0, stim_idx, m_idx] = np.count_nonzero(
                predictions[laser_onset - bin_duration:laser_onset] == bidx) / bin_duration
            epoch_data[bname][1, stim_idx, m_idx] = np.count_nonzero(
                predictions[laser_onset:laser_onset + bin_duration] == bidx) / bin_duration
            epoch_data[bname][2, stim_idx, m_idx] = np.count_nonzero(
                predictions[laser_onset + bin_duration:laser_onset + 2 * bin_duration] == bidx) / bin_duration
    m_idx += 1

# Stats
print('\nAnalysis 2a: iSPN effects on individual behaviors (Pre vs During, paired t-test):')
stats_2a = {}
for bname in target_behaviors:
    pre_means = np.mean(epoch_data[bname][0, :, :], axis=0)   # avg across stims per mouse
    during_means = np.mean(epoch_data[bname][1, :, :], axis=0)
    t_stat, p_val = stats.ttest_rel(pre_means, during_means)
    d = (during_means.mean() - pre_means.mean()) / np.std(during_means - pre_means)
    stats_2a[bname] = {'t': t_stat, 'p': p_val, 'd': d,
                        'pre_mean': pre_means.mean(), 'during_mean': during_means.mean()}
    print(f'  {bname}: pre={pre_means.mean():.3f}, during={during_means.mean():.3f}, '
          f't({n_mice-1})={t_stat:.2f}, p={p_val:.4f}, d={d:.2f}')

# --- Figure 2a: V-graphs for grooming, rearing, locomotion ---
fig, axes = plt.subplots(1, 3, figsize=(150 * mm, 55 * mm), sharey=True)
for ax_i, bname in zip(axes, target_behaviors):
    dat = epoch_data[bname]
    mouse_means = np.mean(dat, axis=1)  # (3, n_mice) — avg across stims
    overall_mean = np.mean(mouse_means, axis=1)  # (3,)
    overall_sem = np.std(mouse_means, axis=1) / np.sqrt(n_mice)

    # Individual mice (thin lines)
    ax_i.plot(mouse_means, alpha=0.2, color='gray', marker='o', markersize=2, lw=0.5)
    # Shaded bar for DURING
    ax_i.axvspan(0.8, 1.2, color='#009FE3', alpha=0.15)
    # Mean ± SEM
    ax_i.errorbar(range(3), overall_mean, yerr=overall_sem,
                  color=target_colors[bname], lw=2, capsize=3, capthick=1.5, marker='o', markersize=4)
    ax_i.set_xticks(range(3))
    ax_i.set_xticklabels(['Pre', 'During', 'Post'], fontsize=7)
    ax_i.set_title(bname, fontsize=8, fontweight='bold', color=target_colors[bname])
    ax_i.spines['top'].set_visible(False)
    ax_i.spines['right'].set_visible(False)
    ax_i.tick_params(labelsize=6)

    p = stats_2a[bname]['p']
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
    ax_i.text(1, ax_i.get_ylim()[1] * 0.9, sig, ha='center', fontsize=7, fontweight='bold')

axes[0].set_ylabel('Fraction of time', fontsize=7)
fig.suptitle(f'iSPN Stimulation: Effect on Behaviors (N={n_mice})', fontsize=9, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(output_folder + 'iSPN_behavior_V_graphs.png', dpi=300)
fig.savefig(output_folder + 'iSPN_behavior_V_graphs.pdf', dpi=300)
plt.close(fig)
print('Analysis 2a figure saved.')

#%% ========== ANALYSIS 2b: Splash Test Grooming Suppression ==========
# Splash test data uses a different behavior encoding:
# Grooming(0), Body licking(1), Wall licking(2), Floor licking(3),
# Rearing(4), Back to camera(5), Stationary(6), Locomotion(7), Jump(8)
# VDB (ventral-directed behaviors = indices 0-3) is the key metric

num_splash_mice = len(splashTest_days)
num_splash_behaviors = len(behaviors_splash)

# Compute fraction of time in each behavior for baseline avg vs splash test
PRE1, PRE2, PRE3, ST = 0, 1, 2, 3
fraction_of_frames = np.zeros((num_splash_behaviors, 4, num_splash_mice))
m_idx = 0
for m in splashTest_days:
    pre1 = splashTest_days[m]['pre1']['merged']
    pre2 = splashTest_days[m]['pre2']['merged']
    pre3 = splashTest_days[m]['pre3']['merged']
    st = splashTest_days[m]['splashTest']['merged']
    for b in range(num_splash_behaviors):
        fraction_of_frames[b, PRE1, m_idx] = np.count_nonzero(pre1 == b) / pre1.size
        fraction_of_frames[b, PRE2, m_idx] = np.count_nonzero(pre2 == b) / pre2.size
        fraction_of_frames[b, PRE3, m_idx] = np.count_nonzero(pre3 == b) / pre3.size
        fraction_of_frames[b, ST, m_idx] = np.count_nonzero(st == b) / st.size
    m_idx += 1

# Grooming = index 0 in splash encoding
baseline_grooming = np.mean(fraction_of_frames[0, :3, :], axis=0)  # avg pre1-3 per mouse
splash_grooming = fraction_of_frames[0, ST, :]

# VDB = indices 0-3
baseline_vdb = np.mean(np.sum(fraction_of_frames[:4, :3, :], axis=0), axis=0)
splash_vdb = np.sum(fraction_of_frames[:4, ST, :], axis=0)

t_groom, p_groom = stats.ttest_rel(baseline_grooming, splash_grooming)
d_groom = (splash_grooming.mean() - baseline_grooming.mean()) / np.std(splash_grooming - baseline_grooming)
t_vdb, p_vdb = stats.ttest_rel(baseline_vdb, splash_vdb)
d_vdb = (splash_vdb.mean() - baseline_vdb.mean()) / np.std(splash_vdb - baseline_vdb)

print(f'\nAnalysis 2b: Splash Test Grooming Suppression (N={num_splash_mice}):')
print(f'  Grooming: baseline={baseline_grooming.mean():.3f}, splash={splash_grooming.mean():.3f}, '
      f't({num_splash_mice-1})={t_groom:.2f}, p={p_groom:.4f}, d={d_groom:.2f}')
print(f'  VDB: baseline={baseline_vdb.mean():.3f}, splash={splash_vdb.mean():.3f}, '
      f't({num_splash_mice-1})={t_vdb:.2f}, p={p_vdb:.4f}, d={d_vdb:.2f}')

# --- Figure 2b: Splash test grooming + VDB suppression ---
fig, axes = plt.subplots(1, 2, figsize=(100 * mm, 60 * mm))

# Panel A: Grooming
ax = axes[0]
for i in range(num_splash_mice):
    ax.plot([0, 1], [baseline_grooming[i], splash_grooming[i]], color='gray', alpha=0.15, lw=0.5)
ax.errorbar([0, 1],
            [baseline_grooming.mean(), splash_grooming.mean()],
            yerr=[baseline_grooming.std() / np.sqrt(num_splash_mice),
                  splash_grooming.std() / np.sqrt(num_splash_mice)],
            color='#c21296', lw=2, capsize=4, capthick=1.5, marker='o', markersize=5)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Baseline', 'Splash Test'], fontsize=7)
ax.set_ylabel('Fraction of time', fontsize=7)
ax.set_title('Grooming', fontsize=8, fontweight='bold', color='#c21296')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=6)
sig_g = '***' if p_groom < 0.001 else '**' if p_groom < 0.01 else '*' if p_groom < 0.05 else 'n.s.'
ax.text(0.5, ax.get_ylim()[1] * 0.95, sig_g, ha='center', fontsize=8, fontweight='bold')

# Panel B: VDB
ax = axes[1]
for i in range(num_splash_mice):
    ax.plot([0, 1], [baseline_vdb[i], splash_vdb[i]], color='gray', alpha=0.15, lw=0.5)
ax.errorbar([0, 1],
            [baseline_vdb.mean(), splash_vdb.mean()],
            yerr=[baseline_vdb.std() / np.sqrt(num_splash_mice),
                  splash_vdb.std() / np.sqrt(num_splash_mice)],
            color='#730AFF', lw=2, capsize=4, capthick=1.5, marker='o', markersize=5)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Baseline', 'Splash Test'], fontsize=7)
ax.set_title('VDB (all licking)', fontsize=8, fontweight='bold', color='#730AFF')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=6)
sig_v = '***' if p_vdb < 0.001 else '**' if p_vdb < 0.01 else '*' if p_vdb < 0.05 else 'n.s.'
ax.text(0.5, ax.get_ylim()[1] * 0.95, sig_v, ha='center', fontsize=8, fontweight='bold')

fig.suptitle(f'Splash Test: Grooming Suppression (N={num_splash_mice})', fontsize=9, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(output_folder + 'splash_test_grooming_suppression.png', dpi=300)
fig.savefig(output_folder + 'splash_test_grooming_suppression.pdf', dpi=300)
plt.close(fig)
print('Analysis 2b figure saved.')

#%% ========== ANALYSIS 2c: iSPN Opto PSTH Dynamics for Key Behaviors ==========
# Peri-stimulus time histogram for grooming, rearing, locomotion around laser onset

dynamics = np.zeros((n_mice, num_of_stims_cocaine, num_of_behaviors, minute))
m_idx = 0
for mouse in mice_used:
    if np.count_nonzero(stim_days['cocaine6laserStim'][mouse]['merged'][:first_stim] <= 3) / first_stim >= 0.5:
        t = 'cocaine6laserStim'
    else:
        t = 'cocaine8laserStim'
    predictions = stim_days[t][mouse]['merged']
    for stim_idx, laser_onset in enumerate(stim_times):
        for behavior in range(num_of_behaviors):
            dynamics[m_idx, stim_idx, behavior, :] = (
                predictions[laser_onset - bin_duration:laser_onset + 2 * bin_duration] == behavior)
    m_idx += 1

fig, axes = plt.subplots(1, 3, figsize=(150 * mm, 50 * mm), sharey=True)
time_axis = np.arange(minute)
x_ticks = np.arange(0, minute, 5 * second)
x_labels = ((x_ticks - bin_duration) / second).astype(int)

for ax_i, bname in zip(axes, target_behaviors):
    bidx = target_behaviors[bname]
    psth = np.mean(dynamics[:, :, bidx, :], axis=1)  # (n_mice, minute)
    for m in range(n_mice):
        psth[m] = hf.smoothing(psth[m], 5)
    mean_psth = np.mean(psth, axis=0)
    sem_psth = np.std(psth, axis=0) / np.sqrt(n_mice)

    ax_i.fill_between(time_axis, mean_psth - sem_psth, mean_psth + sem_psth,
                       color=target_colors[bname], alpha=0.3)
    ax_i.plot(time_axis, mean_psth, color=target_colors[bname], lw=1.5)
    ax_i.axvspan(bin_duration, 2 * bin_duration, color='#009FE3', alpha=0.15)
    ax_i.set_xticks(x_ticks)
    ax_i.set_xticklabels(x_labels, fontsize=5)
    ax_i.set_xlabel('Time (s)', fontsize=7)
    ax_i.set_title(bname, fontsize=8, fontweight='bold', color=target_colors[bname])
    ax_i.spines['top'].set_visible(False)
    ax_i.spines['right'].set_visible(False)
    ax_i.tick_params(labelsize=6)

axes[0].set_ylabel('P(behavior)', fontsize=7)
fig.suptitle(f'iSPN Stimulation: Peri-stimulus Dynamics (N={n_mice})', fontsize=9, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(output_folder + 'iSPN_behavior_dynamics.png', dpi=300)
fig.savefig(output_folder + 'iSPN_behavior_dynamics.pdf', dpi=300)
plt.close(fig)
print('Analysis 2c figure saved.')

#%% ========== COMPOSITE FIGURE ==========
fig = plt.figure(figsize=(180 * mm, 160 * mm))
outer = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[1, 0.9, 0.9], hspace=0.45)

# --- Row 1: Transition matrices (A, B, C) ---
gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], wspace=0.4)
ax_A = fig.add_subplot(gs_top[0])
ax_B = fig.add_subplot(gs_top[1])
ax_C = fig.add_subplot(gs_top[2])

vmax_abs = max(np.nanmax(mean_off), np.nanmax(mean_on))
im_A = ax_A.imshow(mean_off, cmap='Blues', vmin=0, vmax=vmax_abs, aspect='equal')
ax_A.set_title('Laser OFF', fontsize=7, fontweight='bold')
ax_A.set_xticks(range(num_of_behaviors))
ax_A.set_xticklabels(short_labels, fontsize=4, rotation=45, ha='right')
ax_A.set_yticks(range(num_of_behaviors))
ax_A.set_yticklabels(short_labels, fontsize=4)
ax_A.set_ylabel('From', fontsize=6)
ax_A.set_xlabel('To', fontsize=6)
fig.colorbar(im_A, ax=ax_A, shrink=0.6, aspect=12).ax.tick_params(labelsize=4)

im_B = ax_B.imshow(mean_on, cmap='Blues', vmin=0, vmax=vmax_abs, aspect='equal')
ax_B.set_title('Laser ON', fontsize=7, fontweight='bold')
ax_B.set_xticks(range(num_of_behaviors))
ax_B.set_xticklabels(short_labels, fontsize=4, rotation=45, ha='right')
ax_B.set_yticks(range(num_of_behaviors))
ax_B.set_yticklabels(short_labels, fontsize=4)
ax_B.set_xlabel('To', fontsize=6)
fig.colorbar(im_B, ax=ax_B, shrink=0.6, aspect=12).ax.tick_params(labelsize=4)

vmax_diff = np.nanmax(np.abs(diff_matrix))
im_C = ax_C.imshow(diff_matrix, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff, aspect='equal')
ax_C.set_title('ON $-$ OFF', fontsize=7, fontweight='bold')
ax_C.set_xticks(range(num_of_behaviors))
ax_C.set_xticklabels(short_labels, fontsize=4, rotation=45, ha='right')
ax_C.set_yticks(range(num_of_behaviors))
ax_C.set_yticklabels(short_labels, fontsize=4)
ax_C.set_xlabel('To', fontsize=6)
fig.colorbar(im_C, ax=ax_C, shrink=0.6, aspect=12).ax.tick_params(labelsize=4)
for i in range(num_of_behaviors):
    for j in range(num_of_behaviors):
        if p_matrix[i, j] < 0.05:
            ax_C.text(j, i, '*', ha='center', va='center', fontsize=5, fontweight='bold')

# Panel labels
ax_A.text(-0.15, 1.1, 'A', transform=ax_A.transAxes, fontsize=10, fontweight='bold')
ax_B.text(-0.15, 1.1, 'B', transform=ax_B.transAxes, fontsize=10, fontweight='bold')
ax_C.text(-0.15, 1.1, 'C', transform=ax_C.transAxes, fontsize=10, fontweight='bold')

# --- Row 2: V-graphs (D, E, F) ---
gs_mid = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], wspace=0.35)
for panel_idx, bname in enumerate(target_behaviors):
    ax = fig.add_subplot(gs_mid[panel_idx])
    dat = epoch_data[bname]
    mouse_means = np.mean(dat, axis=1)
    overall_mean = np.mean(mouse_means, axis=1)
    overall_sem = np.std(mouse_means, axis=1) / np.sqrt(n_mice)

    ax.plot(mouse_means, alpha=0.15, color='gray', marker='o', markersize=1.5, lw=0.4)
    ax.axvspan(0.8, 1.2, color='#009FE3', alpha=0.15)
    ax.errorbar(range(3), overall_mean, yerr=overall_sem,
                color=target_colors[bname], lw=1.5, capsize=2, capthick=1, marker='o', markersize=3)
    ax.set_xticks(range(3))
    ax.set_xticklabels(['Pre', 'During', 'Post'], fontsize=6)
    ax.set_title(bname, fontsize=7, fontweight='bold', color=target_colors[bname])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=5)
    if panel_idx == 0:
        ax.set_ylabel('Fraction of time', fontsize=6)
    p = stats_2a[bname]['p']
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
    ax.text(1, ax.get_ylim()[1] * 0.92, sig, ha='center', fontsize=6, fontweight='bold')
    label = chr(ord('D') + panel_idx)
    ax.text(-0.15, 1.1, label, transform=ax.transAxes, fontsize=10, fontweight='bold')

# --- Row 3: PSTH dynamics (G, H, I) ---
gs_bot = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2], wspace=0.35)
for panel_idx, bname in enumerate(target_behaviors):
    ax = fig.add_subplot(gs_bot[panel_idx])
    bidx = target_behaviors[bname]
    psth = np.mean(dynamics[:, :, bidx, :], axis=1)
    for m_i in range(n_mice):
        psth[m_i] = hf.smoothing(psth[m_i], 5)
    mean_psth = np.mean(psth, axis=0)
    sem_psth = np.std(psth, axis=0) / np.sqrt(n_mice)

    ax.fill_between(time_axis, mean_psth - sem_psth, mean_psth + sem_psth,
                     color=target_colors[bname], alpha=0.3)
    ax.plot(time_axis, mean_psth, color=target_colors[bname], lw=1)
    ax.axvspan(bin_duration, 2 * bin_duration, color='#009FE3', alpha=0.15)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=4)
    ax.set_xlabel('Time (s)', fontsize=6)
    ax.set_title(bname, fontsize=7, fontweight='bold', color=target_colors[bname])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=5)
    if panel_idx == 0:
        ax.set_ylabel('P(behavior)', fontsize=6)
    label = chr(ord('G') + panel_idx)
    ax.text(-0.15, 1.1, label, transform=ax.transAxes, fontsize=10, fontweight='bold')

fig.suptitle(f'iSPN Optogenetic Stimulation During Cocaine Stereotypy (N={n_mice})',
             fontsize=10, fontweight='bold', y=0.98)
fig.savefig(output_folder + 'iSPN_opto_composite.png', dpi=300)
fig.savefig(output_folder + 'iSPN_opto_composite.pdf', dpi=300)
plt.close(fig)
print('Composite figure saved.')

#%% Print summary statistics
print('\n' + '='*60)
print('SUMMARY OF ALL STATISTICS')
print('='*60)
print(f'\nAnalysis 1: Transition Probability Matrices (N={n_mice})')
print(f'  Significant differences (p<0.05) in {np.sum(p_matrix < 0.05)} / {num_of_behaviors**2} cells')
sig_cells = np.argwhere(p_matrix < 0.05)
for cell in sig_cells:
    i, j = cell
    print(f'    {short_labels[i]} -> {short_labels[j]}: '
          f'OFF={mean_off[i,j]:.3f}, ON={mean_on[i,j]:.3f}, '
          f'diff={diff_matrix[i,j]:+.3f}, p={p_matrix[i,j]:.4f}')

print(f'\nAnalysis 2a: iSPN Effects on Key Behaviors (N={n_mice})')
for bname in target_behaviors:
    s = stats_2a[bname]
    print(f'  {bname}: pre={s["pre_mean"]:.3f}, during={s["during_mean"]:.3f}, '
          f't({n_mice-1})={s["t"]:.2f}, p={s["p"]:.4f}, d={s["d"]:.2f}')

print(f'\nAnalysis 2b: Splash Test Grooming (N={num_splash_mice})')
print(f'  Grooming: baseline={baseline_grooming.mean():.3f}, splash={splash_grooming.mean():.3f}, '
      f't({num_splash_mice-1})={t_groom:.2f}, p={p_groom:.4f}, d={d_groom:.2f}')
print(f'  VDB: baseline={baseline_vdb.mean():.3f}, splash={splash_vdb.mean():.3f}, '
      f't({num_splash_mice-1})={t_vdb:.2f}, p={p_vdb:.4f}, d={d_vdb:.2f}')
print('='*60)
print(f'\nAll figures saved to: {output_folder}')

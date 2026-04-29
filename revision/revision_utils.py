#%% Shared utilities for revision supplementary figures
"""
Shared constants, data loaders, entropy functions, and styling for all
revision supplementary figure scripts.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import bz2, pickle, copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FPS = 15
second = 15
minute = 60 * second
mm = 1 / 25.4
RNN_offset = 7
ISI = 1 * minute + 20 * second   # 80s inter-stim interval (opto)
bin_duration = 20 * second         # 20s stim epoch

behaviors = ['Jump', 'Undefined', 'Floor licking', 'Wall licking',
             'Grooming', 'Body licking', 'Rearing', 'Locomotion', 'Stationary']
n_behaviors = len(behaviors)
short_labels = ['Jmp', 'Udf', 'FL', 'WL', 'Grm', 'BL', 'Rer', 'Loc', 'Stn']

colors = ["#696969", "#d3d3d3", "#d73027", "#e57373", "#c4a7e7",
          "#8e63b8", "#b3e5fc", "#3399cc", "#1f4e79"]

# Behavior grouping
PATHO_LICKING = 0    # Floor licking (2) + Wall licking (3)
NATURAL_LICKING = 1  # Grooming (4) + Body licking (5)
NO_LICKING = 2
grouping_lut = np.array([NO_LICKING, NO_LICKING, PATHO_LICKING, PATHO_LICKING,
                         NATURAL_LICKING, NATURAL_LICKING, NO_LICKING, NO_LICKING, NO_LICKING])
category_names = ['Surface licking', 'Self-directed licking', 'No licking']
category_colors = ["#d73027", "#c4a7e7", "#1f4e79"]

# Session / trial info
trials_sal_coc = ['saline1', 'saline2', 'saline3',
                  'cocaine1', 'cocaine2', 'cocaine3', 'cocaine4', 'cocaine5']
trial_colors = ['#808080', '#808080', '#808080',
                '#FFA500', '#FF8C00', '#FF6347', '#E60000', '#990000']

# Cohorts (excluding hm3Dq)
cohorts_4 = ['drd1_hm4di', 'controls', 'a2a_hm4di', 'a2a_opto']
cohorts_6 = ['drd1_hm4di', 'drd1_hm3dq', 'controls', 'a2a_hm4di', 'a2a_hm3dq', 'a2a_opto']
cohort_labels = {
    'drd1_hm4di': r'Drd1$_{hm4Di}$', 'drd1_hm3dq': r'Drd1$_{hm3Dq}$',
    'controls': 'Controls',
    'a2a_hm4di': r'A2a$_{hm4Di}$', 'a2a_hm3dq': r'A2a$_{hm3Dq}$',
    'a2a_opto': r'A2a$_{opto}$', 'drd1_opto': r'Drd1$_{opto}$',
}

# LUT for remapping old encoding → canonical 9-class
lut = np.array([4, 5, 3, 2, 6, 1, 8, 7, 0])

# dSPN / iSPN colors
dSPN_color = '#d73027'
iSPN_color = '#721515'
laser_color = '#B2E2F6'

# Cocaine gradient (saline3 + cocaine1-5)
trace_trials = ['saline3', 'cocaine1', 'cocaine2', 'cocaine3', 'cocaine4', 'cocaine5']
trace_colors = ['#808080', '#FFA500', '#FF8C00', '#FF6347', '#E60000', '#990000']

# GCaMP / photometry mice to exclude from behavioral analyses
gcamp_mice = ['c528m5', 'c528m10', 'c548m1', 'c514Bm2', 'c514Bm8',
              'cA184m4', 'cA184m7', 'cA242m5', 'cA242m6', 'cA242m8']
subOptimal_infection = ['c512m4']

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
_data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

def _get_preds(entry, version):
    """Extract prediction array from a trial entry, handling format differences."""
    if version == 'Dec24':
        return entry['merged']['predictions']['smartMerge']
    else:
        return entry['merged']

def load_CMT(version='Dec24', remap=True):
    """Load CMT pickle (Cohort→Mouse→Trial). Returns dict with 'merged' as preds array.
    version: 'Dec24' (new, no remap needed), 'Aug24'/'May24' (old, remap via LUT)
    remap: apply LUT to predictions if True (only for Aug24/May24)
    """
    fname = f'CMT_{version}.pkl'
    path = os.path.join(_data_folder, fname)
    if not os.path.exists(path):
        path = os.path.join(_data_folder, 'archive', fname)
    with bz2.BZ2File(path, 'rb') as f:
        CMT = pickle.load(f)
    for cohort in CMT:
        for mouse in CMT[cohort]:
            for trial in CMT[cohort][mouse]:
                preds = _get_preds(CMT[cohort][mouse][trial], version)
                if remap and version != 'Dec24':
                    preds = lut[preds]
                CMT[cohort][mouse][trial]['merged'] = preds
                # Preserve velocity and SSD at top level for easy access
    return CMT

def load_CTM(version='Dec24', remap=True):
    """Load CTM pickle (Cohort→Trial→Mouse). Returns dict with 'merged' as preds array."""
    fname = f'CTM_{version}.pkl'
    path = os.path.join(_data_folder, fname)
    if not os.path.exists(path):
        path = os.path.join(_data_folder, 'archive', fname)
    with bz2.BZ2File(path, 'rb') as f:
        CTM = pickle.load(f)
    for cohort in CTM:
        for trial in CTM[cohort]:
            for mouse in CTM[cohort][trial]:
                preds = _get_preds(CTM[cohort][trial][mouse], version)
                if remap and version != 'Dec24':
                    preds = lut[preds]
                CTM[cohort][trial][mouse]['merged'] = preds
    return CTM

def flatten_CMT(CMT, cohorts=None):
    """Flatten CMT into (MT, mouse_to_cohort) dicts.
    MT[mouse][trial] = {...}, mouse_to_cohort[mouse] = cohort
    """
    MT = {}
    mouse_to_cohort = {}
    target_cohorts = cohorts if cohorts else list(CMT.keys())
    for c in target_cohorts:
        if c not in CMT:
            continue
        for m in CMT[c]:
            MT[m] = CMT[c][m]
            mouse_to_cohort[m] = c
    return MT, mouse_to_cohort

# ---------------------------------------------------------------------------
# Entropy functions (from colleague's definitions)
# ---------------------------------------------------------------------------
def calc_switch_entropy(behavior, window_size, stride, n_beh, base=2):
    """Weighted transition entropy: per-state conditional entropy of destinations,
    weighted by state occupancy. Self-transitions removed."""
    def calc_entropy_in_bin(b):
        b = np.asarray(b, dtype=int)
        counts = np.bincount(b, minlength=n_beh)
        p_state = counts / counts.sum()
        mask = np.concatenate(([True], b[1:] != b[:-1]))
        seq = b[mask]
        T = np.zeros((n_beh, n_beh), dtype=float)
        if seq.size >= 2:
            for t in range(seq.size - 1):
                T[seq[t], seq[t+1]] += 1
        row_sums = T.sum(axis=1, keepdims=True)
        P = np.divide(T, row_sums, out=np.zeros_like(T), where=row_sums != 0)
        H_switch = np.zeros(n_beh, dtype=float)
        for i in range(n_beh):
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

def calc_behavioral_entropy(behavior, window_size, stride, n_beh):
    """Normalized occupancy entropy: H / log2(n_behaviors)."""
    def behavioral_entropy_bin(b):
        b = np.asarray(b, dtype=int)
        counts = np.array([(b == i).sum() for i in range(n_beh)], dtype=float)
        p = counts / counts.sum()
        p = p[p > 0]
        H = -np.sum(p * np.log2(p))
        Hmax = np.log2(n_beh) if n_beh > 1 else 1.0
        return H / Hmax
    out = []
    for t0 in range(0, behavior.size, stride):
        t1 = t0 + window_size
        if t1 <= behavior.size:
            out.append(behavioral_entropy_bin(behavior[t0:t1]))
    return np.array(out, dtype=float)

def transition_rate(preds, n_beh):
    """Switches per minute."""
    n_switches = np.sum(np.diff(preds.astype(int)) != 0)
    duration_min = len(preds) / (FPS * 60)
    return n_switches / duration_min if duration_min > 0 else np.nan

def transition_matrix(preds, n_beh):
    """Row-normalised transition probability matrix."""
    preds = preds.astype(int)
    mat = np.zeros((n_beh, n_beh))
    for k in range(len(preds) - 1):
        if preds[k] != preds[k + 1]:
            mat[preds[k], preds[k + 1]] += 1
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return mat / row_sums

# ---------------------------------------------------------------------------
# Photometry helpers
# ---------------------------------------------------------------------------

# Corrupted signal exclusions (from Figure3/Figure7)
photom_exclusions = {
    'remove_mouse': [('a2a_hm3dq', 'cA242m8'), ('drd1_hm3dq', 'c514Bm2'), ('drd1_hm3dq', 'c514Bm8')],
    'remove_hemisphere': [
        ('controls', 'c548m8', 'left'), ('controls', 'c548m11', 'left'),
        ('a2a_hm3dq', 'cA242m5', 'left'),
    ],
    'remove_trial': [
        ('drd1_hm4di', 'c528m10', 'cocaine4'),
        ('controls', 'c548m11', 'cocaine5'),
        ('a2a_hm3dq', 'cA242m6', 'saline2'), ('a2a_hm3dq', 'cA242m6', 'cocaine5'),
        ('a2a_hm3dq', 'cA184m4', 'saline3'),
        ('controls', 'cA242m9', 'saline2'),
    ],
}

# Pathway assignment (matches Figure3/Figure7)
def get_pathway(cohort, mouse):
    """Return 'drd1' or 'a2a' pathway for a given cohort/mouse."""
    if cohort in ['drd1_hm4di', 'drd1_hm3dq']:
        return 'drd1'
    elif cohort in ['a2a_hm4di', 'a2a_hm3dq', 'a2a_opto']:
        return 'a2a'
    elif cohort == 'controls':
        return 'a2a' if mouse.startswith('cA') else 'drd1'
    return None

def get_photom_signal(entry, side='both', sig_type='z', smooth=False):
    """Extract photometry signal from a trial entry.
    Returns list of (signal, side_name) tuples.
    """
    if 'photom' not in entry:
        return []
    pk = entry['photom']
    results = []
    for s in (['left', 'right'] if side == 'both' else [side]):
        if s in pk and isinstance(pk[s], dict) and sig_type in pk[s]:
            sig = pk[s][sig_type].copy()
            if smooth:
                import helper_functions as hf
                sig = hf.smoothing(sig, 5)
            results.append((sig, s))
    return results

def segment_signal_window(signal, align_idx, pre_window, post_window):
    """Extract a fixed-length window from signal around align_idx."""
    start = align_idx - pre_window
    end = align_idx + post_window
    if start < 0 or end > len(signal):
        return np.full(pre_window + post_window, np.nan)
    seg = signal[start:end].copy()
    return seg

def segment_bouts_for_photom(preds, target_beh, from_beh=None, min_from_dur=5, min_bout_dur=1):
    """Segment bouts of target_beh, optionally requiring preceding from_beh.
    Returns list of (onset_frame, duration_frames) tuples.
    """
    bouts = []
    n = len(preds)
    i = 0
    while i < n:
        if preds[i] == target_beh:
            onset = i
            while i < n and preds[i] == target_beh:
                i += 1
            dur = i - onset
            if dur >= min_bout_dur:
                if from_beh is not None:
                    # Check preceding period
                    pre_start = max(0, onset - min_from_dur)
                    pre = preds[pre_start:onset]
                    if len(pre) >= min_from_dur and np.all(pre == from_beh):
                        bouts.append((onset, dur))
                else:
                    bouts.append((onset, dur))
        else:
            i += 1
    return bouts

def load_master_table():
    """Load the master_table.pkl from data/."""
    path = os.path.join(_data_folder, 'master_table.pkl')
    with bz2.BZ2File(path, 'rb') as f:
        return pickle.load(f)

# Photometry colors
dSPN_photom_color = '#8e63b8'   # self-licking context (from Fig7)
iSPN_photom_color = '#3d1f4d'
quartile_colors = ['#3B82F6', '#6479B3', '#6A5A99', '#8E63B8']


def setup_style():
    """Apply standard STEREO figure style."""
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'legend.frameon': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

def save_fig(fig, output_folder, name):
    """Save figure as both PNG and PDF."""
    os.makedirs(output_folder, exist_ok=True)
    fig.savefig(os.path.join(output_folder, name + '.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_folder, name + '.pdf'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {name}')

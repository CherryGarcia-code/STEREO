"""
Integrative Analysis 5: Transition Matrix Regime Clustering

K-means clustering (k=2) applied to per-session per-mouse behavioral
transition matrices reveals two distinct dynamical regimes: a "flexible"
regime (saline-like, high entropy) and a "stereotyped" regime (cocaine-like,
low entropy). Shows how sessions and cohorts shift between regimes across
cocaine sensitization.

Outputs:
  - Int5_RegimeClustering_4cohorts.png/pdf
  - Int5_RegimeClustering_6cohorts.png/pdf
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'revision'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats

from revision_utils import (
    mm, n_behaviors, behaviors, colors,
    cohorts_4, cohorts_6, cohort_labels,
    load_CMT, flatten_CMT, setup_style, save_fig
)

setup_style()
output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(output_folder, exist_ok=True)

K = 2   # number of regime clusters
CLUSTER_COLORS = ['#3399cc', '#d73027']   # flexible=blue, stereotyped=red
CLUSTER_NAMES  = ['Flexible', 'Stereotyped']

trials = ['saline1', 'saline2', 'saline3',
          'cocaine1', 'cocaine2', 'cocaine3', 'cocaine4', 'cocaine5']
session_labels = ['S1', 'S2', 'S3', 'C1', 'C2', 'C3', 'C4', 'C5']
n_sessions = len(trials)
is_saline = np.array([True, True, True, False, False, False, False, False])


def compute_transition_vector(preds):
    """Row-normalised off-diagonal transition matrix flattened to 1D vector.
    Returns (n_behaviors*(n_behaviors-1),) vector; off-diagonal only."""
    preds = preds.astype(int)
    mat = np.zeros((n_behaviors, n_behaviors), dtype=float)
    for k in range(len(preds) - 1):
        if preds[k] != preds[k + 1]:
            mat[preds[k], preds[k + 1]] += 1
    # Row-normalize
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    mat /= row_sums
    # Flatten off-diagonal elements only
    mask = ~np.eye(n_behaviors, dtype=bool)
    vec = mat[mask]
    return vec, mat


def run_analysis(cohorts, suffix):
    print(f'\n=== Regime Clustering ({suffix}) ===')
    CMT = load_CMT('Dec24')
    MT, mouse_to_cohort = flatten_CMT(CMT, cohorts)
    del CMT

    # Only mice with all 8 sessions
    valid_mice = [m for m in MT if all(t in MT[m] for t in trials)]
    n_mice = len(valid_mice)
    print(f'N = {n_mice} mice with complete data')

    # Build feature matrix: one row per (mouse, session) = n_mice*n_sessions rows
    n_feat = n_behaviors * (n_behaviors - 1)  # off-diagonal elements
    X = np.zeros((n_mice * n_sessions, n_feat))
    full_mats = np.zeros((n_mice * n_sessions, n_behaviors, n_behaviors))
    mouse_ids, session_ids, cohort_ids = [], [], []

    for mi, m in enumerate(valid_mice):
        for si, t in enumerate(trials):
            row_idx = mi * n_sessions + si
            vec, mat = compute_transition_vector(MT[m][t]['merged'])
            X[row_idx] = vec
            full_mats[row_idx] = mat
            mouse_ids.append(mi)
            session_ids.append(si)
            cohort_ids.append(mouse_to_cohort[m])

    mouse_ids   = np.array(mouse_ids)
    session_ids = np.array(session_ids)

    # K-means clustering
    km = KMeans(n_clusters=K, n_init=20, random_state=42)
    labels = km.fit_predict(X)

    # Identify which cluster is "stereotyped": the one with higher cocaine membership
    cocaine_frac = np.array([
        (labels[session_ids >= 3] == k).mean()
        for k in range(K)
    ])
    stereo_cluster = int(np.argmax(cocaine_frac))  # higher cocaine fraction
    flex_cluster   = 1 - stereo_cluster
    cluster_map    = {stereo_cluster: 1, flex_cluster: 0}  # 0=flex, 1=stereo
    labels_mapped  = np.array([cluster_map[l] for l in labels])
    print(f'Cocaine session fraction in "stereotyped" cluster: {cocaine_frac[stereo_cluster]:.3f}')

    # PCA for visualization
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    var_explained = pca.explained_variance_ratio_ * 100
    print(f'PCA variance explained: PC1={var_explained[0]:.1f}%, PC2={var_explained[1]:.1f}%')

    # Per-session cluster membership fraction
    stereo_frac_per_session = np.array([
        (labels_mapped[session_ids == si] == 1).mean()
        for si in range(n_sessions)
    ])

    # Centroid transition matrices
    centroid_mats = np.zeros((K, n_behaviors, n_behaviors))
    for k_mapped in range(K):
        mask = labels_mapped == k_mapped
        centroid_mats[k_mapped] = full_mats[mask].mean(axis=0)

    # ---------------------------------------------------------------------------
    # Figure
    # ---------------------------------------------------------------------------
    fig = plt.figure(figsize=(200 * mm, 150 * mm))
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           left=0.1, right=0.96, top=0.88, bottom=0.12,
                           wspace=0.52, hspace=0.55)

    # --- Panel A: PCA scatter colored by cluster ---
    ax_a = fig.add_subplot(gs[0, 0])
    for k_mapped, (col, name) in enumerate(zip(CLUSTER_COLORS, CLUSTER_NAMES)):
        mask = labels_mapped == k_mapped
        ax_a.scatter(coords[mask, 0], coords[mask, 1],
                     c=col, s=10, alpha=0.5, linewidths=0, zorder=2 + k_mapped,
                     label=name)
    ax_a.set_xlabel(f'PC1 ({var_explained[0]:.1f}%)', fontsize=8)
    ax_a.set_ylabel(f'PC2 ({var_explained[1]:.1f}%)', fontsize=8)
    ax_a.set_title('Regime clusters in\ntransition-matrix space', fontsize=8,
                   fontweight='bold', pad=14)
    ax_a.legend(fontsize=6, loc='upper right', frameon=False)
    ax_a.text(-0.22, 1.15, 'A', transform=ax_a.transAxes,
              fontsize=12, fontweight='bold', va='top')

    # --- Panel B: PCA scatter colored by session type ---
    session_colors_8 = ['#808080', '#808080', '#808080',
                        '#FFA500', '#FF8C00', '#FF6347', '#E60000', '#990000']
    ax_b = fig.add_subplot(gs[0, 1])
    for si in range(n_sessions):
        mask = session_ids == si
        ax_b.scatter(coords[mask, 0], coords[mask, 1],
                     c=session_colors_8[si], s=10, alpha=0.5, linewidths=0, zorder=2)
    ax_b.set_xlabel(f'PC1 ({var_explained[0]:.1f}%)', fontsize=8)
    ax_b.set_ylabel(f'PC2 ({var_explained[1]:.1f}%)', fontsize=8)
    ax_b.set_title('Regime clusters colored\nby session', fontsize=8,
                   fontweight='bold', pad=14)
    # Session legend
    from matplotlib.lines import Line2D
    leg_els = [Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=session_colors_8[i], markersize=6,
                      label=session_labels[i])
               for i in range(n_sessions)]
    ax_b.legend(handles=leg_els, ncol=4, fontsize=5.5, loc='upper right',
                frameon=False, handletextpad=0.2, columnspacing=0.5)
    ax_b.text(-0.22, 1.15, 'B', transform=ax_b.transAxes,
              fontsize=12, fontweight='bold', va='top')

    # --- Panel C: Stereotyped regime fraction per session ---
    ax_c = fig.add_subplot(gs[0, 2])
    bar_colors = [session_colors_8[si] for si in range(n_sessions)]
    x = np.arange(n_sessions)
    ax_c.bar(x, stereo_frac_per_session * 100, color=bar_colors, alpha=0.8, edgecolor='none')
    ax_c.axvline(2.5, color='k', ls='--', lw=0.7, alpha=0.5)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(session_labels, fontsize=7)
    ax_c.set_ylabel('% sessions in\n"Stereotyped" regime', fontsize=8)
    ax_c.set_ylim(0, 105)
    ax_c.set_title('"Stereotyped" regime\nmembership per session', fontsize=8,
                   fontweight='bold', pad=14)
    ax_c.text(-0.22, 1.15, 'C', transform=ax_c.transAxes,
              fontsize=12, fontweight='bold', va='top')

    # --- Panel D: Flexible regime centroid transition matrix ---
    beh_short = ['Jmp', 'Udf', 'FL', 'WL', 'Grm', 'BL', 'Rer', 'Loc', 'Stn']
    ax_d = fig.add_subplot(gs[1, 0])
    im_d = ax_d.imshow(centroid_mats[0], cmap='Blues', vmin=0, vmax=0.5, aspect='equal')
    ax_d.set_title(f'"{CLUSTER_NAMES[0]}" regime\n(mean transition matrix)',
                   fontsize=8, fontweight='bold', pad=14)
    ax_d.set_xticks(range(n_behaviors))
    ax_d.set_yticks(range(n_behaviors))
    ax_d.set_xticklabels(beh_short, fontsize=5.5, rotation=45, ha='right')
    ax_d.set_yticklabels(beh_short, fontsize=5.5)
    ax_d.set_xlabel('To behavior', fontsize=7)
    ax_d.set_ylabel('From behavior', fontsize=7)
    plt.colorbar(im_d, ax=ax_d, shrink=0.8, label='Transition prob.')
    ax_d.text(-0.25, 1.15, 'D', transform=ax_d.transAxes,
              fontsize=12, fontweight='bold', va='top')

    # --- Panel E: Stereotyped regime centroid ---
    ax_e = fig.add_subplot(gs[1, 1])
    im_e = ax_e.imshow(centroid_mats[1], cmap='Reds', vmin=0, vmax=0.5, aspect='equal')
    ax_e.set_title(f'"{CLUSTER_NAMES[1]}" regime\n(mean transition matrix)',
                   fontsize=8, fontweight='bold', pad=14)
    ax_e.set_xticks(range(n_behaviors))
    ax_e.set_yticks(range(n_behaviors))
    ax_e.set_xticklabels(beh_short, fontsize=5.5, rotation=45, ha='right')
    ax_e.set_yticklabels(beh_short, fontsize=5.5)
    ax_e.set_xlabel('To behavior', fontsize=7)
    ax_e.set_ylabel('From behavior', fontsize=7)
    plt.colorbar(im_e, ax=ax_e, shrink=0.8, label='Transition prob.')
    ax_e.text(-0.25, 1.15, 'E', transform=ax_e.transAxes,
              fontsize=12, fontweight='bold', va='top')

    # --- Panel F: Difference matrix (Stereotyped − Flexible) ---
    ax_f = fig.add_subplot(gs[1, 2])
    diff_mat = centroid_mats[1] - centroid_mats[0]
    vmax_diff = np.abs(diff_mat).max()
    im_f = ax_f.imshow(diff_mat, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff,
                       aspect='equal')
    ax_f.set_title('Difference matrix\n(Stereotyped − Flexible)', fontsize=8,
                   fontweight='bold', pad=14)
    ax_f.set_xticks(range(n_behaviors))
    ax_f.set_yticks(range(n_behaviors))
    ax_f.set_xticklabels(beh_short, fontsize=5.5, rotation=45, ha='right')
    ax_f.set_yticklabels(beh_short, fontsize=5.5)
    ax_f.set_xlabel('To behavior', fontsize=7)
    ax_f.set_ylabel('From behavior', fontsize=7)
    plt.colorbar(im_f, ax=ax_f, shrink=0.8, label='ΔTransition prob.')
    ax_f.text(-0.25, 1.15, 'F', transform=ax_f.transAxes,
              fontsize=12, fontweight='bold', va='top')

    # Top legend: sessions
    fig.legend(handles=leg_els, ncol=8, loc='upper center',
               bbox_to_anchor=(0.5, 0.97), fontsize=6.5, frameon=False,
               title='Session', title_fontsize=7)

    save_fig(fig, output_folder, f'Int5_RegimeClustering_{suffix}')


run_analysis(cohorts_4, '4cohorts')
run_analysis(cohorts_6, '6cohorts')
print('\nDone — regime clustering figures saved.')

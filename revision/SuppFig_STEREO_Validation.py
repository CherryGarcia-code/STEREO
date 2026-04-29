#%% SuppFig: Per-Behavior STEREO Validation
"""
Addresses R4-STEREO-2 (per-behavior F1) and R4-Fig1EF (add labels to confusion matrix).

Panels:
  A. Confusion matrix H1 x H2 with behavior labels
  B. Confusion matrix H1 x STEREO with behavior labels
  C. Per-behavior precision, recall, F1 — grouped bar chart (H1×H2 vs H1×STEREO)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from revision_utils import mm, setup_style, save_fig

setup_style()

output_folder = 'revision/output/'
os.makedirs(output_folder, exist_ok=True)

# ---------------------------------------------------------------------------
# Data: confusion matrices from Figure1_June2025.py (lines 116, 132)
# Row order as coded: Grooming, Body licking, Wall licking, Floor licking,
#                     Rearing, Back to camera, Other
# ---------------------------------------------------------------------------
behavior_labels = ['Grooming', 'Body licking', 'Wall licking',
                   'Floor licking', 'Rearing', 'Undefined', 'Loco/Stat']
n_classes = len(behavior_labels)

# H1 × H2 (original row order — NOT reversed)
mat_h1h2 = np.array([
    [80, 1, 0, 0, 0, 5, 14],
    [ 3,83, 0, 0, 0,12,  2],
    [ 0, 0,85, 0, 1,11,  3],
    [ 0, 0, 2,81, 0,10,  6],
    [ 0, 1, 0, 0,84,11,  4],
    [ 8, 1, 1, 0, 1,81,  7],
    [ 1, 0, 0, 0, 0, 3, 95]
], dtype=float)

# H1 × STEREO (original row order)
mat_h1st = np.array([
    [63, 8, 0, 0, 0,21,  9],
    [ 1,92, 0, 0, 1, 3,  3],
    [ 0, 1,88, 1, 0, 8,  2],
    [ 0, 1, 2,77, 0, 5, 15],
    [ 0,12, 0, 0,72, 9,  7],
    [ 4, 2, 2, 1, 2,83,  6],
    [ 3, 3, 0, 0, 0, 6, 87]
], dtype=float)

# ---------------------------------------------------------------------------
# Compute per-class precision, recall, F1 from percentage confusion matrices
# Each row sums to 100 (row = true class, col = predicted class)
# ---------------------------------------------------------------------------
def metrics_from_pct_matrix(mat):
    """Compute precision, recall, F1 from a percentage-based confusion matrix
    where each row sums to ~100."""
    n = mat.shape[0]
    precision = np.zeros(n)
    recall = np.zeros(n)
    f1 = np.zeros(n)
    for i in range(n):
        recall[i] = mat[i, i] / mat[i, :].sum()  # TP / (TP + FN)
        col_sum = mat[:, i].sum()
        precision[i] = mat[i, i] / col_sum if col_sum > 0 else 0  # TP / (TP + FP)
        if precision[i] + recall[i] > 0:
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
    return precision, recall, f1

prec_h1h2, rec_h1h2, f1_h1h2 = metrics_from_pct_matrix(mat_h1h2)
prec_h1st, rec_h1st, f1_h1st = metrics_from_pct_matrix(mat_h1st)

# Print summary
print('\n--- Per-behavior validation metrics ---')
print(f'{"Behavior":<18} {"H1×H2 F1":>10} {"H1×ST F1":>10} {"H1×H2 Prec":>12} {"H1×ST Prec":>12} {"H1×H2 Rec":>11} {"H1×ST Rec":>11}')
for i in range(n_classes):
    print(f'{behavior_labels[i]:<18} {f1_h1h2[i]:>10.3f} {f1_h1st[i]:>10.3f} '
          f'{prec_h1h2[i]:>12.3f} {prec_h1st[i]:>12.3f} {rec_h1h2[i]:>11.3f} {rec_h1st[i]:>11.3f}')
print(f'\nOverall accuracy — H1×H2: {np.diag(mat_h1h2).sum() / mat_h1h2.sum():.3f}'
      f'  H1×STEREO: {np.diag(mat_h1st).sum() / mat_h1st.sum():.3f}')
print(f'Macro-F1 — H1×H2: {np.mean(f1_h1h2):.3f}  H1×STEREO: {np.mean(f1_h1st):.3f}')

# ---------------------------------------------------------------------------
# Figure: 3-panel composite
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(220 * mm, 70 * mm))
gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1.3], wspace=0.7)

# --- Panel A: H1 × H2 confusion matrix with labels ---
ax_a = fig.add_subplot(gs[0])
# Reverse for display (as in original Figure1)
mat_disp = mat_h1h2[::-1, ::-1]
labels_disp = behavior_labels[::-1]

im_a = ax_a.imshow(mat_disp, cmap='Purples', vmin=0, vmax=100, aspect='equal')
for i in range(n_classes):
    for j in range(n_classes):
        val = mat_disp[i, j]
        txt_color = 'white' if val > 60 else 'black'
        ax_a.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=5.5, color=txt_color)
ax_a.set_xticks(range(n_classes))
ax_a.set_xticklabels(labels_disp, fontsize=5, rotation=45, ha='right')
ax_a.set_yticks(range(n_classes))
ax_a.set_yticklabels(labels_disp, fontsize=5)
ax_a.set_xlabel('Observer 2', fontsize=7)
ax_a.set_ylabel('Observer 1', fontsize=7)
ax_a.set_title('H1 × H2', fontsize=8, fontweight='bold', pad=12)
ax_a.spines[:].set_visible(True)
ax_a.text(-0.2, 1.15, 'A', transform=ax_a.transAxes, fontsize=12, fontweight='bold', va='top')

# --- Panel B: H1 × STEREO confusion matrix with labels ---
ax_b = fig.add_subplot(gs[1])
mat_disp_st = mat_h1st[::-1, ::-1]
im_b = ax_b.imshow(mat_disp_st, cmap='Purples', vmin=0, vmax=100, aspect='equal')
for i in range(n_classes):
    for j in range(n_classes):
        val = mat_disp_st[i, j]
        txt_color = 'white' if val > 60 else 'black'
        ax_b.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=5.5, color=txt_color)
ax_b.set_xticks(range(n_classes))
ax_b.set_xticklabels(labels_disp, fontsize=5, rotation=45, ha='right')
ax_b.set_yticks(range(n_classes))
ax_b.set_yticklabels(labels_disp, fontsize=5)
ax_b.set_xlabel('STEREO', fontsize=7)
ax_b.set_ylabel('Observer 1', fontsize=7)
ax_b.set_title('H1 × STEREO', fontsize=8, fontweight='bold', pad=12)
ax_b.spines[:].set_visible(True)
ax_b.text(-0.2, 1.15, 'B', transform=ax_b.transAxes, fontsize=12, fontweight='bold', va='top')

# --- Panel C: Grouped bar chart — F1, precision, recall ---
ax_c = fig.add_subplot(gs[2])

x = np.arange(n_classes)
bar_w = 0.35

# F1 bars
bars1 = ax_c.bar(x - bar_w/2, f1_h1h2, bar_w, label='H1 × H2', color='#6a5acd', alpha=0.7, edgecolor='none')
bars2 = ax_c.bar(x + bar_w/2, f1_h1st, bar_w, label='H1 × STEREO', color='#e57373', alpha=0.7, edgecolor='none')

ax_c.set_xticks(x)
ax_c.set_xticklabels(behavior_labels, fontsize=5.5, rotation=45, ha='right')
ax_c.set_ylabel('F1 score', fontsize=8)
ax_c.set_ylim(0, 1.05)
ax_c.set_title('Per-behavior F1', fontsize=8, fontweight='bold', pad=12)
ax_c.legend(fontsize=6, loc='upper left', bbox_to_anchor=(0.0, 0.98), frameon=False)
ax_c.axhline(y=np.mean(f1_h1h2), color='#6a5acd', ls='--', lw=0.8, alpha=0.5)
ax_c.axhline(y=np.mean(f1_h1st), color='#e57373', ls='--', lw=0.8, alpha=0.5)
ax_c.text(-0.15, 1.15, 'C', transform=ax_c.transAxes, fontsize=12, fontweight='bold', va='top')

fig.tight_layout(pad=0.5)
save_fig(fig, output_folder, 'SuppFig1_STEREO_Validation')

# ---------------------------------------------------------------------------
# Additional: Precision + Recall comparison figure
# ---------------------------------------------------------------------------
fig2, (ax_p, ax_r) = plt.subplots(1, 2, figsize=(130 * mm, 55 * mm))

for ax, metric_h1h2, metric_h1st, ylabel, title, panel in [
    (ax_p, prec_h1h2, prec_h1st, 'Precision', 'Per-behavior precision', 'D'),
    (ax_r, rec_h1h2, rec_h1st, 'Recall', 'Per-behavior recall', 'E')]:
    ax.bar(x - bar_w/2, metric_h1h2, bar_w, label='H1 × H2', color='#6a5acd', alpha=0.7, edgecolor='none')
    ax.bar(x + bar_w/2, metric_h1st, bar_w, label='H1 × STEREO', color='#e57373', alpha=0.7, edgecolor='none')
    ax.set_xticks(x)
    ax.set_xticklabels(behavior_labels, fontsize=5.5, rotation=45, ha='right')
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=8, fontweight='bold', pad=12)
    ax.legend(fontsize=6, loc='upper left', bbox_to_anchor=(0.0, 0.98), frameon=False)
    ax.text(-0.15, 1.15, panel, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')

fig2.tight_layout(pad=0.5)
save_fig(fig2, output_folder, 'SuppFig1_STEREO_Validation_PrecRecall')
print('\nDone — STEREO validation figures saved.')

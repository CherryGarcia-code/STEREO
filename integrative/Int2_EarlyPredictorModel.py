"""
Integrative Analysis 2: Early Predictor Model

Can saline-baseline behavioral features predict eventual cocaine-induced
surface licking severity? Uses linear ridge regression with leave-one-out
cross-validation (LOO-CV) to give unbiased performance estimates with N~24.

Features from saline3: surface licking fraction, switch entropy,
transition rate, grooming fraction, locomotion fraction.

Outputs:
  - Int2_EarlyPredictorModel_4cohorts.png/pdf
  - Int2_EarlyPredictorModel_6cohorts.png/pdf
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'revision'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

from revision_utils import (
    mm, n_behaviors, behaviors, colors, FPS, second, minute,
    cohorts_4, cohorts_6, cohort_labels,
    load_CMT, flatten_CMT, setup_style, save_fig,
    calc_switch_entropy, transition_rate
)

setup_style()
output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(output_folder, exist_ok=True)

FL, WL = 2, 3  # behavior indices
GROOMING, LOCOMOTION = 4, 7

window_frames = 3 * minute
step_frames = 30 * second


def extract_features(preds):
    """Compute behavioral features from a full session prediction array."""
    n = len(preds)
    occ = np.bincount(preds.astype(int), minlength=n_behaviors) / n

    surface_licking = occ[FL] + occ[WL]
    grooming = occ[GROOMING]
    locomotion = occ[LOCOMOTION]

    # Switch entropy (whole session, single window)
    sw = calc_switch_entropy(preds, n, n, n_behaviors)
    entropy = sw[0] if len(sw) > 0 and not np.isnan(sw[0]) else np.nan

    # Transition rate (switches/min)
    tr = transition_rate(preds, n_behaviors)

    return {
        'surface_licking': surface_licking,
        'grooming': grooming,
        'locomotion': locomotion,
        'entropy': entropy,
        'transition_rate': tr,
    }


def standardize(X_train, X_test):
    """Z-score X_test using statistics from X_train (prevent data leakage)."""
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd == 0] = 1.0
    return (X_train - mu) / sd, (X_test - mu) / sd


def ridge_predict(X_train, y_train, X_test, alpha=1.0):
    """Ridge regression prediction for X_test."""
    n, p = X_train.shape
    # Closed form: beta = (X'X + alpha*I)^-1 X'y
    A = X_train.T @ X_train + alpha * np.eye(p)
    b = X_train.T @ y_train
    beta = np.linalg.solve(A, b)
    return X_test @ beta


def run_analysis(cohorts, suffix):
    print(f'\n=== Early Predictor Model ({suffix}) ===')
    CMT = load_CMT('Dec24')
    MT, mouse_to_cohort = flatten_CMT(CMT, cohorts)
    del CMT

    # Only mice with saline3 AND cocaine5
    valid_mice = [m for m in MT if 'saline3' in MT[m] and 'cocaine5' in MT[m]]
    n_mice = len(valid_mice)
    print(f'N = {n_mice} mice')

    feature_names = ['Surface licking\n(saline)', 'Grooming\n(saline)',
                     'Locomotion\n(saline)', 'Switch entropy\n(saline)',
                     'Transition rate\n(saline)']
    n_features = len(feature_names)

    # Build feature matrix and target
    X_rows, y_rows, mouse_labels, cohort_labels_per = [], [], [], []
    for m in valid_mice:
        feats = extract_features(MT[m]['saline3']['merged'])
        target = extract_features(MT[m]['cocaine5']['merged'])['surface_licking']
        row = [feats['surface_licking'], feats['grooming'], feats['locomotion'],
               feats['entropy'], feats['transition_rate']]
        if not any(np.isnan(row)) and not np.isnan(target):
            X_rows.append(row)
            y_rows.append(target)
            mouse_labels.append(m)
            cohort_labels_per.append(mouse_to_cohort[m])

    X = np.array(X_rows)  # (n_mice, n_features)
    y = np.array(y_rows)  # (n_mice,)
    n_valid = len(y)
    print(f'Valid mice (no NaN): {n_valid}')

    # Leave-one-out cross-validation
    y_pred_loo = np.zeros(n_valid)
    for i in range(n_valid):
        mask_train = np.ones(n_valid, dtype=bool)
        mask_train[i] = False
        X_tr = X[mask_train]
        y_tr = y[mask_train]
        X_ts = X[i:i+1]
        X_tr_z, X_ts_z = standardize(X_tr, X_ts)
        y_pred_loo[i] = ridge_predict(X_tr_z, y_tr, X_ts_z, alpha=1.0)[0]

    # LOO performance
    r_loo, p_loo = stats.pearsonr(y, y_pred_loo)
    ss_res = np.sum((y - y_pred_loo) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2_loo = 1 - ss_res / ss_tot
    rmse_loo = np.sqrt(ss_res / n_valid)
    print(f'LOO-CV: r={r_loo:.3f}, R²={r2_loo:.3f}, RMSE={rmse_loo:.3f}, p={p_loo:.4f}')

    # Feature importance: Pearson r of each feature with target
    feat_r = []
    feat_p = []
    for f in range(n_features):
        r, p = stats.pearsonr(X[:, f], y)
        feat_r.append(r)
        feat_p.append(p)
        print(f'  {feature_names[f].replace(chr(10)," ")}: r={r:.3f}, p={p:.4f}')

    # ---------------------------------------------------------------------------
    # Figure: 3-panel
    # ---------------------------------------------------------------------------
    fig = plt.figure(figsize=(200 * mm, 75 * mm))
    gs = gridspec.GridSpec(1, 3, figure=fig,
                           left=0.1, right=0.96, top=0.84, bottom=0.22,
                           wspace=0.48)

    # --- Panel A: Predicted vs Actual (LOO-CV) ---
    ax_a = fig.add_subplot(gs[0])

    cohort_color_map = {
        'drd1_hm4di': '#d73027', 'drd1_hm3dq': '#f4a582',
        'controls': '#808080', 'a2a_hm4di': '#721515',
        'a2a_hm3dq': '#d9a0a0', 'a2a_opto': '#3399cc'
    }
    for i in range(n_valid):
        c = cohort_color_map.get(cohort_labels_per[i], '#808080')
        ax_a.scatter(y[i], y_pred_loo[i], color=c, s=30, alpha=0.8,
                     edgecolors='white', linewidths=0.5, zorder=3)

    # Unity line
    lim = [min(y.min(), y_pred_loo.min()) - 0.02,
           max(y.max(), y_pred_loo.max()) + 0.02]
    ax_a.plot(lim, lim, 'k--', lw=0.8, alpha=0.5)
    ax_a.set_xlim(lim)
    ax_a.set_ylim(lim)
    ax_a.set_xlabel('Actual (cocaine 5 surface licking)', fontsize=8)
    ax_a.set_ylabel('Predicted (LOO-CV)', fontsize=8)
    sig_str = '***' if p_loo < 0.001 else '**' if p_loo < 0.01 else '*' if p_loo < 0.05 else 'n.s.'
    ax_a.set_title(f'LOO cross-validation\nr={r_loo:.2f}, R²={r2_loo:.2f} {sig_str}',
                   fontsize=8, fontweight='bold', pad=14)
    ax_a.text(-0.22, 1.15, 'A', transform=ax_a.transAxes,
              fontsize=12, fontweight='bold', va='top')

    # --- Panel B: Feature importances (Pearson r) ---
    ax_b = fig.add_subplot(gs[1])
    feat_colors = ['#d73027' if r > 0 else '#3399cc' for r in feat_r]
    x_pos = np.arange(n_features)
    bars = ax_b.barh(x_pos, feat_r, color=feat_colors, alpha=0.75, edgecolor='none')
    # Significance stars
    for i, (r, p) in enumerate(zip(feat_r, feat_p)):
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        xpos_star = r + 0.02 * np.sign(r)
        ax_b.text(xpos_star, i, sig, va='center', ha='left' if r >= 0 else 'right',
                  fontsize=7, color='k')
    ax_b.axvline(0, color='k', lw=0.7)
    ax_b.set_yticks(x_pos)
    ax_b.set_yticklabels(feature_names, fontsize=7)
    ax_b.set_xlabel('Pearson r with cocaine 5 licking', fontsize=8)
    ax_b.set_title('Feature–target correlation\n(saline 3 features)', fontsize=8,
                   fontweight='bold', pad=14)
    ax_b.set_xlim(-1.0, 1.0)
    ax_b.text(-0.3, 1.15, 'B', transform=ax_b.transAxes,
              fontsize=12, fontweight='bold', va='top')

    # --- Panel C: Scatter — best predictor vs cocaine5 ---
    # Find feature with largest |r|
    best_feat = int(np.argmax(np.abs(feat_r)))
    ax_c = fig.add_subplot(gs[2])
    for i in range(n_valid):
        c = cohort_color_map.get(cohort_labels_per[i], '#808080')
        ax_c.scatter(X[i, best_feat], y[i], color=c, s=30, alpha=0.8,
                     edgecolors='white', linewidths=0.5, zorder=3)
    # Regression line
    slope, intercept, r_val, p_val, se = stats.linregress(X[:, best_feat], y)
    x_range = np.linspace(X[:, best_feat].min(), X[:, best_feat].max(), 100)
    ax_c.plot(x_range, slope * x_range + intercept,
              color='#d73027' if feat_r[best_feat] > 0 else '#3399cc',
              lw=1.5, alpha=0.8)
    ax_c.set_xlabel(feature_names[best_feat].replace('\n', ' '), fontsize=8)
    ax_c.set_ylabel('Cocaine 5 surface licking', fontsize=8)
    sig_str2 = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
    ax_c.set_title(f'Best single predictor\nr={r_val:.2f}, {sig_str2}',
                   fontsize=8, fontweight='bold', pad=14)
    ax_c.text(-0.22, 1.15, 'C', transform=ax_c.transAxes,
              fontsize=12, fontweight='bold', va='top')

    # Cohort legend
    from matplotlib.lines import Line2D
    legend_cohorts = [c for c in cohorts if c in cohort_color_map]
    from revision_utils import cohort_labels as cl
    legend_elems = [Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=cohort_color_map[c], markersize=7,
                           label=cl[c])
                    for c in legend_cohorts]
    fig.legend(handles=legend_elems, ncol=len(legend_cohorts),
               loc='lower center', bbox_to_anchor=(0.5, 0.01),
               fontsize=6.5, frameon=False)

    save_fig(fig, output_folder, f'Int2_EarlyPredictorModel_{suffix}')


run_analysis(cohorts_4, '4cohorts')
run_analysis(cohorts_6, '6cohorts')
print('\nDone — early predictor model saved.')

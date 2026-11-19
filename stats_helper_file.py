import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance
import itertools

# ---------- unified output helper ----------
def emit(stats_file, msg=""):
    """
    Write a message to a sink:
      - file-like object with .write()
      - 'print' (default) -> print to stdout
      - 'none' or None     -> do nothing
    """
    if stats_file is None or stats_file == 'none':
        return
    if stats_file == 'print':
        print(msg, end="" if msg.endswith("\n") else "\n")
    else:
        stats_file.write(msg if msg.endswith("\n") else msg + "\n")

# ---------- analyses ----------
def rmANOVA(data, conditions, stats_file='print'):
    """
    Repeated-measures ANOVA across 'condition' within subjects (mouse).
    data: np.ndarray (n_mice x n_conditions)
    conditions: list/array of condition labels
    """
    n_mice, n_conditions = data.shape
    df = pd.DataFrame({
        'mouse': np.repeat(np.arange(n_mice), n_conditions),
        'condition': np.tile([f'{conditions[i]}' for i in range(n_conditions)], n_mice),
        'value': data.flatten()
    })

    model = ols('value ~ C(condition) + C(mouse)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    emit(stats_file, str(anova_table))
    p_condition = anova_table.loc["C(condition)", "PR(>F)"]
    emit(stats_file, f'-> {"reject" if p_condition < 0.05 else "ns"}\n')
    return model


def premutation_rmANOVA(data, real_F, stats_file='print', n_permutations=10000):
    """
    Permutation test for rm-ANOVA F across conditions (labels shuffled within each mouse).
    """
    F_perm = []
    n_mice, n_conditions = data.shape
    for _ in range(n_permutations):
        shuff_data = np.array([np.random.permutation(row) for row in data])
        df = pd.DataFrame({
            'mouse': np.repeat(np.arange(n_mice), n_conditions),
            'condition': np.tile([f'{i+1}' for i in range(n_conditions)], n_mice),
            'value': shuff_data.flatten()
        })
        model = ols('value ~ C(condition) + C(mouse)', data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        F_perm.append(anova_table.loc["C(condition)", "F"])
    p_perm = (np.sum(np.array(F_perm) >= real_F) + 1) / (n_permutations + 1)
    emit(stats_file, f'Repeated measures ANOVA permutation test (statistic = F) ; '
                     f'Num permutations = {n_permutations}; p = {p_perm:.6g}')
    emit(stats_file, f'-> {"reject" if p_perm < 0.05 else "ns"}\n')
    return p_perm, np.array(F_perm)


def rmFriedman(data, stats_file='print'):
    """
    Friedman test across k conditions (columns). data shape: (n_mice, k).
    """
    stat, p = stats.friedmanchisquare(*[data[:, j] for j in range(data.shape[1])])
    emit(stats_file, f"Friedman χ² = {stat:.3f}, p = {p:.4g}")
    emit(stats_file, f' -> {"reject" if p < 0.05 else "ns"}\n')
    return stat, p


def conclude_LMM(results, stats_file='print', factor='day'):
    """
    Joint Wald test that all dummies of C(factor) equal 0 (omnibus).
    """
    names = results.params.index
    cols = [i for i, n in enumerate(names) if n.startswith(f"C({factor})[T.")]
    if not cols:
        emit(stats_file, f"[warn] No C({factor}) dummy coefficients found.")
        return np.nan, np.nan
    L = np.eye(len(names))[cols]
    wald = results.wald_test(L)
    chi2 = float(wald.statistic)
    pval = float(wald.pvalue)
    emit(stats_file, f"Omnibus {factor} effect (Wald): χ²({L.shape[0]}) = {chi2:.3f}, p = {pval:.4g}")
    emit(stats_file, f'-> {"Significant" if pval < 0.05 else "n.s."}\n')
    return chi2, pval


def paired_ttest(arr1, arr2, stats_file='print', twoSided=True, direction='less'):
    """
    Paired t-test; if twoSided=False, uses 'direction' in {'less','greater'}.
    """
    if twoSided:
        res = stats.ttest_rel(arr1, arr2)
    else:
        res = stats.ttest_rel(arr1, arr2, alternative=direction)
    emit(stats_file, f"Paired t-test: t = {res.statistic:.4g}, p = {res.pvalue:.4g}")
    emit(stats_file, f'-> {"reject" if res.pvalue < 0.05 else "n.s."}\n')
    return res


def permutation_paired_t_test(data, real_t, stats_file='print', n_permutations=10000):
    """
    Permutation test for paired t (sign-flip across subjects).
    data: shape (n, 2) with [condA, condB]
    """
    t_perm = []
    for _ in range(n_permutations):
        flips = np.random.rand(data.shape[0]) < 0.5
        perm = data.copy()
        perm[flips, :] = perm[flips, ::-1]
        t_res = stats.ttest_rel(perm[:, 0], perm[:, 1])
        t_perm.append(t_res.statistic)
    t_perm = np.array(t_perm)
    p_perm = (np.sum(np.abs(t_perm) >= np.abs(real_t)) + 1) / (n_permutations + 1)
    emit(stats_file, f'Paired t permutation (stat = t); N={n_permutations}; p = {p_perm:.6g}')
    emit(stats_file, f'-> {"reject" if p_perm < 0.05 else "ns"}\n')
    return p_perm, t_perm


def paired_wilcoxon(arr1, arr2, stats_file='print', twoSided=True, direction='less'):
    """
    Wilcoxon signed-rank; if twoSided=False, uses 'direction' in {'less','greater'}.
    """
    if twoSided:
        res = stats.wilcoxon(arr1, arr2)
    else:
        res = stats.wilcoxon(arr1, arr2, alternative=direction)
    emit(stats_file, f"Wilcoxon: W = {res.statistic:.4g}, p = {res.pvalue:.4g}")
    emit(stats_file, f'-> {"reject" if res.pvalue < 0.05 else "n.s."}\n')
    return res


def pairwise_wilcoxon(data, day_labels, stats_file='print', two_sided=True, direction=None):
    """
    All-pairs Wilcoxon across columns (within-subject). Holm correction reported.
    """
    raw_p = []
    pairs = list(itertools.combinations(range(data.shape[1]), r=2))
    if not two_sided and direction is None:
        raise ValueError("Provide 'direction' list when two_sided=False.")
    idx = 0
    for (d_m, d_n) in pairs:
        if two_sided:
            res = stats.wilcoxon(data[:, d_m], data[:, d_n])
        else:
            res = stats.wilcoxon(data[:, d_m], data[:, d_n], alternative=direction[idx])
        emit(stats_file, f'Wilcoxon: {day_labels[d_m]} vs {day_labels[d_n]} '
                         f'W = {res.statistic:.4g}; raw p = {res.pvalue:.4g}')
        raw_p.append(res.pvalue)
        idx += 1

    order = np.argsort(raw_p)
    p_sorted = np.array(raw_p)[order]
    pairs_sorted = np.array(pairs)[order]
    m = len(raw_p)
    p_holm = p_sorted * (m - np.arange(m))  # (basic Holm step-down; no monotonic fix)
    for (d_m, d_n), p_raw, p_adj in zip(pairs_sorted, p_sorted, p_holm):
        emit(stats_file, f'{day_labels[d_m]} vs {day_labels[d_n]}  : Holm p = {p_adj:.4g} '
                         f'-> {"reject" if p_adj < 0.05 else "ns"}')
    emit(stats_file, "")


def pairwise_ttest(data, day_labels, stats_file='print', two_sided=True, direction=None):
    """
    All-pairs paired t-tests across columns (within-subject). Holm correction reported.
    """
    if not two_sided and direction is None:
        raise ValueError("Provide 'direction' list when two_sided=False.")
    raw_p = []
    pairs = list(itertools.combinations(range(data.shape[1]), r=2))
    idx = 0
    for (d_m, d_n) in pairs:
        if two_sided:
            res = stats.ttest_rel(data[:, d_m], data[:, d_n])
        else:
            res = stats.ttest_rel(data[:, d_m], data[:, d_n], alternative=direction[idx])
        emit(stats_file, f'Paired t-test: {day_labels[d_m]} vs {day_labels[d_n]} '
                         f't = {res.statistic:.4g}; raw p = {res.pvalue:.4g}')
        raw_p.append(res.pvalue)
        idx += 1

    order = np.argsort(raw_p)
    p_sorted = np.array(raw_p)[order]
    pairs_sorted = np.array(pairs)[order]
    m = len(raw_p)
    p_holm = p_sorted * (m - np.arange(m))
    for (d_m, d_n), p_raw, p_adj in zip(pairs_sorted, p_sorted, p_holm):
        emit(stats_file, f'{day_labels[d_m]} vs {day_labels[d_n]}  : Holm p = {p_adj:.4g} '
                         f'-> {"reject" if p_adj < 0.05 else "ns"}')
    emit(stats_file, "")


def paired_cdf_permutation_test(cdfs_cond1, cdfs_cond2, stats_file='print',
                                n_permutations=10000, statistic='signed area'):
    """
    Permutation test on paired CDFs; statistic = signed area (default) or Wasserstein distance.
    """
    if statistic == 'Wasserstein':
        mouse_diffs = [wasserstein_distance(c1, c2) for c1, c2 in zip(cdfs_cond1, cdfs_cond2)]
    else:
        mouse_diffs = [np.sum((c1 - c2)) for c1, c2 in zip(cdfs_cond1, cdfs_cond2)]
    observed_stat = float(np.mean(mouse_diffs))

    null_stats = []
    for _ in range(n_permutations):
        permuted = []
        for c1, c2 in zip(cdfs_cond1, cdfs_cond2):
            if statistic == 'Wasserstein':
                d = wasserstein_distance(c1, c2)
            else:
                d = np.sum((c1 - c2))
            permuted.append(d if np.random.rand() < 0.5 else -d)
        null_stats.append(np.mean(permuted))

    null_stats = np.array(null_stats)
    p_val = float(np.mean(np.abs(null_stats) >= np.abs(observed_stat)))
    emit(stats_file, f'Number of permutations: {n_permutations}')
    emit(stats_file, f'statistic: {"Wasserstein" if statistic=="Wasserstein" else "signed area between CDFs"}')
    emit(stats_file, f'Two-sided test')
    emit(stats_file, f'observed stat: {observed_stat:.6g}')
    emit(stats_file, f'p-value: {p_val:.6g}')
    emit(stats_file, f' -> {"reject" if p_val < 0.05 else "n.s."}\n')
    return p_val, null_stats


def validate_lmm(result, pathway, parameter):
    residuals = result.resid
    fitted = result.fittedvalues

    plt.figure(figsize=(4, 3))
    sns.histplot(residuals, bins=40, kde=True, color='steelblue', alpha=.85)
    plt.axvline(0, color='k', ls='--')
    plt.title(f'{pathway} - LMM on {parameter} data; Residual Histogram')
    plt.xlabel('Residual'); plt.ylabel('Count'); plt.tight_layout(); plt.show()

    sm.qqplot(residuals, line='45', fit=True)
    plt.title(f'{pathway} - LMM on {parameter} data ; QQ-plot of Residuals')
    plt.tight_layout(); plt.show()

    plt.figure(figsize=(4, 3))
    sns.scatterplot(x=fitted, y=residuals, s=8, alpha=.25)
    plt.axhline(0, color='k', lw=.8)
    plt.xlabel('Fitted value'); plt.ylabel('Residual')
    plt.title(f'{pathway} - LMM on {parameter} data ; Residuals vs Fitted')
    plt.tight_layout(); plt.show()


def holm_correction(raw_p, ref_condition, conditions, stats_file='print'):
    """
    Step-down Holm over raw p-values against ref_condition.
    """
    order = np.argsort(raw_p)
    p_sorted = np.array(raw_p)[order]
    conditions_sorted = np.array(conditions)[order]
    m = len(raw_p)
    p_holm = p_sorted * (m - np.arange(m))
    for cond, p_raw, p_adj in zip(conditions_sorted, p_sorted, p_holm):
        emit(stats_file, f'{ref_condition} vs {cond} : Holm p = {p_adj:.4g} '
                         f'-> {"reject" if p_adj < 0.05 else "ns"}')

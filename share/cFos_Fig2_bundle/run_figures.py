"""
Reproduce the two cFos striatal figures for manuscript Fig 1 (cocaine dose series).

Self-contained: needs only this script + ./data/fos_cell_counts_flat.pkl.
No repo helpers, no notebook. Produces both figures into ./figures/ as PNG + PDF.

Figures
-------
1. fold_change_vs_control_by_region_with_SEM
      Section-level fold change in the cFos+ fraction vs the control mean, for
      VLS and DMS, at 20 and 30 mg/kg cocaine. Bars = group mean fold change,
      error bars = SEM, open circles = individual sections. Fraction uses the
      region-specific denominator (cFos+ cells / all cells in that region), and
      each region is normalised to its own control. Accompanying statistics
      (fold_change_statistics.txt) are computed from the same table.

2. cfos_density_map_pooled_mice_and_sections
      2D kernel-density map of cFos+ cell locations, pooled across BOTH mice and
      ALL their sections within each group (control / 20 / 30 mg/kg).

Run:  python run_figures.py
Tested with: Python 3.10, numpy 1.26.4, pandas 1.5.3, matplotlib 3.7.1,
             seaborn 0.13.2, scipy 1.10.1  (see requirements.txt).
"""
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")                     # no display needed; safe on any machine
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from scipy.stats import gaussian_kde as sp_kde
import seaborn as sns

# ── Style (matches the source notebook; matplotlibrc there is stock/no-op) ─────
mpl.rcParams["pdf.fonttype"] = 42          # editable text in Illustrator
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["pdf.use14corefonts"] = True
sns.set_context("paper")

mm = 1 / 25.4                              # mm → inches
np.random.seed(0)                          # reproducible scatter jitter

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "fos_cell_counts_flat.pkl"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD + PREPROCESS  (source notebook cells 4, 7, 9, 10)
# ══════════════════════════════════════════════════════════════════════════════
def load_and_preprocess():
    with open(DATA, "rb") as fh:
        all_data = pickle.load(fh)

    # -- EXCLUSION: drop neg1 section 003 (bad-quality section) ------------------
    all_data = all_data[
        ~((all_data["subject_id"] == "neg1") & (all_data["section_number"] == "003"))
    ].reset_index(drop=True)

    # -- Orient coordinates: invert y per section; mirror x for listed sections --
    all_data["y_inverted"] = all_data.groupby(
        ["subject_id", "section_number"]
    )["y_location"].transform(lambda y: y.max() - y)
    all_data["x_inverted"] = all_data["x_location"]

    flip_groups = {                         # sections imaged on the mirrored side
        "c577m11": ["002", "004"],
        "neg1": ["004"],
        "wt1": ["004"],
        "wt3": ["005"],
    }
    for subject, sections in flip_groups.items():
        for section in sections:
            sel = (all_data["subject_id"] == subject) & (all_data["section_number"] == section)
            max_x = all_data.loc[sel, "x_location"].max()
            all_data.loc[sel, "x_inverted"] = max_x - all_data["x_inverted"]

    # -- Assign one of 4 quadrant "regions" via per-section median crosshair -----
    medians = (
        all_data.groupby(["subject_id", "section_number"])
        .agg({"x_inverted": "median", "y_inverted": "median"})
        .rename(columns={"x_inverted": "mid_x", "y_inverted": "mid_y"})
    )
    all_data = all_data.merge(medians, on=["subject_id", "section_number"])
    conditions = [
        (all_data["x_inverted"] >= all_data["mid_x"]) & (all_data["y_inverted"] <= all_data["mid_y"]),
        (all_data["x_inverted"] >= all_data["mid_x"]) & (all_data["y_inverted"] > all_data["mid_y"]),
        (all_data["x_inverted"] < all_data["mid_x"]) & (all_data["y_inverted"] > all_data["mid_y"]),
        (all_data["x_inverted"] < all_data["mid_x"]) & (all_data["y_inverted"] <= all_data["mid_y"]),
    ]
    all_data["region"] = np.select(conditions, ["vls", "dls", "dms", "vms"])
    all_data.drop(columns=["mid_x", "mid_y"], inplace=True)

    # -- Clean fos_count: >1 are segmentation artefacts → treat as not-positive --
    fos_data = all_data.copy()
    fos_data.loc[fos_data["fos_count"] == 0, "fos_int_intensity"] = 0
    fos_data.loc[fos_data["fos_count"] > 1, "fos_count"] = 0
    max_int = fos_data.groupby("subject_id")["fos_int_intensity"].transform("max")
    fos_data["normalized_intensity"] = fos_data["fos_int_intensity"] / max_int
    return fos_data


def region_fraction_table(fos_data):
    """
    cFos+ fraction per subject / section / region, using the REGION-SPECIFIC
    denominator: cFos+ cells in the region / ALL cells in that region (i.e. the
    cFos+ rate within the region). Both the figure and the statistics are built
    from this one table so they match exactly.
    """
    tab = (fos_data.groupby(["subject_id", "exp_group", "section_number", "region"])
           .agg(positive_cell_count=("fos_count", lambda s: int((s == 1).sum())),
                region_cell_count=("fos_count", "size")).reset_index())
    tab["fraction"] = tab["positive_cell_count"] / tab["region_cell_count"]
    return tab


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — section-level fold change vs control, VLS & DMS  (notebook cell 27)
# ══════════════════════════════════════════════════════════════════════════════
def figure_fold_change(merged_df):
    region_order = ["dms", "vls"]
    dose_order = ["coc_20mgKg", "coc_30mgKg"]
    colors = ["#ff8c00", "#990000"]

    mean_control = {
        r: merged_df[(merged_df["region"] == r) & (merged_df["exp_group"] == "control")]["fraction"].mean()
        for r in region_order
    }
    fold_change_data = {r: {d: np.array([]) for d in dose_order} for r in region_order}
    for r in region_order:
        for d in dose_order:
            dd = merged_df[(merged_df["region"] == r) & (merged_df["exp_group"] == d)]
            fold_change_data[r][d] = (dd["fraction"] / mean_control[r]).values

    means = np.zeros((len(region_order), len(dose_order)))
    sems = np.zeros((len(region_order), len(dose_order)))
    for i, r in enumerate(region_order):
        for j, d in enumerate(dose_order):
            vals = fold_change_data[r][d]
            means[i, j] = np.nanmean(vals)
            sems[i, j] = np.nanstd(vals, ddof=1) / np.sqrt(np.sum(~np.isnan(vals)))

    def _draw(ax):
        bar_width = 0.35
        x = np.arange(len(region_order))
        for j, d in enumerate(dose_order):
            ax.bar(x + j * bar_width, means[:, j], bar_width, yerr=sems[:, j],
                   label=d, facecolor="none", edgecolor=colors[j], alpha=0.7, capsize=5)
            for i, r in enumerate(region_order):
                vals = fold_change_data[r][d]
                jitter = np.random.normal(0, 0.04, size=len(vals))
                ax.scatter(x[i] + j * bar_width + jitter, vals,
                           facecolors="none", edgecolors=colors[j], marker="o", s=18, alpha=0.6)
        ax.set_xticks(x + bar_width / 2)
        ax.set_xticklabels(region_order)
        ax.set_ylabel("Fold change\nvs control")
        ax.set_xlabel("Region")
        ax.legend(fontsize=8, frameon=False)
        sns.despine(ax=ax)

    # (a) native notebook size, 78 x 40 mm, cropped to content (bbox='tight')
    fig, ax = plt.subplots(figsize=(78 * mm, 40 * mm), constrained_layout=True)
    _draw(ax)
    stem = OUT / "fold_change_vs_control_by_region_with_SEM"
    fig.savefig(f"{stem}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # (b) exact manuscript page size, 95.955 x 89.777 mm (no tight bbox → exact)
    fig, ax = plt.subplots(figsize=(95.955 * mm, 89.777 * mm), constrained_layout=True)
    _draw(ax)
    fig.savefig(OUT / "fold_change_vs_control_by_region_with_SEM_95.955x89.777mm.pdf", dpi=300)
    plt.close(fig)

    print("\n[Figure 1] section-level fold change (mean, SEM):")
    for i, r in enumerate(region_order):
        for j, d in enumerate(dose_order):
            print(f"  {r} {d}: mean={means[i, j]:.2f}  SEM={sems[i, j]:.2f}  "
                  f"(n sections={np.sum(~np.isnan(fold_change_data[r][d]))})")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — density pooled across BOTH mice + all sections per group
# ══════════════════════════════════════════════════════════════════════════════
def figure_density(fos_data, normalize_globally=True):
    cmap = LinearSegmentedColormap.from_list(
        "YlOrBr_offset", mpl.colormaps["YlOrBr"](np.linspace(-0.15, 1.0, 256))
    )
    group_specs = [
        {"exp_group": "control", "label": "Control"},
        {"exp_group": "coc_20mgKg", "label": "Cocaine\n20 mg/kg"},
        {"exp_group": "coc_30mgKg", "label": "Cocaine\n30 mg/kg"},
    ]
    grouped = fos_data[fos_data["fos_count"] == 1].copy()

    peaks = []
    for spec in group_specs:
        sub = grouped[grouped["exp_group"] == spec["exp_group"]][["x_inverted", "y_inverted"]].dropna()
        if len(sub) > 1:
            k = sp_kde(sub.values.T)
            peaks.append(k(sub.values.T).max())
    gmax = max(peaks)
    norm = Normalize(vmin=0, vmax=gmax) if normalize_globally else None

    fig, axes = plt.subplots(1, len(group_specs),
                             figsize=(len(group_specs) * 80 * mm, 80 * mm),
                             constrained_layout=True)
    for ax, spec in zip(axes, group_specs):
        sub = grouped[grouped["exp_group"] == spec["exp_group"]]
        n_mice = sub["subject_id"].nunique()
        sns.kdeplot(x=sub["x_inverted"], y=sub["y_inverted"],
                    fill=True, cmap=cmap, levels=50, ax=ax, norm=norm)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        ax.set_title(f"{spec['label']}\n(n={n_mice} mice)", fontsize=10, pad=4)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm if normalize_globally else Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[-1], fraction=0.046, pad=0.04, shrink=0.8, orientation="vertical")
    cbar.set_label("Density (normalized)", fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    if normalize_globally:
        cbar.set_ticks([0, gmax * 0.25, gmax * 0.50, gmax * 0.75, gmax])
        cbar.set_ticklabels(["0", "0.25", "0.50", "0.75", "1.00"])

    stem = OUT / "cfos_density_map_pooled_mice_and_sections"
    fig.savefig(f"{stem}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\n[Figure 2] density pooled across mice + sections:")
    for spec in group_specs:
        sub = grouped[grouped["exp_group"] == spec["exp_group"]]
        print(f"  {spec['exp_group']}: mice={sorted(sub['subject_id'].unique())}  "
              f"n cFos+ cells={len(sub)}")


def _fdr_bh(pvals):
    """Benjamini-Hochberg FDR-adjusted p-values (no statsmodels dependency)."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(n)
    adj[order] = np.clip(ranked, 0, 1)
    return adj


def _cohens_d(x, y):
    """Cohen's d (pooled SD) for x vs y, with an approximate 95% CI."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    n1, n2 = len(x), len(y)
    s1, s2 = x.std(ddof=1), y.std(ddof=1)
    sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    d = (x.mean() - y.mean()) / sp if sp > 0 else np.nan
    se = np.sqrt((n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2)))
    return d, d - 1.96 * se, d + 1.96 * se


def figure_fold_change_stats(frac_df):
    """
    Statistics for the fold-change figure — reproduces the ORIGINAL section-level
    analysis. Built from the SAME region-specific fraction table as the figure
    (region_fraction_table), so the folds reported here match the figure exactly.
    Each region uses its own control baseline. Both the nonparametric (primary)
    and parametric (secondary) tests are reported.

        Fraction    = cFos+ cells in region / ALL cells in region  (region-specific)
        Fold change = section fraction / mean control fraction in that region
        Unit        = SECTION (as in the original analysis)

    INDUCTION (control vs dose, within region)
        PRIMARY   (nonparametric): Kruskal-Wallis + Mann-Whitney U, FDR-BH; rank-biserial r
        SECONDARY (parametric)   : one-way ANOVA + Welch t, Cohen's d [95% CI], FDR-BH
    REGIONAL (VLS vs DMS fold change, per dose)
        PRIMARY Mann-Whitney U;  SECONDARY Student t, Cohen's d [95% CI], Bonferroni x2

    Writes figures/fold_change_statistics.txt and returns the text.
    """
    from scipy import stats as spstats

    region_order = ["dms", "vls"]
    dose_order = ["coc_20mgKg", "coc_30mgKg"]
    groups = ["control", "coc_20mgKg", "coc_30mgKg"]

    tab = frac_df[frac_df["region"].isin(region_order)].copy()

    # Fold change per section, relative to the region's own control mean.
    ctrl_mean = {r: tab[(tab.region == r) & (tab.exp_group == "control")]["fraction"].mean()
                 for r in region_order}
    tab["fold"] = tab.apply(lambda r: r["fraction"] / ctrl_mean[r["region"]], axis=1)

    lines = []
    def emit(s=""):
        lines.append(s)

    def sem(a):
        a = np.asarray(a, float)
        return a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else np.nan

    emit("=" * 78)
    emit("STATISTICS - fold-change figure (cFos+ fraction by region and dose)")
    emit("=" * 78)
    emit("Fraction    = cFos+ cells in region / ALL cells in region (region-specific)")
    emit("Fold change = section fraction / mean control fraction in that region")
    emit("Unit        = SECTION (original analysis; n per group shown below)")

    # ── Descriptive fold change ───────────────────────────────────────────────
    emit("")
    emit("-" * 78)
    emit("FOLD CHANGE vs control (descriptive)")
    emit("-" * 78)
    for region in region_order:
        emit(f"\n{region.upper()}  (control fraction mean = {ctrl_mean[region]:.5f}):")
        for dose in dose_order:
            f = tab[(tab.region == region) & (tab.exp_group == dose)]["fold"].values
            emit(f"  {dose}: fold = {np.mean(f):.1f} +/- {sem(f):.1f} SEM  (n = {len(f)} sections)")

    # ── INDUCTION — PRIMARY (nonparametric) ────────────────────────────────────
    emit("")
    emit("-" * 78)
    emit("INDUCTION - control vs dose, within region")
    emit("PRIMARY (nonparametric): Kruskal-Wallis + Mann-Whitney U, FDR-BH")
    emit("-" * 78)
    for region in region_order:
        rt = tab[tab.region == region]
        gv = {g: rt[rt.exp_group == g]["fraction"].values for g in groups}
        H, pk = spstats.kruskal(*[gv[g] for g in groups])
        emit(f"\n{region.upper()}: Kruskal-Wallis H = {H:.2f}, p = {pk:.2e}  "
             f"(n = {', '.join(str(len(gv[g])) for g in groups)})")
        pairs = [("control", "coc_20mgKg"), ("control", "coc_30mgKg"), ("coc_20mgKg", "coc_30mgKg")]
        raw, rows = [], []
        for a, b in pairs:
            U, p = spstats.mannwhitneyu(gv[a], gv[b], alternative="two-sided")
            n1, n2 = len(gv[a]), len(gv[b])
            r_rb = abs(1 - 2 * U / (n1 * n2))
            direction = ">" if gv[b].mean() > gv[a].mean() else "<"
            raw.append(p); rows.append((a, b, U, p, r_rb, direction))
        for (a, b, U, p, r_rb, direction), pa in zip(rows, _fdr_bh(raw)):
            emit(f"    {a} vs {b}: U = {U:.0f}, p = {p:.2e}, p_FDR = {pa:.2e}, "
                 f"rank-biserial r = {r_rb:.2f}   ({b} {direction} {a})")

    # ── INDUCTION — SECONDARY (parametric) ─────────────────────────────────────
    emit("")
    emit("-" * 78)
    emit("SECONDARY (parametric): one-way ANOVA + Welch t, Cohen's d [95% CI], FDR-BH")
    emit("-" * 78)
    for region in region_order:
        rt = tab[tab.region == region]
        gv = {g: rt[rt.exp_group == g]["fraction"].values for g in groups}
        F, pf = spstats.f_oneway(*[gv[g] for g in groups])
        df_within = sum(len(gv[g]) for g in groups) - len(groups)
        emit(f"\n{region.upper()}: one-way ANOVA F({len(groups) - 1},{df_within}) = {F:.2f}, p = {pf:.2e}")
        pairs = [("control", "coc_20mgKg"), ("control", "coc_30mgKg")]
        raw, rows = [], []
        for a, b in pairs:
            t, p = spstats.ttest_ind(gv[a], gv[b], equal_var=False)
            d, lo, hi = _cohens_d(gv[b], gv[a])
            raw.append(p); rows.append((a, b, t, p, d, lo, hi))
        for (a, b, t, p, d, lo, hi), pa in zip(rows, _fdr_bh(raw)):
            emit(f"    {a} vs {b}: Welch t = {t:.2f}, p = {p:.2e}, p_FDR = {pa:.2e}, "
                 f"Cohen's d = {d:.2f} [{lo:.2f}, {hi:.2f}]")

    # ── REGIONAL — VLS vs DMS fold change, per dose ────────────────────────────
    emit("")
    emit("-" * 78)
    emit("REGIONAL - VLS vs DMS fold change, per dose")
    emit("PRIMARY: Mann-Whitney U   |   SECONDARY: Student t, Cohen's d [95% CI], Bonferroni x2")
    emit("-" * 78)
    for dose in dose_order:
        vls = tab[(tab.region == "vls") & (tab.exp_group == dose)]["fold"].values
        dms = tab[(tab.region == "dms") & (tab.exp_group == dose)]["fold"].values
        U, pmw = spstats.mannwhitneyu(vls, dms, alternative="two-sided")
        t, pt = spstats.ttest_ind(vls, dms, equal_var=True)      # Student's, matches manuscript table
        d, lo, hi = _cohens_d(vls, dms)
        direction = "VLS > DMS" if vls.mean() > dms.mean() else "DMS > VLS"
        emit(f"\n{dose}  ({direction}; VLS {vls.mean():.1f}x vs DMS {dms.mean():.1f}x; "
             f"n = {len(vls)},{len(dms)}):")
        emit(f"    PRIMARY   Mann-Whitney U = {U:.0f}, p = {pmw:.3f}  "
             f"(Bonferroni x2 = {min(pmw * 2, 1):.3f})")
        emit(f"    SECONDARY Student t = {t:.2f}, p = {pt:.3f}  "
             f"(Bonferroni x2 = {min(pt * 2, 1):.3f}), Cohen's d = {d:.2f} [{lo:.2f}, {hi:.2f}]")

    emit("")
    emit("=" * 78)
    text = "\n".join(lines)
    (OUT / "fold_change_statistics.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    return text


def main():
    print(f"Loading {DATA} ...")
    fos_data = load_and_preprocess()
    print(f"  {len(fos_data):,} cells after preprocessing; "
          f"groups: {sorted(fos_data['exp_group'].unique())}")
    frac_df = region_fraction_table(fos_data)     # shared by figure + stats
    figure_fold_change(frac_df)
    figure_density(fos_data)
    figure_fold_change_stats(frac_df)
    print(f"\nDone. Figures + statistics written to: {OUT}")


if __name__ == "__main__":
    main()

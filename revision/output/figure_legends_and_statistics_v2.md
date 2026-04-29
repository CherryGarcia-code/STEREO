# Figure Legends and Statistical Methods (v2)

All figures located in `revision/output/`. N = 37 mice from 6 cohorts (drd1_hm4di, drd1_hm3dq, controls, a2a_hm4di, a2a_hm3dq, a2a_opto) unless otherwise stated. Error bars and shading represent SEM throughout. All figures saved as PNG (300 dpi) and PDF. File names include figure number prefixes (e.g., `SuppFig2_AllBehaviors_Development`).

---

## Supplementary Figures

---

### SuppFig1_STEREO_Validation

**Supplementary Figure 1 | STEREO per-behavior classification performance.** (**A**) Confusion matrix between two independent human observers (H1 x H2), expressed as row-normalized percentages across seven behavior categories (Grooming, Body licking, Wall licking, Floor licking, Rearing, Undefined, Locomotion/Stationary). Cell values represent percentage agreement for each true-class row. (**B**) Confusion matrix between Observer 1 and the STEREO automated classifier (H1 x STEREO) using the same categories. (**C**) Per-behavior F1 scores for H1 x H2 (purple) and H1 x STEREO (red). Dashed horizontal lines indicate the macro-average F1 for each comparison.

#### Statistics

| Metric | Method |
|---|---|
| Precision, Recall, F1 | Computed per class from percentage-based confusion matrices. Recall = TP/(TP+FN); Precision = TP/(TP+FP); F1 = 2PR/(P+R). |
| Overall accuracy | Sum of diagonal / total matrix sum. |
| Macro-F1 | Unweighted mean of per-class F1. |

No inferential statistics. Descriptive validation from hardcoded human-annotated data.

---

### SuppFig1_STEREO_Validation_PrecRecall

**Supplementary Figure 1 (continued) | Per-behavior precision and recall.** (**D**) Per-behavior precision for H1 x H2 (purple) versus H1 x STEREO (red). (**E**) Per-behavior recall for the same comparisons. The STEREO classifier achieves comparable agreement to inter-rater reliability across most behaviors.

#### Statistics

Same as above. No inferential statistics.

---

### SuppFig2_AllBehaviors_Development

**Supplementary Figure 2 | Development of all nine behaviors across cocaine sensitization.** (**A**-**I**) Mean fraction of time spent in each of the 9 STEREO-classified behaviors across saline (S1-S3) and cocaine (C1-C5) sessions (N = 37 mice, 6 cohorts). Bar colors follow the cocaine escalation gradient (gray = saline, orange-to-dark-red = cocaine). Error bars denote SEM. Asterisks indicate significant differences between saline baseline (mean S1-S3) and late cocaine sessions (mean C3-C5). (**J**-**L**) Within-session timecourse metrics computed in a sliding 3-min window with 30-s stride: switch entropy (J), number of distinct behaviors (K), and transition rate (L) for saline 3 through cocaine 5. Shading represents SEM. (**M**-**Q**) Cumulative distribution functions (CDFs) of bout duration for floor licking, wall licking, grooming, locomotion, and rearing, comparing saline 3 (gray) and cocaine 5 (dark red). Bout counts shown in legends.

#### Statistics

| Analysis | Test | Details |
|---|---|---|
| Panels A-I: Saline vs late cocaine | Paired t-test | Mean of S1-S3 vs mean of C3-C5 per mouse per behavior. Cohen's d = mean(diff)/SD(diff). |
| Panels A-I: Monotonic trend | Spearman rho | Session-averaged means across 8 sessions per behavior. |
| Panels J-L | Descriptive | Mean +/- SEM timecourses. No inferential test. |
| Panels M-Q: Bout durations | Two-sample KS test | Saline 3 vs cocaine 5 bout duration distributions. Bout detection: `segment_bouts`, theta = 8 frames (~0.5 s). |

**Per-behavior results (N = 37 mice, paired t-test saline avg vs late cocaine avg):**

| Behavior | Saline mean | Late cocaine mean | t | p | Cohen's d | Spearman rho |
|---|---|---|---|---|---|---|
| Jump | 0.001 | 0.000 | 1.19 | 0.241 | 0.20 | -0.36 |
| Undefined | 0.051 | 0.039 | 2.21 | 0.034 | 0.36 | -0.76 |
| Floor licking | 0.037 | 0.323 | -8.20 | < 0.001 | -1.35 | 0.98 |
| Wall licking | 0.036 | 0.161 | -6.15 | < 0.001 | -1.01 | 0.95 |
| Grooming | 0.051 | 0.028 | 1.99 | 0.054 | 0.33 | -0.45 |
| Body licking | 0.039 | 0.001 | 8.13 | < 0.001 | 1.34 | -0.71 |
| Rearing | 0.066 | 0.020 | 7.63 | < 0.001 | 1.25 | -0.93 |
| Locomotion | 0.628 | 0.426 | 5.66 | < 0.001 | 0.93 | -0.88 |
| Stationary | 0.091 | 0.002 | 9.57 | < 0.001 | 1.57 | -0.93 |

**Bout duration KS tests (saline 3 vs cocaine 5, N = 37 mice):**

| Behavior | Saline 3 median | n (S3) | Cocaine 5 median | n (C5) | KS | p |
|---|---|---|---|---|---|---|
| Floor licking | 1.0 s | 1003 | 1.5 s | 7397 | 0.215 | < 0.001 |
| Wall licking | 1.0 s | 1192 | 1.1 s | 2890 | 0.131 | < 0.001 |
| Grooming | 1.3 s | 1340 | 1.6 s | 800 | 0.099 | < 0.001 |
| Locomotion | 1.7 s | 15087 | 1.2 s | 9470 | 0.179 | < 0.001 |
| Rearing | 1.4 s | 2515 | 0.9 s | 468 | 0.258 | < 0.001 |

---

### SuppFig2v2_AllBehaviors_Development

**Supplementary Figure 2 (v2) | Development of all nine behaviors across cocaine sensitization — individual trajectories.** Identical to SuppFig2 except panels (**A**-**I**) display individual mouse trajectories (thin gray lines, α = 0.15) overlaid with group mean ± SEM (colored line with shaded band). Saline sessions (S1-S3) are indicated by a light gray background. All statistics, panels J-L, and panels M-Q are identical to SuppFig2.

#### Statistics

Same as SuppFig2_AllBehaviors_Development above.

---

### SuppFig3_iSPN_Opto_Transitions

**Supplementary Figure 3 | iSPN optogenetic stimulation alters behavioral transition structure.** (**A**-**B**) Mean row-normalized 9x9 behavioral transition probability matrices during laser-OFF (A) and laser-ON (B) epochs in iSPN (A2a-ChR2) optogenetic sessions. Rows = source behavior, columns = destination. (**C**) Difference matrix (ON - OFF). Asterisks mark significantly altered transitions (paired t-test, P < 0.05). (**D**-**F**) V-graphs showing fraction of frames in surface licking (D), locomotion (E), and grooming (F) across Pre, During, and Post laser epochs. Gray lines = individual mice; colored points = mean +/- SEM. Only mice with >= 50% pre-stim surface licking included (N = 7 mice).

#### Statistics

| Analysis | Test | Details |
|---|---|---|
| Panel C: Transition differences | Paired t-test | Per-cell ON vs OFF transition probability across mice (min 3 valid observations). |
| Panels D-F: Epoch fractions | Paired t-test | Pre vs During laser, averaged across 14 stimulations per mouse. |

Mouse selection: >= 50% surface licking in pre-stim period (before 10-min mark). Stim timing: ISI = 80 s.

---

### SuppFig4_dSPN_Opto_BoutDur_TransProb

**Supplementary Figure 4 | dSPN optogenetic stimulation: transition probabilities and bout durations.** (**A**-**C**) Mean transition probability matrices during laser-OFF (A), laser-ON (B), and their difference (C) for dSPN (Drd1-ChR2) stimulation (N = 11 mice). Asterisks in C indicate significant changes (paired t-test, P < 0.05). (**D**-**H**) Bout duration CDFs for grooming (D), body licking (E), wall licking (F), floor licking (G), and locomotion (H) during laser-ON (purple) versus laser-OFF (gray).

#### Statistics

| Analysis | Test | Details |
|---|---|---|
| Panel C: Transition differences | Paired t-test | Per-cell ON vs OFF across mice (min 3 valid observations). |
| Panels D-H: Bout durations | Two-sample KS test | ON vs OFF bout duration distributions. Bouts classified as ON if > 50% of bout frames overlap laser-ON epochs. Theta = 1 frame. |

Data: drd1_opto cohort, AloneStim day. Per-mouse alignment via `opto_alignment_dict.pkl`.

---

### SuppFig5_DREADDs_AllCohorts

**Supplementary Figure 5 | Bidirectional DREADDs effects across all five cohorts.** Five rows (Drd1-hm4Di, Drd1-hm3Dq, Controls, A2a-hm4Di, A2a-hm3Dq) x four columns. Column 1: Grouped V-graphs for surface licking, self-directed licking, and non-licking fractions across the sandwich design (Pre-CNO, During-CNO, Post-CNO). Columns 2-4: Individual behavior V-graphs for floor licking, wall licking, and locomotion. Gray lines = individual mice; colored lines = group mean +/- SEM. Mouse counts indicated in row titles.

#### Statistics

| Analysis | Test | Details |
|---|---|---|
| Grouped categories | Paired t-test | Pre vs During and During vs Post per category per cohort. |
| Individual behaviors | Paired t-test | Pre vs During for FL, WL, Locomotion per cohort. |
| Comprehensive table | Paired t-test | All 9 behaviors x 5 cohorts printed to console. |

Sandwich detection: cocaineOnly -> cocaineCNO -> cocaineOnly. Session truncated to (50 - CNO gap) minutes. Undefined frames excluded. GCaMP and suboptimal-infection mice excluded.

---

### SuppFig6a_Stacked_AcrossSessions

**Supplementary Figure 6a | Stacked behavioral distribution across sessions.** Stacked bar chart of mean behavioral fraction across 8 sessions (S1-S3, C1-C5) for all 9 behaviors (N = 37 mice, 6 cohorts). Stack order from bottom: Stationary, Locomotion, Rearing, Body licking, Grooming, Wall licking, Floor licking, Undefined, Jump.

#### Statistics

Descriptive only.

---

### SuppFig6b_Stacked_WithinSession

**Supplementary Figure 6b | Within-session stacked behavioral time courses.** (**A**-**C**) Stacked area plots for saline 3 (A), cocaine 1 (B), and cocaine 5 (C) showing behavioral composition over session time (1-min window, 30-s stride, up to 27 min). Same color scheme and stacking order as Figure 6a.

#### Statistics

Descriptive only.

---

### SuppFig7_Entropy_Manipulations

**Supplementary Figure 7 | Behavioral entropy during optogenetic and chemogenetic manipulations.** (**A**) Switch entropy for iSPN opto: laser-OFF vs laser-ON. Individual lines (gray), group mean +/- SEM (OFF = gray, ON = light blue). (**B**) Transition rate (switches/min) for the same comparison. (**C**) Switch entropy across DREADDs sandwich design (Pre/CNO/Post) for all 5 cohorts. Cohort-colored V-graphs with individual mouse lines. (**D**) Transition rate for the same conditions.

#### Statistics

| Analysis | Test | Details |
|---|---|---|
| Panels A-B: iSPN opto | Paired t-test | Laser OFF vs ON, computed on concatenated epoch frames per mouse. |
| Panels C-D: DREADDs | Paired t-test | Pre vs CNO per cohort, whole-session entropy/rate. |

---

### SuppFig8_iSPN_Transient

**Supplementary Figure 8 | Transient photometry dynamics in iSPN and dSPN across cocaine sessions.** (**A**-**B**) CDFs of peak prominence (A) and inter-event interval (B) for iSPN photometry transients during saline (gray), early cocaine C1-C2 (orange), and late cocaine C3-C5 (dark red). (**C**-**D**) Same for dSPN. (**E**) Mean peak prominence per session for both pathways (iSPN dashed, dSPN solid). (**F**) Mean inter-event interval per session. Red shading marks cocaine sessions. N = 6 iSPN mice, N = 6 dSPN mice, ~31,000 total peaks.

#### Statistics

| Analysis | Method | Details |
|---|---|---|
| Peak detection | `scipy.signal.find_peaks` | Prominence >= 2, width >= 5 on z-scored photometry |
| CDFs | Descriptive | Pooled across mice and hemispheres per condition. |
| Panels E-F | Descriptive | Per-session per-mouse-hemisphere means, then group mean +/- SEM. |

Photometry exclusions applied: cA242m8 (all), hemispheres (c548m8 left, c548m11 left, cA242m5 left).

---

### SuppFig9_dSPN_vs_iSPN_BoutDur

**Supplementary Figure 9 | dSPN versus iSPN bout duration comparison.** (**A**-**B**) Bout duration CDFs for floor licking (A) and wall licking (B) during cocaine 5, comparing dSPN-recorded (red, solid; N = 3) and iSPN-recorded (dark red, dashed; N = 7) mice. (**C**) Mean floor licking bout duration across sessions (S1-S3, C1-C5) for both groups.

#### Statistics

| Analysis | Test | Details |
|---|---|---|
| Panels A-B: Duration CDFs | Two-sample KS test | dSPN vs iSPN bout durations at cocaine 5. |
| Panels A-B: Medians | Mann-Whitney U test (two-sided) | Same data. |
| Panel C | Descriptive | Per-mouse mean bout duration with SEM. Theta = 8 frames. |

---

### SuppFig10_Photom_BoutCorrelation

**Supplementary Figure 10 | Photometry amplitude versus bout duration correlation.** (**A**) Scatter: bout duration vs peak z-scored amplitude for dSPN up-regulated bouts during cocaine 3-5, with linear regression. (**B**) Same for iSPN. (**C**) Duration-matched comparison: bouts binned by duration quartiles; mean peak amplitude compared per bin. (**D**) Box plot of peak amplitude for dSPN vs iSPN across all cocaine bouts (~229 dSPN, ~571 iSPN bouts).

#### Statistics

| Analysis | Test | Details |
|---|---|---|
| Panels A-B | Pearson correlation | Duration vs peak amplitude (post-onset z-max, frame 45+). |
| Panel C | Descriptive | Quartile-binned mean +/- SEM per pathway. |
| Panel D | Mann-Whitney U (two-sided) | Peak amplitude: dSPN vs iSPN. |

Data: `master_table.pkl`, `up_regulated` bouts, cocaine 3-5.

---

### SuppFig11_Context_Specificity

**Supplementary Figure 11 | Context specificity of cocaine-induced behavioral changes.** (**A**) Fold change (salineOnly / saline baseline) per behavior, displayed as horizontal bars. Dashed line at 1.0 = no change. Asterisks: paired t-test, P < 0.05. (**B**) Paired comparison of surface licking fraction (FL + WL) between saline baseline and salineOnly. Gray lines = individual mice. (**C**) Paired comparison of behavioral entropy. N = mice with salineOnly sessions and >= 2 saline baselines (subset of 6-cohort dataset).

#### Statistics

| Analysis | Test | Details |
|---|---|---|
| Panel A: 9 behaviors | Paired t-test | Saline baseline vs salineOnly per behavior. |
| Panel B: Surface licking | Paired t-test | Grouped FL+WL. |
| Panel C: Entropy | Paired t-test | Occupancy-based entropy on full session. |

---

### SuppFig12_Velocity_Licking

**Supplementary Figure 12 | Velocity during surface licking confirms stationary classification.** (**A**) CDF of instantaneous velocity during surface licking frames, saline 3 vs cocaine 5. Dashed line at 0.5 cm/s threshold. (**B**) Velocity density histograms. (**C**) Fraction of licking frames below 0.5 cm/s across sessions (N = 37 mice, 6 cohorts). (**D**) Mean velocity during floor licking vs wall licking across sessions.

#### Statistics

| Analysis | Method | Details |
|---|---|---|
| Panel A | Descriptive | CDFs pooled across mice. Velocity = raw * FPS * 10 (cm/s). |
| Panel C | Descriptive | Per-mouse fraction below threshold, group mean +/- SEM. |
| Panel D | Descriptive | Per-mouse mean velocity per licking type, group mean +/- SEM. |

---

### SuppFig13_Grooming_Photometry

**Supplementary Figure 13 | Grooming and body licking photometry during the splash test.** Row 1 (dSPN): (**A**) Mean +/- SEM z-scored dF/F trace aligned to grooming bout onset. (**B**) Same for body licking bouts. (**C**) Bout-response-triggered (BRT) heatmap for grooming, sorted by bout duration. Row 2 (iSPN): (**D**-**F**) Same layout.

#### Statistics

Descriptive. Baseline-subtracted (first 2 s = 30 frames). Window: 3 s pre to 5 s post onset (120 frames at 15 FPS). BRT heatmap: magma colormap, vmin = -0.5, vmax = 1.5. Bout detection: min_bout_dur = 3 frames, no preceding-behavior constraint. Both hemispheres pooled after `photom_exclusions`.

---

### SuppFig14_TimeWarped_SplashTest

**Supplementary Figure 14 | Time-warped splash test photometry.** Row 1 (dSPN): (**A**) Time-warped mean +/- SEM for body licking bouts: pre-onset (3 s) preserved, bout segment interpolated to 2 s, post-bout tail (2 s) preserved. Dashed lines mark bout start (S) and end (E). (**B**) Quartile split: raw traces grouped by bout duration (Q1-Q4). (**C**) Time-warped BRT heatmap sorted by original duration. Row 2 (iSPN): (**D**-**F**) Same.

#### Statistics

Descriptive. Time warping: bout segment linearly interpolated to 30 frames (2 s). Pre-bout = 45 frames (3 s), tail = 30 frames (2 s). Total warped trace = 105 frames. Quartile boundaries at 25th, 50th, 75th percentile of bout duration.

---

## Impact / Additional Analyses

---

### Impact_CompositeSummary

**Composite Summary | Key results across analyses.** (**A**) Surface licking heatmap (mice x sessions, sorted by cocaine 5; Reds colormap, N = 37). (**B**) Switch entropy time course (mean +/- SEM). (**C**-**D**) Opto transition difference matrices (ON - OFF) for iSPN (C) and dSPN (D). RdBu_r colormap. (**E**-**H**) DREADDs V-graphs: Drd1-hm4Di (E), Drd1-hm3Dq (F), A2a-hm4Di (G), A2a-hm3Dq (H). Paired t-test significance on Pre vs CNO.

#### Statistics

| Analysis | Test | Details |
|---|---|---|
| Panels A-D | Descriptive | Heatmap, timecourse, difference matrices. |
| Panels E-H | Paired t-test | Pre-CNO vs During-CNO surface licking. |

---

### Impact_DoseResponse

**Dose-Response | Sigmoid fit of surface licking sensitization.** (**A**) Surface licking across all sessions with individual trajectories (gray) and group mean +/- SEM (N = 37 mice, 6 cohorts). (**B**) Cocaine sessions C1-C5 with 4-parameter sigmoid fit: L/(1+exp(-k(x-x0)))+b. EC50 = half-maximal sensitization session (~2.7).

#### Statistics

| Analysis | Method | Details |
|---|---|---|
| Sigmoid fit | `scipy.optimize.curve_fit` | Parameters: L [0,1], k [0,10], x0 [0,6], b [0,0.5]. Fitted to group means. |
| EC50 | x0 parameter | ~2.71 sessions (between C2 and C3). |

Descriptive model. No formal hypothesis test.

---

### Impact_IndividualTrajectories

**Individual Trajectories | Per-mouse behavioral spaghetti plots.** (**A**-**I**) Individual mouse trajectories (gray) and group mean +/- SEM for all 9 behaviors across S1-S3, C1-C5 (N = 37 mice, 6 cohorts). Saline region shaded.

#### Statistics

Descriptive only.

---

### Impact_TransitionFlow

**Transition Flow Diagrams | Behavioral transition patterns during optogenetic stimulation.** (**A**-**B**) iSPN laser-OFF (A) vs laser-ON (B) showing top 8 transitions as curved arrows (width proportional to count). (**C**-**D**) Same for dSPN. Transition probabilities annotated on arrows.

#### Statistics

| Analysis | Method | Details |
|---|---|---|
| Transition matrices | Raw counts | Pooled across mice per ON/OFF condition. Row-normalized for probability annotations. |
| Transition entropy | Shannon entropy | H = -sum(p log2(p)) on flattened normalized matrix. |

Descriptive. No formal hypothesis tests.

---

### Impact_ForestPlot

**DREADDs Forest Plot | Effect sizes for surface licking.** Cohen's d (paired, CNO - Pre-CNO) for surface licking across 5 DREADDs cohorts. Points = d, horizontal lines = 95% CI. Vertical dashed line at d = 0. Expected directions: Drd1-hm4Di (inhibit dSPN, d < 0), Drd1-hm3Dq (excite dSPN, d > 0), Controls (d ~ 0), A2a-hm4Di (inhibit iSPN, d > 0), A2a-hm3Dq (excite iSPN, d < 0).

#### Statistics

| Analysis | Method | Details |
|---|---|---|
| Cohen's d | Paired | d = mean(CNO-Pre) / SD(CNO-Pre, ddof=1). |
| 95% CI | t-distribution | SE_d = sqrt(1/n + d^2/(2n)); CI = d +/- t_crit * SE_d. |
| P-value | Paired t-test | Pre vs CNO surface licking per cohort. |

---

### Impact_BoutDecomposition

**Bout Decomposition | Rate versus duration across sensitization.** Row 1 (Floor licking): (**A**) Bout initiation rate (bouts/min). (**B**) Mean bout duration (seconds). (**C**) Scatter: delta rate vs delta duration (C5 - S3). Row 2 (Wall licking): (**D**-**F**) Same layout. N = 37 mice, 6 cohorts.

#### Statistics

| Analysis | Test | Details |
|---|---|---|
| Panels A-B, D-E | Paired t-test | Saline 3 vs cocaine 5 for rate and duration. |
| Panels C, F | Pearson r | Delta rate vs delta duration (C5 - S3). |

Bout detection: `segment_bouts`, theta = 1 frame. Rate = bouts/session duration in minutes.

---

## General Statistical Notes

1. **Error metric:** SEM (SD/sqrt(N), ddof=0) throughout.
2. **Paired t-tests:** Two-tailed `scipy.stats.ttest_rel` for within-subject comparisons.
3. **KS tests:** `scipy.stats.ks_2samp` for distribution comparisons.
4. **Mann-Whitney U:** `scipy.stats.mannwhitneyu` for between-group comparisons.
5. **Correlations:** Pearson (`pearsonr`) for linear, Spearman (`spearmanr`) for monotonic trends.
6. **Multiple comparisons:** No global correction applied; each comparison reported independently.
7. **Significance thresholds:** *P < 0.05, **P < 0.01, ***P < 0.001.
8. **Behavior encoding:** 9-class canonical STEREO (Dec24): Jump(0), Undefined(1), Floor licking(2), Wall licking(3), Grooming(4), Body licking(5), Rearing(6), Locomotion(7), Stationary(8).
9. **Grouped behaviors:** Surface licking = FL(2)+WL(3); Self-directed = Grooming(4)+Body licking(5); Non-licking = all others.
10. **Photometry:** Z-scored dF/F. Peak detection: `find_peaks(prominence=2, width=5)`. BRT: 3s pre, 5s post, 2s baseline subtraction.
11. **DREADDs sandwich:** Pre-CNO, During-CNO, Post-CNO consecutive cocaine sessions. Analysis window = (50 - per-mouse CNO gap) minutes.
12. **Opto timing:** iSPN = 14 stims from 10 min, ISI = 80 s. dSPN = per-mouse alignment dict, ISI = 80 s.
13. **Velocity conversion:** Raw topcam velocity * FPS * 10 = cm/s.
14. **Default cohort set (behavioral analyses):** All 6 cohorts — drd1_hm4di, drd1_hm3dq, controls, a2a_hm4di, a2a_hm3dq, a2a_opto (N = 37 mice with complete saline + cocaine sessions).

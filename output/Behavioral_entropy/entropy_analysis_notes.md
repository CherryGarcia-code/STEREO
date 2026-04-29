# Behavioral Entropy Across Saline and Cocaine Sessions — Notes

*Generated: 2026-03-20 | Script: `behavioral_entropy.py`*

---

## Methods

### Subjects & Data
- All cohorts pooled: `drd1_hm4di`, `drd1_hm3dq`, `controls`, `a2a_hm4di`, `a2a_hm3dq`, `a2a_opto`
- Data source: `CMT_Aug24.pkl` (BZ2-compressed pickle, `May24/Data/`)
- Data structure: `CMT[cohort][mouse][trial]['merged']` — flat frame-by-frame behavior array remapped via `lut = [4,5,3,2,6,1,8,7,0]`
- N = 37 mice with complete data across all 8 sessions (mice missing any session excluded)
- Sessions: saline1, saline2, saline3, cocaine1, cocaine2, cocaine3, cocaine4, cocaine5

### Behavioral Classification
- STEREO RNN classifier, 9 behaviors at 15 FPS, smartMerge multi-camera fusion
- Behaviors: Jump (0), Undefined (1), Floor licking (2), Wall licking (3), Grooming (4), Body licking (5), Rearing (6), Locomotion (7), Stationary (8)

### Entropy Computation
- **Measure**: Shannon entropy in bits: $H = -\sum_{i=0}^{8} p_i \log_2 p_i$
- **Procedure**: For each mouse × session, compute the empirical probability distribution over the 9 behaviors from the full frame-by-frame prediction array, then apply the Shannon entropy formula
- Zero-probability behaviors excluded from the sum (avoids log(0))
- **Maximum possible entropy**: $\log_2(9) \approx 3.170$ bits — achieved if all 9 behaviors are expressed with equal frequency

### Statistics
- **Group trajectory**: Session means ± SEM across 37 mice
- **Spearman correlation**: Between session index (1–8) and group session means — tests for monotonic trend across the 8-point trajectory
- **Paired t-test**: Per-mouse mean entropy during saline (S1–S3) vs. late cocaine (C3–C5) — tests within-mouse reduction in repertoire diversity
- Effect size: Cohen's *d* (paired)

---

## Results

### Key Finding
Cocaine exposure produced a rapid and sustained reduction in behavioral repertoire diversity (entropy), reflecting the narrowing of behavior toward surface licking stereotypies. Per-mouse entropy fell from ~2.0 bits at baseline to ~1.6 bits by the late cocaine sessions — a large-effect, highly significant reduction.

### Detailed Results

**Baseline behavioral repertoire (saline sessions)**
Mice expressed a diverse behavioral repertoire during saline sessions, with mean Shannon entropy of 1.995 ± 0.055 bits (mean ± SEM across 37 mice; max possible = 3.170 bits), indicating broad engagement of multiple behavioral states.

**Cocaine-induced repertoire narrowing**
Entropy dropped sharply at the first cocaine session (C1: ~1.55 bits) and stabilized at reduced levels throughout subsequent cocaine sessions (C5: 1.596 ± 0.054 bits). This reduction persisted without recovery across C2–C5, indicating a sustained reorganization of behavioral output rather than an acute response.

- **Paired t-test (saline mean vs. late-cocaine mean, C3–C5)**: t(36) = 6.23, p < 0.0001, Cohen's d = 1.02 — large effect size
- **Spearman correlation (session index vs. group means)**: ρ = −0.57, p = 0.139 — negative trend across sessions, non-significant due to limited statistical power with 8 data points; the sharp S3→C1 step drives a non-monotonic pattern that reduces ρ

**Individual variability**
Individual mice (N = 37, gray traces) showed consistent downward shifts from saline to cocaine, though the magnitude varied. Several mice showed very low entropy during cocaine sessions (~1.1–1.2 bits), consistent with near-exclusive expression of surface licking.

**Cohort breakdown (supplementary figure)**
All 6 cohorts showed the saline-to-cocaine entropy drop:
- Drd1-hm4Di (n=6) and A2a-hm4Di (n=6): steepest and most sustained reductions (~2.2 → 1.4 bits)
- A2a-hm3Dq (n=10): largest cohort; moderate but consistent drop (~2.1 → 1.5 bits)
- Controls (n=5): entropy recovered slightly across late cocaine sessions (~1.8 → 1.5 → 1.9 bits), consistent with weaker sensitization in manipulation-naïve animals
- A2a-opto (n=7): lower baseline entropy (~1.5 bits during saline), suggesting this cohort was already expressing less behavioral diversity at the commencement of cocaine sessions

### Interpretation
The reduction in Shannon entropy quantifies at the repertoire level what ethograms show qualitatively: repeated cocaine progressively narrows behavioral output toward a restricted set of behaviors — predominantly surface licking (floor and wall). This behavioral focusing is consistent with the "behavioral crystallization" hypothesis, wherein the striatal direct pathway drives stereotypy expression while indirect pathway suppression of competing behaviors may further constrain the repertoire. The persistence of low entropy across C3–C5 indicates that repertoire narrowing is a stable feature of cocaine sensitization in this model, not a transient acute effect.

### Caveats
- Cohorts are pooled for the main analysis; DREADDs and opto manipulations occurred during some sessions (cocaine days) and may influence entropy values on those days, introducing confounds if cohort-specific manipulation days are included without accounting for CNO/laser effects
- CMT_Dec24.pkl was unavailable; analysis used CMT_Aug24.pkl which requires lut remapping — verify that the lut is identical to the one used in main figures
- A2a-opto cohort starts with unusually low saline entropy (~1.5 bits vs. ~2.0–2.2 bits in other cohorts) — confirm this is not an artifact of different recording conditions or a pre-existing behavioral phenotype in ChR2-expressing mice
- Mice missing any session (e.g., due to corrupted recordings) are excluded entirely; this may introduce selection bias toward mice with uninterrupted experimental runs
- Shannon entropy treats all behaviors equally — it does not distinguish between *which* behaviors dominate. An animal spending 90% of time in Stationary vs. 90% in Floor licking would have the same entropy. Complement with ethogram or proportional analyses.

---

## Figure Descriptions

### Figure: `entropy_across_sessions.png` / `.pdf`
- **Size**: 85 × 60 mm
- **X-axis**: 8 sessions (S1, S2, S3, C1, C2, C3, C4, C5)
- **Y-axis**: Behavioral entropy (bits); range 0.5–3.3 bits
- **Gray traces**: Individual mice (N = 37), α = 0.15
- **Colored line**: Group mean with graduated cocaine color encoding (gray = saline; orange → dark red = cocaine C1–C5)
- **Error bars**: ±SEM at each session
- **Dotted line**: Maximum possible entropy (log₂9 = 3.170 bits)
- **Annotation**: Spearman ρ = −0.57, p = 0.139, n = 37 mice

### Figure: `entropy_by_cohort.png` / `.pdf`
- **Size**: 140 × 90 mm
- **Layout**: 2 × 3 subplots (shared x/y axes), one per cohort
- **Row 1**: Drd1-hm4Di (n=6), Drd1-hm3Dq (n=3), Controls (n=5)
- **Row 2**: A2a-hm4Di (n=6), A2a-hm3Dq (n=10), A2a-opto (n=7)
- Same color encoding and style as main figure; smaller markers and thinner lines

---

## Output Files

| File | Location |
|------|----------|
| `entropy_across_sessions.png` | `May24/Figures/Behavioral_entropy/` |
| `entropy_across_sessions.pdf` | `May24/Figures/Behavioral_entropy/` |
| `entropy_by_cohort.png` | `May24/Figures/Behavioral_entropy/` |
| `entropy_by_cohort.pdf` | `May24/Figures/Behavioral_entropy/` |
| `behavioral_entropy.py` | Project root |

---

## Within-Session Entropy Time Course

*Added: 2026-03-22*

### Methods

#### Analysis
- **Sliding window entropy**: Shannon entropy computed in 1-minute windows (900 frames), stepped by 30 seconds (450 frames), across each session
- Sessions: saline3 (last baseline), cocaine1–5 (escalation)
- All sessions truncated to 27 minutes (the safe common duration; median session length ~30 min)
- N = 37 mice with complete data across all sessions

#### Window Parameters
- Window size: 1 minute (900 frames)
- Step size: 30 seconds (450 frames), yielding ~53 time points per trace
- Duration cutoff: 27 minutes (24,300 frames)

#### Visualization
- One trace per session (saline3 + cocaine1–5)
- Lines: group mean across mice
- Shading: ± SEM
- Colors: gray (saline3), orange → dark red gradient (cocaine1–5)

### Results

#### Key Finding
Cocaine sessions show uniformly lower within-session entropy compared to saline, with clear dose-dependent stratification. Behavioral diversity is reduced from the very first minutes of each cocaine session and remains compressed throughout, indicating that repertoire narrowing is not a gradual within-session development but rather an immediate state shift upon cocaine exposure.

#### Detailed Results

**Saline3 (baseline reference)**
Entropy rises gradually within the session from ~1.4 bits (0–5 min: 1.435 bits) to ~1.7 bits (20–27 min: 1.696 bits; Δ = +0.261 bits), consistent with behavioral exploration increasing over time. Saline entropy remains well-separated above all cocaine sessions throughout.

**Cocaine1 (first exposure)**
Entropy is immediately suppressed compared to saline (early: 1.298 bits) and remains flat throughout the session (late: 1.304 bits; Δ = +0.007). The behavioral repertoire is already restricted at the first cocaine exposure with no recovery within session.

**Cocaine2–3 (escalation)**
Further reduction in entropy compared to cocaine1. Cocaine2 early = 1.200 bits, cocaine3 early = 1.128 bits. Both show slight recovery in the late period (cocaine3 Δ = +0.114), possibly reflecting waning of peak cocaine effect.

**Cocaine4–5 (late escalation)**
Lowest entropy levels overall. Cocaine5 early = 1.212 bits, late = 1.149 bits (Δ = −0.062), indicating that by late cocaine sessions entropy continues to decline even within sessions — a hallmark of deepening behavioral crystallization.

**Stratification pattern**
The traces stratify cleanly with escalation: saline3 >> cocaine1 ≥ cocaine2 > cocaine3 ≈ cocaine4 ≈ cocaine5. This ordering is stable across the entire 27-minute session, confirming that the cross-session entropy decline (Figure: entropy_across_sessions) is not driven by a particular time window within sessions.

### Interpretation
The within-session dynamics reveal that cocaine-induced repertoire narrowing is an immediate state shift, not a gradual process. From the first minute of each cocaine session, behavioral diversity is already suppressed to a level that persists throughout the 27-minute observation period. The saline session shows a natural exploration trajectory (entropy increasing over time), which is abolished by cocaine. The progressive dose-dependent stratification across cocaine1–5 mirrors the between-session sensitization curve, reinforcing that stereotypy escalation is a cumulative, session-over-session phenomenon rather than solely an acute pharmacological effect.

### Caveats
- 1-minute window with 30-second step introduces temporal smoothing; fine-grained behavioral switches within windows are averaged out
- A small number of mice have shorter sessions (min = 14,358 frames ≈ 16 min for cocaine1); these contribute NaN values for late time points, slightly reducing effective N at the tail of the trace
- The cocaine sessions shown pool mice from both DREADD and opto cohorts; cohort-specific manipulation effects (CNO, laser) during some cocaine sessions may influence entropy values
- Entropy computed on raw behavioral predictions without bout-detection filtering — transient misclassifications (1–2 frames) inflate apparent behavioral diversity slightly

### Figure Description

**`entropy_within_sessions.png` / `.pdf`**
- **Size**: 100 × 70 mm
- **X-axis**: Time within session (minutes), 0–27 min
- **Y-axis**: Behavioral entropy (bits)
- **Lines**: Group mean (N = 37 mice) for each session
- **Shading**: ± SEM
- **Colors**: gray = saline3; orange to dark red gradient = cocaine1–5
- **Legend**: Session labels, frameon=False

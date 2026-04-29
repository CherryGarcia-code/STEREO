# Switch Entropy Analysis — Notes

*Generated: 2026-03-23 | Script: `switch_entropy.py`*

---

## Methods

### Subjects & Data
- All cohorts pooled: `drd1_hm4di`, `drd1_hm3dq`, `controls`, `a2a_hm4di`, `a2a_hm3dq`, `a2a_opto`
- Data source: `CMT_Aug24.pkl` (BZ2-compressed pickle, `May24/Data/`)
- N = 37 mice with complete data across all 8 sessions (mice missing any session excluded)
- Sessions: saline1–3, cocaine1–5

### Behavioral Classification
- STEREO RNN classifier, 9 behaviors at 15 FPS, smartMerge multi-camera fusion
- Behaviors: Jump (0), Undefined (1), Floor licking (2), Wall licking (3), Grooming (4), Body licking (5), Rearing (6), Locomotion (7), Stationary (8)

### Switch Entropy Computation
- **Measure**: Shannon entropy of the behavioral switch destination distribution
- **Procedure**: (1) Identify all frames where the classified behavior changes from the preceding frame (switch points). (2) Record the "destination" behavior at each switch point. (3) Compute the empirical probability distribution over the 9 behaviors from these destinations. (4) Apply Shannon entropy: $H_{switch} = -\sum_{i=0}^{8} p_i \log_2 p_i$
- Minimum 2 switches required per window; otherwise NaN
- **Maximum possible switch entropy**: $\log_2(9) \approx 3.170$ bits — achieved if switches are equally distributed across all 9 behaviors
- **Key distinction from behavioral entropy**: Behavioral entropy measures the diversity of the *time-fraction* distribution (how much time is spent in each behavior). Switch entropy measures the diversity of *transitions* (what behavior does the animal switch to). An animal that spends 80% of time in surface licking but still switches to diverse behaviors during the remaining 20% would have low behavioral entropy but relatively high switch entropy.

### Within-Session Time Course
- Sliding window: 1 minute (900 frames), stepped by 30 seconds (450 frames)
- Sessions plotted: saline3, cocaine1–5
- Duration truncated at 27 minutes (~53 time bins)

### Statistics
- **Spearman correlation**: Session index (1–8) vs. group session means — monotonic trend test
- **Paired t-test**: Per-mouse mean switch entropy during saline (S1–S3) vs. late cocaine (C3–C5)
- Effect size: Cohen's *d* (paired)

---

## Results

### Key Finding
Switch entropy — the diversity of behavioral transition destinations — decreases progressively across cocaine sessions, with a strong monotonic trend (Spearman ρ = −0.90, p = 0.002). This indicates that cocaine not only restricts *which* behaviors dominate (behavioral entropy) but also restricts *where* the animal transitions when it does switch, reflecting a deepening rigidity of the behavioral sequence structure. Critically, the within-session time courses reveal that cocaine1–2 maintain higher switch diversity than cocaine3–5, with all cocaine sessions showing within-session decreases — a pattern distinct from behavioral entropy where all cocaine sessions were uniformly low.

### Detailed Results

**Across-session trajectory**
- Mean switch entropy during saline: 2.494 ± 0.041 bits (mean ± SEM, N = 37)
- Mean switch entropy during cocaine5: 2.081 ± 0.030 bits
- Switch entropy declined monotonically from saline through cocaine5
- Spearman ρ = −0.90, p = 0.002 — highly significant monotonic trend (unlike behavioral entropy where ρ = −0.57, p = 0.139, n.s.)
- Paired t-test (saline mean vs. late cocaine mean, C3–C5): t(36) = 9.79, p < 0.0001, Cohen's d = 1.61 — very large effect

**Within-session time courses**

*Saline3 (baseline)*:
Entropy increases within session from 1.971 bits (early, 0–5 min) to 2.219 bits (late, 20–27 min; Δ = +0.248), paralleling the behavioral entropy exploration ramp. Under saline, mice explore increasingly diverse transition patterns over time.

*Cocaine1–2 (early cocaine)*:
Start at high switch diversity (~2.04–2.08 bits) comparable to early saline, but *decrease* within session (cocaine1 Δ = −0.145; cocaine2 Δ = −0.128). Cocaine1–2 remain separated above cocaine3–5 for most of the session, converging only late. This gradual within-session stratification is the key difference from behavioral entropy, where cocaine1 was immediately indistinguishable from cocaine3–5.

*Cocaine3–5 (late cocaine)*:
Start lower (~1.90–1.97 bits) and decline further within session (cocaine5 Δ = −0.148). By late in the session, cocaine5 reaches ~1.75 bits — the lowest switch diversity observed.

**Per-cohort breakdown**
All 6 cohorts show the saline > cocaine stratification pattern with within-session narrowing. Drd1-hm3Dq (n=3) shows the strongest separation between sessions. A2a-opto (n=7) shows the weakest stratification, consistent with its lower baseline behavioral entropy noted previously.

### Interpretation
Switch entropy captures an aspect of behavioral crystallization distinct from simple repertoire narrowing. While behavioral entropy shows that cocaine shifts the time-fraction distribution toward surface licking from the very first session, switch entropy reveals that transition *flexibility* degrades more gradually. During cocaine1–2, the animal still switches to diverse behaviors when it transitions — it just spends more time in surface licking between switches. By cocaine3–5, even the transitions themselves become stereotyped: when the animal does switch, it overwhelmingly switches to the same few behaviors (likely alternating between floor licking, wall licking, and brief locomotion). This progressive loss of transition diversity aligns with the concept of behavioral crystallization as a multi-stage process: first the dominant behavior captures more time (behavioral entropy drops), then the sequential structure rigidifies (switch entropy drops).

The highly significant Spearman trend (ρ = −0.90, p = 0.002) — compared to the non-significant trend for behavioral entropy (ρ = −0.57, p = 0.139) — underscores that switch entropy captures a more gradual, monotonic sensitization trajectory, making it a potentially more sensitive marker of progressive stereotypy development.

### Caveats
- Switch entropy is sensitive to classifier noise: brief misclassifications (1–2 frame glitches) create spurious "switches" that inflate entropy. The STEREO smartMerge pipeline mitigates but does not eliminate this.
- The minimum of 2 switches per 1-minute window is lenient; very few windows should fail this, but late cocaine sessions with sustained licking bouts could occasionally have windows with <2 switches.
- Pooling across DREADD/opto cohorts: manipulation days may affect switch patterns differently than time-fraction distributions.
- Self-transitions (staying in the same behavior) are excluded by definition of switch detection (`diff != 0`), so entropy is computed only over behavior-change events.

---

## Figure Descriptions

### Figure: `switch_entropy_across_sessions.png` / `.pdf`
- **Size**: 85 × 60 mm
- **X-axis**: 8 sessions (S1–S3, C1–C5)
- **Y-axis**: Switch entropy (bits)
- **Gray traces**: Individual mice (N = 37), α = 0.15
- **Colored line**: Group mean with cocaine gradient (gray → dark red)
- **Error bars**: ± SEM
- **Dotted line**: Max entropy (log₂9 = 3.170 bits)
- **Annotation**: Spearman ρ = −0.90, p = 0.002, n = 37 mice

### Figure: `switch_entropy_within_sessions.png` / `.pdf`
- **Size**: 100 × 70 mm
- **X-axis**: Time within session (min), 0–27 min
- **Y-axis**: Switch entropy (bits)
- **Lines**: Group mean per session (saline3 + cocaine1–5)
- **Shading**: ± SEM
- **Colors**: gray = saline3; orange → dark red = cocaine1–5

### Figure: `switch_entropy_by_cohort.png` / `.pdf`
- **Size**: 170 × 100 mm
- **Layout**: 2 × 3 subplots, one per cohort, shared legend
- **Each panel**: Within-session time course for all 6 sessions
- **Row 1**: Drd1-hm4Di (n=6), Drd1-hm3Dq (n=3), Controls (n=5)
- **Row 2**: A2a-hm4Di (n=6), A2a-hm3Dq (n=10), A2a-opto (n=7)

---

## Output Files

| File | Location |
|------|----------|
| `switch_entropy_across_sessions.png/.pdf` | `May24/Figures/Switch_entropy/` |
| `switch_entropy_within_sessions.png/.pdf` | `May24/Figures/Switch_entropy/` |
| `switch_entropy_by_cohort.png/.pdf` | `May24/Figures/Switch_entropy/` |
| `switch_entropy.py` | Project root |

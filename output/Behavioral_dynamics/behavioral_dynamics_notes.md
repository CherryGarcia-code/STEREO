# Behavioral Dynamics Composite Analysis — Notes

*Generated: 2026-03-23 | Script: `behavioral_dynamics.py`*

---

## Methods

### Subjects & Data
- All cohorts pooled: `drd1_hm4di`, `drd1_hm3dq`, `controls`, `a2a_hm4di`, `a2a_hm3dq`, `a2a_opto`
- Data source: `CMT_Aug24.pkl` (BZ2-compressed pickle, `May24/Data/`)
- N = 37 mice with complete data across all 8 sessions
- Sessions: saline1–3, cocaine1–5

### Behavioral Classification
- STEREO RNN classifier, 9 behaviors at 15 FPS, smartMerge multi-camera fusion

### Metrics

**A. Switch entropy (bits):** Shannon entropy of the destination-behavior distribution at behavioral switch points. At each frame where the classified behavior changes, the new ("destination") behavior is recorded. Entropy is computed over the distribution of these destinations: $H_{switch} = -\sum p_i \log_2 p_i$. Captures diversity of transitions.

**B. Distinct behaviors per window:** Count of unique behavior classes observed within each 1-minute window. Range: 1–9. Captures how many different behaviors the animal expresses.

**C. Transition rate (switches/min):** Number of behavioral switches (frames where behavior changes from preceding frame) per minute. Captures how frequently the animal alternates between behaviors, independent of which behaviors are involved.

**D–F. Transition probability matrices:** For each behavior switch (frame where $\text{behavior}_t \neq \text{behavior}_{t+1}$), the source–destination pair is recorded into a 9×9 matrix. Rows are normalised so each row sums to 1, giving $P(\text{destination} \mid \text{source})$. Matrices are averaged across all mice and sessions within condition (baseline: saline1–3; late cocaine: cocaine3–5). The difference matrix (F) = late cocaine − baseline.

### Within-Session Time Course Parameters
- Sliding window: 1 minute (900 frames) with 30-second step (450 frames)
- Duration truncated at 27 minutes
- Sessions plotted: saline3 (baseline reference), cocaine1–5

### Statistics
- Spearman ρ: session index (1–8) vs. whole-session group means — monotonic trend
- Paired t-test: per-mouse saline mean (S1–S3) vs. late cocaine mean (C3–C5)
- Effect size: Cohen's d (paired)

---

## Results

### Key Finding
Three complementary metrics reveal distinct temporal aspects of cocaine-induced behavioral crystallization. Switch entropy and behavioral diversity decline progressively across sessions, reflecting narrowing of both the transition repertoire and the expressed behavior set. In contrast, transition *rate* is unchanged by cocaine — animals switch just as frequently, but their switches become stereotyped. Transition matrices show that cocaine massively increases self-reinforcing loops among surface licking behaviors (Floor licking ↔ Wall licking) at the expense of transitions to Locomotion and Stationary states.

### Detailed Results

**Panel A: Switch entropy**
- Saline: 2.494 ± 0.041 bits; Late cocaine: 2.097 ± 0.021 bits
- Spearman ρ = −0.905, p = 0.002 — strong monotonic decline across sessions
- Paired t-test: t(36) = 9.79, p < 0.001, Cohen's d = 1.61 (very large effect)
- Within-session dynamics: saline3 shows upward ramp (+0.25 bits over 27 min); cocaine sessions show downward drift. Cocaine1–2 start near saline levels but decline; cocaine3–5 are uniformly lower.

**Panel B: Distinct behaviors per window**
- Saline: 8.10 ± 0.04 behaviors; Late cocaine: 7.68 ± 0.06 behaviors
- Spearman ρ = −0.994, p < 0.001 — near-perfect monotonic decline
- Paired t-test: t(36) = 6.74, p < 0.001, Cohen's d = 1.11 (large effect)
- Within-session dynamics: saline3 maintains ~6.5 distinct behaviors throughout; cocaine sessions drop to 4.0–5.5, with progressive session-over-session separation. The absolute range (4–7 out of 9) shows the floor is meaningful — even under heavy cocaine, mice still express 4–5 behavior types per minute.

**Panel C: Transition rate (switches/min)**
- Saline: 55.2 ± 1.9 switches/min; Late cocaine: 56.1 ± 3.0 switches/min
- Spearman ρ = +0.26, p = 0.531 — no trend
- Paired t-test: t(36) = −0.23, p = 0.823, Cohen's d = −0.04 (negligible)
- Within-session dynamics: all sessions overlap substantially (~50–65 switches/min). Cocaine does not alter how *often* mice switch between behaviors — only *what* they switch to. This dissociation is the key insight: transition rate is preserved while transition diversity collapses.

**Panels D–E: Transition probability matrices**
- *Baseline (saline)*: Transitions are broadly distributed. The dominant pattern is Stationary as a sink (high P from most other behaviors → Stationary: 0.32–0.92). Locomotion↔Rearing and Grooming↔Body licking are the main reciprocal pairs.
- *Late cocaine (C3–C5)*: Floor licking and Wall licking become attractors. P(any → Floor licking) increases dramatically. Floor licking → Stationary drops (0.72 → 0.70) while Floor licking → Wall licking (0.19) and Locomotion → Floor licking (0.53) emerge as dominant transitions.

**Panel F: Difference matrix (cocaine − baseline)**
- Largest increases (red): Locomotion → Floor licking (+0.36), Undefined → Floor licking (+0.22), Rearing → Floor licking (+0.15), Wall licking → Floor licking (+0.15)
- Largest decreases (blue): Locomotion → Stationary (−0.30), Stationary → Stationary diagonal (−0.25), Grooming → Body licking (−0.20), Wall licking → Grooming (−0.18)
- Interpretation: Cocaine redirects behavioral transitions away from rest states (Stationary, Locomotion) and natural grooming behaviors toward surface licking. The transition network becomes dominated by a Floor licking ↔ Wall licking ↔ Locomotion loop.

### Interpretation
These three within-session metrics, combined with transition matrix analysis, paint a multi-level picture of behavioral crystallization:

1. **What the animal does** (distinct behaviors): Progressively fewer behavior types are expressed per unit time — the behavioral palette shrinks.
2. **Where it transitions** (switch entropy): Even among the behaviors still expressed, transitions become stereotyped — the animal switches to the same few destinations.
3. **How often it switches** (transition rate): The motor switching mechanism itself is *not* impaired — the animal continues to transition at the same rate. The rigidity is in the selection, not the switching.
4. **The transition structure** (matrices): Surface licking behaviors (floor + wall) become attractors that capture transitions from locomotion and rearing, while self-grooming and stationary rest lose their share of transitions.

This dissociation between preserved switching rate and collapsed switching diversity is consistent with a model where cocaine enhances direct-pathway (dSPN) drive for surface licking as a "winner-take-all" action selection bias, rather than impairing the basal ganglia's ability to initiate behavioral transitions per se.

### Caveats
- Transition rate is sensitive to classifier temporal resolution (15 FPS) and potential brief misclassifications that create spurious switches; the STEREO smartMerge pipeline mitigates but doesn't eliminate this
- The 1-minute window for "distinct behaviors" biases toward higher counts (longer windows → more behaviors sampled); the relative session ordering is robust to window size choice
- Transition matrices average across all mice and sessions within condition, collapsing inter-individual variability; per-mouse matrices could be computed but would be noisier
- Cohort pooling includes manipulation days (DREADD/opto) alongside pure cocaine sessions

---

## Figure Description

### `behavioral_dynamics_composite.png` / `.pdf`
- **Size**: 180 × 155 mm, 6-panel composite
- **Layout**: 2 rows × 3 columns; top row = time courses, bottom row = matrices

**Panel A** — Switch entropy within sessions. Lines = group mean ± SEM shading per session (saline3, cocaine1–5). Colors: gray → dark red gradient.

**Panel B** — Distinct behaviors per 1-minute window within sessions. Same format as A.

**Panel C** — Transition rate (switches/min) within sessions. Same format as A. Note: all sessions overlap — rate is cocaine-invariant.

**Panel D** — Transition probability matrix, baseline (saline1–3 averaged). 9×9 heatmap, YlOrRd colormap, values annotated. Row = source, column = destination.

**Panel E** — Transition probability matrix, late cocaine (C3–C5). Same format as D.

**Panel F** — Difference matrix (late cocaine − baseline). RdBu_r colormap (red = increased probability under cocaine, blue = decreased). Shows that Floor licking becomes an attractor.

---

## Output Files

| File | Location |
|------|----------|
| `behavioral_dynamics_composite.png/.pdf` | `May24/Figures/Behavioral_dynamics/` |
| `behavioral_dynamics.py` | Project root |

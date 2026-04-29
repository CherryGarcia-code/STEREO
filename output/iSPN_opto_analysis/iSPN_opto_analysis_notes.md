# iSPN Optogenetic Analysis — Research Notes

## Date
2025-03-23

## Overview
Analyzed three aspects of iSPN (A2a-Cre) optogenetic manipulation during cocaine-induced stereotypy:
1. Transition probability matrices (laser ON vs OFF)
2. Effects on grooming, rearing, and locomotion around laser epochs
3. Splash test grooming induction as a behavioral benchmark

## Data Sources
- **A2a opto cocaine stim**: `CTM_Aug24.pkl['a2a_opto']`, trials `cocaine6laserStim`/`cocaine8laserStim`
- **Splash test**: `CMT_May24.pkl` (all cohorts), baseline days vs splashTest day
- Behavior encoding (post-LUT): Jump(0), Undefined(1), Floor licking(2), Wall licking(3), Grooming(4), Body licking(5), Rearing(6), Locomotion(7), Stationary(8)
- Splash test encoding: Grooming(0), Body licking(1), Wall licking(2), Floor licking(3), Rearing(4), Back to camera(5), Stationary(6), Locomotion(7), Jump(8)

## Protocol
- **A2a opto**: 14 stimulations, ISI = 80s, epoch = 20s, first stim at ~10 min
- Mice selected based on >50% licking in pre-stim period (cocaine6 or cocaine8)
- N=7 mice after excluding 1 outlier (cA182m7: insufficient pre-stim licking)
- **Splash test**: N=19 mice across all cohorts, baseline = mean of 3 pre-days

## Key Results

### Analysis 1: Transition Probability Matrices
- 8/81 transition cells significantly different (p<0.05, paired t-test)
- **Dominant pattern**: During laser ON, transitions shift **away from wall licking** and **toward locomotion**
  - FlL → WlL: OFF=0.281, ON=0.090, p=0.006
  - Rer → WlL: OFF=0.414, ON=0.026, p=0.018
  - Udf → WlL: OFF=0.168, ON=0.012, p=0.049
  - FlL → Loc: OFF=0.643, ON=0.838, p=0.012
  - Rer → Loc: OFF=0.399, ON=0.784, p=0.016
  - Udf → Loc: OFF=0.549, ON=0.822, p=0.027
- iSPN stimulation breaks stereotypic transition loops by redirecting behavior toward locomotion

### Analysis 2a: iSPN Effects on Grooming, Rearing, Locomotion
- **Locomotion**: massive increase during stim, pre=0.113 → during=0.598, t(6)=−10.58, p<0.0001, d=4.32
- **Grooming**: significant increase, pre=0.020 → during=0.035, t(6)=−2.69, p=0.036, d=1.10
- **Rearing**: no significant change, p=0.57

### Analysis 2b: Splash Test Grooming Suppression
- **Grooming**: baseline=0.059 → splash=0.180, t(18)=−7.10, p<0.0001, d=1.67
- **VDB (all licking)**: baseline=0.156 → splash=0.372, t(18)=−8.01, p<0.0001, d=1.89
- The splash test reliably induces grooming/licking, confirming the behavioral assay validity

## Interpretation
- iSPN activation during cocaine stereotypy primarily **generates locomotion** (d=4.32), consistent with indirect pathway disinhibiting locomotor circuits
- The transition matrix analysis reveals that iSPN stim doesn't just suppress licking — it **redirects behavioral transitions toward locomotion** regardless of the current behavior
- The small but significant grooming increase (d=1.10) during stim may reflect displacement grooming or brief grooming bouts between locomotion episodes
- Wall licking transitions are most strongly suppressed, suggesting iSPN activation preferentially disrupts the stereotypic wall-licking component of the cocaine-induced behavioral syndrome

## Figures Generated
- `iSPN_transition_matrices.png/pdf` — 3-panel: OFF, ON, difference matrices
- `iSPN_behavior_V_graphs.png/pdf` — Pre/During/Post for grooming, rearing, locomotion
- `iSPN_behavior_dynamics.png/pdf` — Peri-stimulus time histograms
- `splash_test_grooming_suppression.png/pdf` — Baseline vs splash test
- `iSPN_opto_composite.png/pdf` — 9-panel composite (A-I)

## dSPN Analysis Status
- Script `dSPN_opto_analysis.py` created but requires `May25/Data/CTM_May25.pkl` and `May25/Data/opto_alignment_dict.pkl`
- These files are not present in the current workspace
- Analysis will compute: transition probability matrices + bout duration CDFs for VDB behaviors

# CLAUDE.md — Project Instructions for Claude

## Project Overview

Multi-cohort behavioral neuroscience study of cocaine-induced stereotypies in mice, using the STEREO automated behavioral classification system. The project tracks the development of pathological surface licking (floor/wall licking) across repeated cocaine exposures, investigates the neural correlates via fiber photometry in direct (Drd1/dSPN) and indirect (A2a/iSPN) striatal pathways, and tests causal roles via DREADDs (chemogenetics) and optogenetics.

## Skills

Eight specialized skills are available in `.claude/skills/`:

### Research Skills (Analysis & Figures)

| Skill | Folder | When to Invoke |
|-------|--------|---------------|
| **Research Visualizer** | `research-visualizer/` | Figure design, color choices, layout, labeling, multi-option visual proposals |
| **Research Statistician** | `research-statistician/` | Statistical test selection, effect sizes, multiple comparisons, results tables |
| **Research Notes Summarizer** | `research-notes-summarizer/` | Methods documentation, results summaries, scientific writing |

### Operations Skills (Running & Debugging)

| Skill | Folder | When to Invoke |
|-------|--------|---------------|
| **Analysis Runner** | `analysis-runner/` | Running figure scripts, monitoring output, diagnosing failures |
| **Codebase Auditor** | `codebase-auditor/` | Systematic quality audits: encoding consistency, color palettes, exclusions |
| **Pre-Commit Checker** | `pre-commit-checker/` | Fast quality gate on changed files before committing |
| **Systematic Debugging** | `systematic-debugging/` | Any bug, failure, or unexpected behavior — root cause before fixes |
| **Verification Before Completion** | `verification-before-completion/` | Evidence before assertions — run and verify before claiming done |

### How Research Skills Interact

```
User Request (analysis/figure)
     │
     ├──► Research Statistician
     │      → Selects tests (paired t-test, rmANOVA, LMM, permutation)
     │      → Computes stats and effect sizes
     │      → Passes results to Visualizer and Summarizer
     │
     ├──► Research Visualizer
     │      → Proposes 3+ visual designs using project palette/conventions
     │      → Implements chosen design (mm sizing, correct fonts, dual PNG+PDF save)
     │      → Passes figure description to Summarizer
     │
     └──► Research Notes Summarizer
            → Documents methods with all STEREO/photometry parameters
            → Writes results summary with inline statistics
            → Saves notes file alongside figure
```

### Invoking Skills

Skills activate automatically based on context, but can be explicitly requested:
- "Use the Research Visualizer skill to propose figure options for this analysis"
- "Use the Research Statistician skill to select the best tests"
- "Use the Research Notes Summarizer skill to document this analysis"
- "Run Figure 3" or "run the entropy script" (triggers Analysis Runner)
- "Audit the codebase" or "check for issues" (triggers Codebase Auditor)
- "Check before commit" (triggers Pre-Commit Checker)
- Or simply: "Analyze [X] with full visualization, statistics, and documentation"

## Environment

- Windows, Python
- Scripts use `%% ` cell markers (Spyder/VS Code scientific mode)
- Run figure scripts: `python figures/Figure1_June2025.py` (cell-by-cell execution typical)

## Project Structure

```
STEREO/
├── CLAUDE.md                       # This file
├── helper_functions.py             # Core utility: bout detection, photometry processing, smoothing, CDF
├── matplotlibrc.txt                # Matplotlib style config
├── .claude/skills/                 # Skill definitions (8 skills in subfolders)
│   ├── research-visualizer/SKILL.md
│   ├── research-statistician/SKILL.md
│   ├── research-notes-summarizer/SKILL.md
│   ├── analysis-runner/SKILL.md
│   ├── codebase-auditor/SKILL.md
│   ├── pre-commit-checker/SKILL.md
│   ├── systematic-debugging/SKILL.md
│   └── verification-before-completion/SKILL.md
├── figures/                        # Publication figure scripts
│   ├── Figure1_June2025.py         # STEREO validation + saline/splash test behavior
│   ├── Figure3_June2025.py         # Fiber photometry: peaks, BRTs, transient dynamics
│   ├── Figure4_June2025.py         # A2a optogenetics
│   ├── Figure5_June2025.py         # DREADDs (chemogenetics) — sandwich design
│   ├── Figure6_June2025.py         # Drd1 optogenetics
│   └── Figure7_June2025.py         # Photometry during splash test
├── analyses/                       # Standalone analysis scripts
│   ├── stereotypies_development.py # Cocaine escalation across sessions (all cohorts)
│   ├── switch_entropy.py           # Behavioral switch entropy
│   ├── behavioral_entropy.py       # Occupancy-based behavioral entropy
│   ├── behavioral_dynamics.py      # Behavioral dynamics analysis
│   ├── dSPN_opto_analysis.py       # Drd1 optogenetics analysis
│   ├── iSPN_opto_analysis.py       # A2a optogenetics analysis
│   ├── splashTest_behaviorOnly.py  # Splash test behavioral analysis
│   ├── locomotionBasedSeparation.py# Locomotion-based behavioral separation
│   └── photom_PP_validation.py     # Photometry preprocessing validation
├── data_io.py                      # Single source of truth for loading CMT/CTM
├── tools/build_data_cache.py       # Builds the fast uncompressed cache
├── stats_helper_file.py            # rmANOVA, paired tests, CDF permutation, LMM checks
├── data/                           # All data files
│   ├── CMT_Apr26.pkl               # Cohort→Mouse→Trial (current)
│   ├── CTM_Apr26.pkl               # Cohort→Trial→Mouse (current)
│   ├── CMT_Dec24.pkl / CTM_Dec24.pkl  # Previous version, kept for comparison
│   ├── cache/                      # Uncompressed fast-load copies (git-ignored)
│   ├── archive/                    # Aug24 / May24 (old class order, need the LUT)
│   ├── master_table.pkl            # Pre-segmented photometry traces (no generator in repo)
│   ├── opto_alignment_dict.pkl     # Optogenetics timing alignment (11 drd1_opto mice)
│   └── prefBehaviorByMouse_wDepths.csv
├── output/                         # All figure outputs (organized by category)
│   ├── Stereotypies_development/
│   ├── Behavioral_entropy/
│   ├── Switch_entropy/
│   ├── Optogenetics/
│   └── ...
├── notebooks/                      # Jupyter notebooks
│   ├── stereotypies_development.ipynb
│   ├── A2a_opto.ipynb
│   └── FiberDepth_behaviorCorrelation.ipynb
└── archive/                        # Superseded older script versions (ignore)
```

## Data Architecture

### Loading data — always via `data_io`

**Never open the pickles directly.** `data_io.py` at the repo root is the single
source of truth for which dataset version is current:

```python
from data_io import load_CMT, load_CTM
CMT = load_CMT()                 # current version = data_io.DATA_VERSION
CTM = load_CTM(flatten=True)     # entry['merged'] becomes the label array
```

- `DATA_VERSION = 'Apr26'`. Change it in `data_io.py` and the whole repo follows.
- `flatten=True` normalises `entry['merged']` to a 1-D canonical label array
  (handles the nesting, and applies the LUT for archived Aug24/May24 files).
- `relabel_lut(behaviors)` permutes canonical labels onto a script's own
  behaviour-name ordering, derived from the names so a mismatch raises.
- A fast uncompressed cache lives in `data/cache/` (git-ignored). It cuts a load
  from ~175 s to ~5 s. Rebuild with `python tools/build_data_cache.py`; delete it
  freely, `data_io` falls back to the bz2 originals.

### Primary Data Dictionaries

Two BZ2-compressed pickle files in `data/` (~2.8 GB each):

- **CMT_Apr26.pkl**: Cohort → Mouse → Trial hierarchy
- **CTM_Apr26.pkl**: Cohort → Trial → Mouse hierarchy — the same 818 entries, mirrored

8 cohorts, 60 mice, 818 mouse×trial entries, 20,765,012 frames.

Each trial entry contains **five** top-level keys (plus `photom` on 270 of 818):
```python
CMT[cohort][mouse][trial] = {
    'cam1': {                      # and 'cam2', identical shape
        'raw':        np.array,    # int, 0-6, pre-LUT class order
        'final':      np.array,    # int, canonical order; values {0..6,8} — never 7
        'confidence': np.array,    # (N, 7) float32
    },
    'merged': {
        'predictions': {
            'smartMerge':  np.array,   # canonical labels 0-8 — USE THIS
            'simpleMerge': np.array,   # see warning below
        },
        'confidence': {'smartMerge': np.array, 'simpleMerge': np.array},
    },
    'topcam': {
        'velocity':   np.array,    # float64, NaN on 5,181 frames across 24 trials
        'locomotion': np.array,    # binary moving/not-moving flag
    },
    'SSD': {
        'cam1': np.array,          # Sum of squared differences, camera 1
        'cam2': np.array,          # Sum of squared differences, camera 2
    },
    'photom': {                    # (when available)
        'left':  {'df': np.array, 'z': np.array},
        'right': {'df': np.array, 'z': np.array},
        'offset': int              # Alignment offset
    }
}
```

**How the 9 classes are built.** The cameras classify into 8 classes and never
emit 7 — `cam*.final` has no Locomotion. The merge step splits the camera's
at-rest class (8) into 7 (Locomotion) vs 8 (Stationary) using
`topcam.locomotion`, which is a velocity threshold. Class 0 (Jump) is an
override applied on top of any class, and coincides with NaN velocity.

**⚠️ `simpleMerge` is stale in Apr26.** Its 7/8 split still follows the Dec24
locomotion threshold (818/818 trials), while `smartMerge` follows the new one.
Always use `smartMerge`.

### Apr26 vs Dec24

Apr26 differs from Dec24 in exactly one respect: the locomotion velocity
threshold was raised from 1e-5 to ~4.5333e-4, moving **2,108,692 frames from
7 (Locomotion) to 8 (Stationary)** — 100% of all differing frames, no other
transition. Classes 0-6 are identical in every trial; velocity, photometry,
SSD and the per-camera streams are bit-identical.

Consequences:
- The **3-way grouping is provably invariant** (0 of 20.7M frames differ), since
  `grouping_lut` maps both 7 and 8 to `NO_LICKING`. Surface-licking results
  cannot move.
- **9-class analyses do move** — entropy, occupancy vectors, transition
  matrices, state-space PCA. Corpus-wide Locomotion 50.61% → 40.45%,
  Stationary 4.02% → 14.17%.

### Key Constants

```python
FPS = 15                            # Frames per second
second = 15                         # Frames per second (alias)
minute = 60 * second                # 900 frames
mm = 1/25.4                         # mm-to-inches conversion for figure sizing
RNN_offset = 7                      # STEREO classifier temporal offset
ISI = 1 * minute + 20 * second      # 80-second inter-stim interval (opto)
```

### Behavior Encoding

```python
behaviors = ['Jump', 'Undefined', 'Floor licking', 'Wall licking',
             'Grooming', 'Body licking', 'Rearing', 'Locomotion', 'Stationary']
# Indices:    0        1             2                3
#             4          5              6         7             8

# Behavior grouping
PATHO_LICKING = 0    # Floor licking (2) + Wall licking (3)
NATURAL_LICKING = 1  # Grooming (4) + Body licking (5)
NO_LICKING = 2       # Everything else
grouping_lut = np.array([NO_LK, NO_LK, PATHO, PATHO, NATURAL, NATURAL, NO_LK, NO_LK, NO_LK])
```

### Experimental Cohorts

| Cohort | Pathway | Manipulation | Expected CNO Effect |
|--------|---------|-------------|-------------------|
| `drd1_hm4di` | Direct (Drd1) | Inhibitory DREADD | Reduce stereotypies |
| `drd1_hm3dq` | Direct (Drd1) | Excitatory DREADD | Increase stereotypies |
| `a2a_hm4di` | Indirect (A2a) | Inhibitory DREADD | Increase stereotypies |
| `a2a_hm3dq` | Indirect (A2a) | Excitatory DREADD | Reduce stereotypies |
| `controls` | Both | No DREADD | No change (negative control) |
| `a2a_opto` | Indirect (A2a) | ChR2 optogenetics | Acute behavior change |
| `drd1_opto` | Direct (Drd1) | ChR2 optogenetics | Acute behavior change |

### Statistics helper

`stats_helper_file.py` (imported as `shf`) now lives at the repo root, taken from
the manuscript's companion repo `shalomitay/opponent-vls-circuits-code`. It
provides `rmANOVA()`, `paired_ttest()`, `paired_wilcoxon()`, `pairwise_ttest()`,
`pairwise_wilcoxon()`, `paired_cdf_permutation_test()`, `permutation_paired_t_test()`,
`rmFriedman()`, `conclude_LMM()`, `validate_lmm()`, `holm_correction()`.

Note the current signatures take more arguments than some older call sites pass —
`rmANOVA(data, conditions)` and `validate_lmm(result, pathway, parameter)`.

`helper_functions.py` came from the same source and is a strict superset of the
old local copy. Two behaviour changes to be aware of: `remove_IQR_outliers()`
returns **inlier indices** (not a NaN-masked array), and `smoothing()` takes an
optional `window` argument and is NaN-aware.

### Upstream manuscript code

`shalomitay/opponent-vls-circuits-code` holds the publication versions of the
figure scripts (`Figure1–6`, `FigureS1–S10`). They are already Apr26-native,
expect data at `Apr26/Data/` and write to `Apr26/Figures/July26/`. They share
almost no code with `figures/*_June2025.py`, which are the exploratory originals.

## Visual Style Quick Reference

- **Behavior colors**: `["#696969", "#d3d3d3", "#d73027", "#e57373", "#c4a7e7", "#8e63b8", "#b3e5fc", "#3399cc", "#1f4e79"]`
- **Grouped colors**: Surface licking `#d73027`, Self-licking `#c4a7e7`, No licking `#1f4e79`
- **dSPN**: `#d73027`, **iSPN**: `#721515`
- **Cocaine gradient**: `['#808080', '#FFA500', '#FF8C00', '#FF6347', '#E60000', '#990000']`
- **Laser stim**: `#B2E2F6`
- **Axis labels**: 10pt, **Tick labels**: 8pt, **Legends**: 8pt, frameon=False
- **Spines**: top/right removed, left/bottom kept
- **Error**: SEM (not SD) throughout
- **Save**: always both PNG + PDF, dpi=300, bbox_inches='tight'
- **Figure sizes**: in mm (using `mm = 1/25.4` conversion)
- **Confusion matrices**: `Purples` colormap, `square=True`

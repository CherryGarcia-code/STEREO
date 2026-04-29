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
├── data/                           # All data files
│   ├── CMT_Aug24.pkl               # Cohort→Mouse→Trial (Aug 2024)
│   ├── CTM_Aug24.pkl               # Cohort→Trial→Mouse (Aug 2024)
│   ├── CMT_May24.pkl               # Cohort→Mouse→Trial (May 2024)
│   ├── CTM_May24.pkl               # Cohort→Trial→Mouse (May 2024)
│   ├── opto_alignment_dict.pkl     # Optogenetics timing alignment
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

### Primary Data Dictionaries

Two BZ2-compressed pickle files in `data/`:

- **CMT_Dec24.pkl**: Cohort → Mouse → Trial hierarchy
- **CTM_Dec24.pkl**: Cohort → Trial → Mouse hierarchy

Each trial entry contains:
```python
CMT[cohort][mouse][trial] = {
    'merged': {
        'predictions': {
            'smartMerge': np.array  # Frame-by-frame behavior labels (0-8)
        }
    },
    'topcam': {
        'velocity': np.array       # Per-frame velocity (raw, multiply by FPS*10 for cm/s)
    },
    'SSD': {
        'cam1': np.array,          # Sum of squared differences, camera 1
        'cam2': np.array           # Sum of squared differences, camera 2
    },
    'photom': {                    # (when available)
        'left': {'df': np.array, 'z': np.array},
        'right': {'df': np.array, 'z': np.array},
        'offset': int              # Alignment offset
    }
}
```

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

### External Dependencies (Not in Repo)

- `stats_helper_file` (imported as `shf`): Contains `rmANOVA()`, `paired_ttest()`, `paired_cdf_permutation_test()`, `validate_lmm()`
- Raw data pickles in `data/`

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

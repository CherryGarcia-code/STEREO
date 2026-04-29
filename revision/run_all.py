"""
Run all revision figures.

Usage:
    py revision/run_all.py
"""
import subprocess, sys, os, time

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

scripts = [
    # Supplementary figures
    'revision/SuppFig_STEREO_Validation.py',
    'revision/SuppFig_AllBehaviors_Development.py',
    'revision/SuppFig_iSPN_Opto_Transitions.py',
    'revision/SuppFig_dSPN_Opto_BoutDur_TransProb.py',
    'revision/SuppFig_DREADDs_AllCohorts.py',
    'revision/SuppFig_Stacked_Development.py',
    'revision/SuppFig_iSPN_Transient.py',
    'revision/SuppFig_Photom_BoutCorrelation.py',
    'revision/SuppFig_Entropy_Manipulations.py',
    'revision/SuppFig_Grooming_Photometry.py',
    'revision/SuppFig_Context_Specificity.py',
    'revision/SuppFig_Velocity_Licking.py',
    'revision/SuppFig_dSPN_vs_iSPN_BoutDur.py',
    'revision/SuppFig_TimeWarped_SplashTest.py',
    # Impact figures
    'revision/Impact_CompositeSummary.py',
    'revision/Impact_DoseResponse.py',
    'revision/Impact_ForestPlot.py',
    'revision/Impact_BoutDecomposition.py',
    'revision/Impact_IndividualTrajectories.py',
    'revision/Impact_TransitionFlow.py',
]

passed, failed = [], []
t0 = time.time()

for script in scripts:
    name = os.path.basename(script).replace('.py', '')
    print(f'\n{"="*60}')
    print(f'Running: {name}')
    print(f'{"="*60}')
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)
    if result.returncode == 0:
        passed.append(name)
        print(f'  OK ({name})')
    else:
        failed.append(name)
        print(f'  FAILED ({name})')
        print(result.stderr[-500:] if result.stderr else 'No stderr')

elapsed = time.time() - t0
print(f'\n{"="*60}')
print(f'Done in {elapsed:.0f}s: {len(passed)} passed, {len(failed)} failed')
if failed:
    print(f'Failed: {", ".join(failed)}')
print(f'{"="*60}')

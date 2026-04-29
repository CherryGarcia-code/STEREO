"""
Run all integrative analyses.

Usage:
    py integrative/run_all.py
"""
import subprocess, sys, os, time

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

scripts = [
    'integrative/Int1_BehavioralStateSpace.py',
    'integrative/Int2_EarlyPredictorModel.py',
    'integrative/Int3_EntropyLickingRelationship.py',
    'integrative/Int4_CircuitImbalanceModel.py',
    'integrative/Int5_RegimeClustering.py',
]

passed, failed = [], []
t0 = time.time()

for script in scripts:
    name = os.path.basename(script).replace('.py', '')
    print(f'\n{"="*60}')
    print(f'Running: {name}')
    print(f'{"="*60}')
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)
    print(result.stdout[-800:] if result.stdout else '')
    if result.returncode == 0:
        passed.append(name)
        print(f'  OK ({name})')
    else:
        failed.append(name)
        print(f'  FAILED ({name})')
        print(result.stderr[-600:] if result.stderr else 'No stderr')

elapsed = time.time() - t0
print(f'\n{"="*60}')
print(f'Done in {elapsed:.0f}s: {len(passed)} passed, {len(failed)} failed')
if failed:
    print(f'Failed: {", ".join(failed)}')
print(f'{"="*60}')

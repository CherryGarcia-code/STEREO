"""
Transient-triggered behaviour: flip the analysis around.

Instead of asking "what does the signal do during behaviour X", take the
calcium transients as the events and ask what the animal was doing before and
after them. This sidesteps behavioural state labelling entirely, and it is the
natural way in to a signal that is heavy-tailed rather than graded -- only ~1%
of transients are large (>3x the median prominence), and the tail is much
heavier in dSPN (max prominence 39.3) than iSPN (16.1).

Two questions:
  1  peri-transient behaviour occupancy -- which behaviours surround a transient
  2  behavioural switch rate around a transient -- do transients mark transitions

Null model: event times circularly shifted within the session. That preserves
the number of events and the autocorrelation of the behaviour, so it tests
time-locking specifically rather than overall behaviour composition.

    python analyses/photometry_transient_triggered.py --selftest

Outputs -> output/Photometry/TransientTriggered/
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'revision'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.signal
from scipy import stats
from collections import defaultdict

import data_io
from revision_utils import mm, photom_exclusions, get_pathway, setup_style, save_fig

FPS = 15
WIN = 5 * FPS                    # +/- 5 s peri-event window
LAGS = np.arange(-WIN, WIN + 1)
LAG_S = LAGS / FPS

PROM, WIDTH = 2.0, 5             # repo convention (Figure3, SuppFig_iSPN_Transient)
LARGE_PCTL = 90                  # "large" = top decile of that session's transients
N_SHUFFLE = 200                  # circular-shift null replicates

BEHAVIORS = data_io.CANONICAL_BEHAVIORS
COLORS = data_io.CANONICAL_COLORS
COCAINE = ['cocaine1', 'cocaine2', 'cocaine3', 'cocaine4', 'cocaine5']
SALINE = ['saline1', 'saline2', 'saline3']      # naive baseline, before any cocaine
SALINE_ONLY = ['salineOnly']                    # saline AFTER a cocaine history
# salineOnly is kept separate on purpose: it is a saline session given to an
# animal that has already been sensitised, so it probes what persists without
# the drug on board. Pooling it with the naive baseline would blur exactly the
# comparison that makes it interesting.
CONDITIONS = {'cocaine': COCAINE, 'saline': SALINE, 'salineOnly': SALINE_ONLY}
BASELINE_COND = 'saline'

OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', 'output', 'Photometry', 'TransientTriggered'))


# --------------------------------------------------------------------------
def peri_event(series, idx, win=WIN, n_values=None):
    """Stack of peri-event slices; rows are events, columns lags."""
    keep = [i for i in idx if i - win >= 0 and i + win + 1 <= len(series)]
    if not keep:
        return np.empty((0, 2 * win + 1))
    return np.stack([series[i - win: i + win + 1] for i in keep])


def occupancy_curves(preds, idx, win=WIN):
    """P(behaviour b) at each lag, given events at idx. Shape (9, n_lags)."""
    seg = peri_event(preds, idx, win)
    if seg.shape[0] == 0:
        return None
    return np.stack([(seg == b).mean(axis=0) for b in range(len(BEHAVIORS))])


def switch_curve(preds, idx, win=WIN):
    """P(behaviour changes) at each lag."""
    sw = np.concatenate(([0], (preds[1:] != preds[:-1]).astype(float)))
    seg = peri_event(sw, idx, win)
    return seg.mean(axis=0) if seg.shape[0] else None


def circular_null(preds, idx, n_shuffle=N_SHUFFLE, win=WIN, rng=None, kind='switch'):
    """Null curves from circularly shifted event times."""
    rng = rng or np.random.default_rng(0)
    n = len(preds)
    out = []
    for _ in range(n_shuffle):
        shift = int(rng.integers(win + 1, n - win - 1))
        sidx = (np.asarray(idx) + shift) % n
        c = (switch_curve(preds, sidx, win) if kind == 'switch'
             else occupancy_curves(preds, sidx, win))
        if c is not None:
            out.append(c)
    return np.stack(out) if out else None


# --------------------------------------------------------------------------
def selftest(verbose=True):
    """Plant transients exactly at behavioural switches; the switch curve must
    peak at lag 0 and the circular-shift null must be flat."""
    rng = np.random.default_rng(0)
    ok = True

    def say(*a):
        if verbose:
            print(*a)

    n = 30 * 60 * FPS
    # behaviour that changes every ~3 s
    preds = np.zeros(n, int)
    t, cur = 0, 0
    while t < n:
        dur = int(rng.normal(3 * FPS, FPS))
        dur = max(dur, 5)
        preds[t:t + dur] = cur
        cur = int(rng.integers(0, 9))
        t += dur
    switches = np.where(np.concatenate(([False], preds[1:] != preds[:-1])))[0]

    say('=' * 70)
    say('SELF-TEST 1: transients planted AT switches -> switch curve peaks at 0')
    say('=' * 70)
    ev = switches[(switches > WIN) & (switches < n - WIN)][::3]
    curve = switch_curve(preds, ev)
    peak_lag = LAG_S[np.argmax(curve)]
    good = abs(peak_lag) < 0.1 and curve.max() > 5 * np.median(curve)
    ok &= good
    say(f'  peak at {peak_lag:+.2f}s, peak/median ratio = '
        f'{curve.max()/np.median(curve):.1f}  {"PASS" if good else "FAIL"}')

    say('\n' + '=' * 70)
    say('SELF-TEST 2: circular-shift null is flat and unbiased')
    say('=' * 70)
    null = circular_null(preds, ev, n_shuffle=100, rng=rng)
    nm = null.mean(axis=0)
    flat = (nm.max() - nm.min()) < 0.25 * (curve.max() - curve.min())
    ok &= flat
    say(f'  null range = {nm.max()-nm.min():.4f} vs observed range '
        f'{curve.max()-curve.min():.4f}  {"PASS" if flat else "FAIL"}')
    say(f'  null mean = {nm.mean():.4f}, base switch rate = '
        f'{np.mean(np.concatenate(([0],(preds[1:]!=preds[:-1]).astype(float)))):.4f}')

    say('\n' + '=' * 70)
    say('SELF-TEST 3: unrelated events give no time-locking')
    say('=' * 70)
    rand_ev = rng.integers(WIN + 1, n - WIN - 1, len(ev))
    rc = switch_curve(preds, rand_ev)
    nolock = (rc.max() - rc.min()) < 0.25 * (curve.max() - curve.min())
    ok &= nolock
    say(f'  random-event range = {rc.max()-rc.min():.4f}  '
        f'{"PASS" if nolock else "FAIL"}')

    say('\n' + '=' * 70)
    say('SELF-TEST 4: occupancy curves are proper probabilities')
    say('=' * 70)
    occ = occupancy_curves(preds, ev)
    sums = occ.sum(axis=0)
    good = np.allclose(sums, 1.0)
    ok &= good
    say(f'  occupancy sums to 1 at every lag: {"PASS" if good else "FAIL"} '
        f'(min {sums.min():.6f}, max {sums.max():.6f})')

    say('\n' + ('ALL SELF-TESTS PASSED' if ok else '*** SELF-TEST FAILURE ***'))
    return ok


# --------------------------------------------------------------------------
def analyse(CMT, trials, rng):
    """Run the whole peri-transient analysis for one drug condition."""
    rm_mouse = {m for _, m in photom_exclusions['remove_mouse']}
    rm_hem = {(m, h) for _, m, h in photom_exclusions['remove_hemisphere']}
    rm_trial = {(m, t) for _, m, t in photom_exclusions['remove_trial']}

    sw = defaultdict(lambda: defaultdict(list))     # [pathway][size] -> per-mouse curves
    sw_null = defaultdict(lambda: defaultdict(list))
    occ = defaultdict(lambda: defaultdict(list))
    occ_null = defaultdict(lambda: defaultdict(list))
    counts = defaultdict(lambda: defaultdict(int))
    proms = defaultdict(list)
    mouse_ids = defaultdict(lambda: defaultdict(list))
    COCAINE = trials     # local alias so the loop below is unchanged

    for c in CMT:
        for m in CMT[c]:
            if m in rm_mouse:
                continue
            pathway = 'dSPN' if get_pathway(c, m) == 'drd1' else 'iSPN'
            per_mouse_sw = defaultdict(list)
            per_mouse_null = defaultdict(list)
            per_mouse_occ = defaultdict(list)
            per_mouse_occ_null = defaultdict(list)
            for t in COCAINE:
                if t not in CMT[c][m] or (m, t) in rm_trial:
                    continue
                e = CMT[c][m][t]
                if 'photom' not in e:
                    continue
                hems = [h for h in ('left', 'right')
                        if isinstance(e['photom'].get(h), dict) and (m, h) not in rm_hem]
                if not hems:
                    continue
                z = np.nanmean(np.vstack([e['photom'][h]['z'] for h in hems]), axis=0)
                preds = e['merged'].astype(int)
                pk, props = scipy.signal.find_peaks(z, prominence=PROM, width=WIDTH)
                if len(pk) < 20:
                    continue
                pr = props['prominences']
                proms[pathway].extend(pr)
                thr = np.percentile(pr, LARGE_PCTL)
                for size, sel in (('large', pk[pr >= thr]), ('typical', pk[pr < thr])):
                    if len(sel) < 5:
                        continue
                    cs = switch_curve(preds, sel)
                    if cs is None:
                        continue
                    per_mouse_sw[size].append(cs)
                    nl = circular_null(preds, sel, rng=rng)
                    if nl is not None:
                        per_mouse_null[size].append(nl.mean(axis=0))
                    oc = occupancy_curves(preds, sel)
                    if oc is not None:
                        per_mouse_occ[size].append(oc)
                        # occupancy needs its own null: a dip at lag 0 could
                        # otherwise just reflect overall behaviour composition,
                        # or transients being easier to detect when the animal
                        # is still. Fewer shuffles here -- 9 behaviours each.
                        onl = circular_null(preds, sel, n_shuffle=50, rng=rng,
                                            kind='occupancy')
                        if onl is not None:
                            per_mouse_occ_null[size].append(onl.mean(axis=0))
                    counts[pathway][size] += len(sel)
            for size in ('large', 'typical'):
                if per_mouse_sw[size]:
                    sw[pathway][size].append(np.mean(per_mouse_sw[size], axis=0))
                    sw_null[pathway][size].append(np.mean(per_mouse_null[size], axis=0))
                    occ[pathway][size].append(np.mean(per_mouse_occ[size], axis=0))
                    if per_mouse_occ_null[size]:
                        occ_null[pathway][size].append(
                            np.mean(per_mouse_occ_null[size], axis=0))
                    mouse_ids[pathway][size].append(m)
    return dict(sw=sw, sw_null=sw_null, occ=occ, occ_null=occ_null,
                counts=counts, proms=proms, mice=mouse_ids)


def main():
    setup_style()
    os.makedirs(OUT, exist_ok=True)
    print('Self-test before touching real data...')
    if not selftest(verbose=False):
        print('SELF-TEST FAILED -- aborting'); sys.exit(1)
    print('  passed.\n')

    CMT = data_io.load_CMT(flatten=True)
    rng = np.random.default_rng(0)
    R = {}
    for cond, trials in CONDITIONS.items():
        print(f'  analysing {cond} sessions...')
        R[cond] = analyse(CMT, trials, rng)
    del CMT


    sink = open(os.path.join(OUT, 'statistics.txt'), 'w')
    sink.write('Transient-triggered behaviour\n' + '=' * 70 + '\n')
    sink.write(f'peaks: prominence>={PROM}, width>={WIDTH} frames (repo convention)\n')
    sink.write(f'"large" = top {100-LARGE_PCTL}% of each session\'s transients by prominence\n')
    sink.write(f'window +/-{WIN/FPS:.0f}s; null = {N_SHUFFLE} circular shifts of event times\n')
    sink.write('Unit of analysis: sessions -> mouse curves -> across mice.\n')
    sink.write('Saline sessions are the within-subject control: if an effect is a\n')
    sink.write('general property of striatal transients it should appear there too.\n')

    def eff(res, pathway, size, b_index):
        """Per-mouse (observed - null) occupancy at lag 0, keyed by mouse."""
        if len(res['occ'][pathway][size]) < 2:
            return {}
        O = np.stack(res['occ'][pathway][size])
        Nn = np.stack(res['occ_null'][pathway][size])
        mice = res['mice'][pathway][size]
        return {m: O[i, b_index, WIN] - Nn[i, b_index, WIN] for i, m in enumerate(mice)}

    for cond in CONDITIONS:
        res = R[cond]
        sink.write('\n' + '=' * 70 + f'\n{cond.upper()} SESSIONS\n' + '=' * 70 + '\n')
        for pathway in ('dSPN', 'iSPN'):
            p = np.array(res['proms'][pathway])
            if p.size == 0:
                continue
            sink.write(f'{pathway}: {len(p):,} transients, median prominence '
                       f'{np.median(p):.2f}, p90 {np.percentile(p,90):.2f}, max {p.max():.1f}\n')
            for size in ('large', 'typical'):
                sink.write(f'    {size:<8}{res["counts"][pathway][size]:>8,} events, '
                           f'{len(res["sw"][pathway][size])} mice\n')

        sink.write('\n  Switch rate at lag 0 vs null\n')
        sink.write(f'    {"pathway":<8}{"size":<9}{"obs":>9}{"null":>9}{"diff":>9}'
                   f'{"t":>8}{"p":>9}{"N":>4}\n')
        for pathway in ('dSPN', 'iSPN'):
            for size in ('large', 'typical'):
                if len(res['sw'][pathway][size]) < 3:
                    continue
                obs = np.vstack(res['sw'][pathway][size])[:, WIN]
                nul = np.vstack(res['sw_null'][pathway][size])[:, WIN]
                t_, p_ = stats.ttest_rel(obs, nul)
                sink.write(f'    {pathway:<8}{size:<9}{obs.mean():>9.4f}{nul.mean():>9.4f}'
                           f'{obs.mean()-nul.mean():>9.4f}{t_:>8.2f}{p_:>9.4f}{len(obs):>4}\n')

        sink.write('\n  Behaviour occupancy at lag 0 vs null\n')
        sink.write(f'    {"pathway":<8}{"size":<9}{"behaviour":<15}{"obs":>9}{"null":>9}'
                   f'{"diff":>9}{"t":>8}{"p":>9}{"N":>4}\n')
        for pathway in ('dSPN', 'iSPN'):
            for size in ('large', 'typical'):
                if len(res['occ'][pathway][size]) < 3:
                    continue
                O = np.stack(res['occ'][pathway][size])
                Nn = np.stack(res['occ_null'][pathway][size])
                for b in range(len(BEHAVIORS)):
                    a, bn = O[:, b, WIN], Nn[:, b, WIN]
                    if max(a.mean(), bn.mean()) < 0.03:
                        continue
                    t_, p_ = stats.ttest_rel(a, bn)
                    sink.write(f'    {pathway:<8}{size:<9}{BEHAVIORS[b]:<15}{a.mean():>9.4f}'
                               f'{bn.mean():>9.4f}{a.mean()-bn.mean():>9.4f}{t_:>8.2f}'
                               f'{p_:>9.4f}{len(a):>4}'
                               f'{"  *" if p_ < 0.05 else ""}\n')

    # ------------- each condition vs naive saline, paired within mouse -------------
    for other in [c for c in CONDITIONS if c != BASELINE_COND]:
        sink.write('\n' + '=' * 70 + f'\n{other.upper()} vs {BASELINE_COND.upper()} '
                   f'(paired within mouse)\n' + '=' * 70 + '\n')
        sink.write('Effect = (observed - null) occupancy at lag 0. A difference here\n')
        sink.write('means the coupling changed, not merely the behaviour composition\n')
        sink.write('(which the circular-shift null already absorbs).\n')
        sink.write(f'  {"pathway":<8}{"size":<9}{"behaviour":<15}{other:>11}'
                   f'{BASELINE_COND:>10}{"diff":>10}{"t":>8}{"p":>9}{"N":>4}\n')
        for pathway in ('dSPN', 'iSPN'):
            for size in ('large', 'typical'):
                for b in range(len(BEHAVIORS)):
                    ec = eff(R[other], pathway, size, b)
                    es = eff(R[BASELINE_COND], pathway, size, b)
                    shared = sorted(set(ec) & set(es))
                    if len(shared) < 4:
                        continue
                    a = np.array([ec[m] for m in shared])
                    s = np.array([es[m] for m in shared])
                    if max(abs(a.mean()), abs(s.mean())) < 0.02:
                        continue
                    t_, p_ = stats.ttest_rel(a, s)
                    sink.write(f'  {pathway:<8}{size:<9}{BEHAVIORS[b]:<15}{a.mean():>11.4f}'
                               f'{s.mean():>10.4f}{a.mean()-s.mean():>10.4f}{t_:>8.2f}'
                               f'{p_:>9.4f}{len(shared):>4}'
                               f'{"  *" if p_ < 0.05 else ""}\n')

    # ---------------- figures, one per condition ----------------
    for cond in CONDITIONS:
        res = R[cond]
        fig = plt.figure(figsize=(210 * mm, 130 * mm))
        gs = gridspec.GridSpec(2, 3, figure=fig, left=0.08, right=0.98,
                               top=0.88, bottom=0.14, wspace=0.35, hspace=0.55)
        for row, pathway in enumerate(('dSPN', 'iSPN')):
            ax = fig.add_subplot(gs[row, 0])
            for size, colr in (('large', '#d73027'), ('typical', '#999999')):
                if len(res['sw'][pathway][size]) < 2:
                    continue
                obs = np.vstack(res['sw'][pathway][size])
                ax.plot(LAG_S, obs.mean(0), color=colr, lw=1.2, label=size)
                ax.fill_between(LAG_S, obs.mean(0) - stats.sem(obs, axis=0),
                                obs.mean(0) + stats.sem(obs, axis=0),
                                color=colr, alpha=0.25, lw=0)
                nul = np.vstack(res['sw_null'][pathway][size])
                ax.plot(LAG_S, nul.mean(0), color=colr, lw=0.7, ls=':')
            ax.axvline(0, color='k', ls='--', lw=0.6)
            ax.set_title(f'{pathway} Â· switch rate', fontsize=7, fontweight='bold')
            ax.set_xlabel('Lag from transient (s)', fontsize=6)
            ax.set_ylabel('P(switch)', fontsize=6)
            if row == 0:
                ax.legend(fontsize=5.5, frameon=False, title='dotted = null',
                          title_fontsize=5)

            for col, size in ((1, 'large'), (2, 'typical')):
                ax = fig.add_subplot(gs[row, col])
                if len(res['occ'][pathway][size]) < 2:
                    continue
                mean = np.stack(res['occ'][pathway][size]).mean(0)
                nullm = np.stack(res['occ_null'][pathway][size]).mean(0)
                for b in range(9):
                    if mean[b].max() < 0.05:
                        continue
                    ax.plot(LAG_S, mean[b], color=COLORS[b], lw=1.0, label=BEHAVIORS[b])
                    ax.plot(LAG_S, nullm[b], color=COLORS[b], lw=0.6, ls=':', alpha=0.7)
                ax.axvline(0, color='k', ls='--', lw=0.6)
                ax.set_title(f'{pathway} Â· {size} transients', fontsize=7, fontweight='bold')
                ax.set_xlabel('Lag from transient (s)', fontsize=6)
                ax.set_ylabel('P(behaviour)', fontsize=6)
                if row == 0 and col == 2:
                    ax.legend(fontsize=4.5, frameon=False, ncol=2)

        for ax in fig.axes:
            ax.tick_params(labelsize=6)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        fig.suptitle(f'Behaviour surrounding calcium transients â€” {cond} sessions '
                     f'(dotted = circular-shift null)', fontsize=9, fontweight='bold')
        save_fig(fig, OUT, f'transient_triggered_{cond}')
        print(f'  figure -> transient_triggered_{cond}')

    sink.close()
    print(f'\nDone. Outputs in {OUT}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    main()


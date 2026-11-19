"""
Encoding model for striatal photometry.

Instead of carving behaviour into discrete states -- which failed because the
behaviour interleaves on a seconds timescale -- fit a linear FIR ("kernel")
model of the photometry signal against behavioural events and movement:

    z(t) = sum_p sum_tau  beta_p(tau) * x_p(t - tau)  +  eps

Predictors are behaviour ONSET impulses (not occupancy boxcars, which sum to 1
at every frame and are therefore collinear by construction), plus velocity and
SSD as continuous covariates. beta_p(tau) is then directly readable as the
response to event p at lag tau, with the other predictors' contributions
regressed out -- so movement is modelled rather than left as a confound.

Run the self-test before trusting anything:
    python analyses/photometry_encoding_model.py --selftest

It plants known kernels in synthetic data and checks they are recovered, and
checks that a circularly shifted design predicts nothing.

Outputs -> output/Photometry/EncodingModel/
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'revision'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.linear_model import Ridge
from collections import defaultdict

import data_io
from revision_utils import mm, photom_exclusions, get_pathway, setup_style, save_fig

FPS = 15
LAG_MIN = -2 * FPS          # 2 s before the event (anticipatory)
LAG_MAX = 6 * FPS           # 6 s after
LAG_STEP = 3                # 0.2 s kernel resolution; GCaMP kinetics are ~1 s
LAGS = np.arange(LAG_MIN, LAG_MAX + 1, LAG_STEP)
LAG_S = LAGS / FPS

BEHAVIORS = data_io.CANONICAL_BEHAVIORS
COVARIATES = ['velocity', 'SSD']
PREDICTORS = BEHAVIORS + COVARIATES

ALPHAS = np.logspace(1, 5, 9)

COCAINE = ['cocaine1', 'cocaine2', 'cocaine3', 'cocaine4', 'cocaine5']
SALINE = ['saline1', 'saline2', 'saline3']      # naive baseline, before any cocaine
SALINE_ONLY = ['salineOnly']                    # saline AFTER a cocaine history
CONDITIONS = {'cocaine': COCAINE, 'saline': SALINE, 'salineOnly': SALINE_ONLY}
BASELINE_COND = 'saline'

# Events are BOUT onsets, not every label change. Counting every change diluted
# the bout-locked transient 3-10x: dSPN floor licking has 6,030 label changes
# but only 344 real bouts, so 94% of "events" were sub-second fragments.
# These criteria match the repo's own bout analyses (segment_bouts_per_history).
MIN_BOUT = 1 * FPS          # bout must last at least 1 s
MIN_PRIOR = 2 * FPS         # and follow >= 2 s of NOT doing that behaviour

OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', 'output', 'Photometry', 'EncodingModel'))


# --------------------------------------------------------------------------
# Design matrix
# --------------------------------------------------------------------------
def shift(x, k):
    """x delayed by k samples: out[t] = x[t-k], zero-padded. k>0 shifts later."""
    out = np.zeros_like(x, dtype=np.float32)
    if k == 0:
        out[:] = x
    elif k > 0:
        out[k:] = x[:-k]
    else:
        out[:k] = x[-k:]
    return out


def build_design(predictors, n_samples, lags=LAGS):
    """(n_samples, n_predictors*n_lags) float32 design matrix.

    Column block p, lag index j holds x_p delayed by lags[j], so the fitted
    coefficient is the response at `lags[j]` samples AFTER the event.
    """
    cols = []
    for x in predictors:
        x = np.asarray(x, dtype=np.float32)
        for k in lags:
            cols.append(shift(x, int(k)))
    return np.stack(cols, axis=1)


def unpack(coef, n_pred, n_lags=None):
    n_lags = n_lags or len(LAGS)
    return coef.reshape(n_pred, n_lags)


def fit_kernels(predictors, y, lags=LAGS, alphas=ALPHAS, return_model=False):
    """Ridge fit; alpha picked by generalised cross-validation."""
    X = build_design(predictors, len(y), lags)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(y)
    from sklearn.linear_model import RidgeCV
    model = RidgeCV(alphas=alphas)
    model.fit(X[keep], y[keep])
    K = unpack(model.coef_, len(predictors), len(lags))
    if return_model:
        return K, model, X, keep
    return K


def r2_score_manual(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


# --------------------------------------------------------------------------
# Self-test: can the fitter recover kernels we planted?
# --------------------------------------------------------------------------
def selftest(seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    n = 60 * 60 * FPS                       # 60 min
    ok = True

    def say(*a):
        if verbose:
            print(*a)

    say('=' * 70)
    say('SELF-TEST 1: recover planted kernels from synthetic data')
    say('=' * 70)

    # two event predictors + one continuous covariate
    ev_a = np.zeros(n, np.float32); ev_a[rng.choice(n, 900, replace=False)] = 1
    ev_b = np.zeros(n, np.float32); ev_b[rng.choice(n, 900, replace=False)] = 1
    cov = rng.standard_normal(n).astype(np.float32)
    cov = np.convolve(cov, np.ones(15) / 15, mode='same')      # smooth, like velocity

    t = LAG_S
    k_a = 1.2 * np.exp(-((t - 0.8) ** 2) / (2 * 0.5 ** 2))      # positive bump
    k_b = -0.8 * np.exp(-((t - 1.5) ** 2) / (2 * 0.7 ** 2))     # negative bump
    k_c = 0.5 * np.exp(-((t - 0.0) ** 2) / (2 * 0.3 ** 2))      # covariate kernel
    truth = np.stack([k_a, k_b, k_c])

    X = build_design([ev_a, ev_b, cov], n)
    y = X @ truth.ravel() + rng.standard_normal(n) * 0.5

    K = fit_kernels([ev_a, ev_b, cov], y)
    for i, name in enumerate(['event A', 'event B', 'covariate']):
        r = np.corrcoef(K[i], truth[i])[0, 1]
        rmse = float(np.sqrt(np.mean((K[i] - truth[i]) ** 2)))
        peak_err = abs(LAG_S[np.argmax(np.abs(K[i]))] - LAG_S[np.argmax(np.abs(truth[i]))])
        good = r > 0.95 and peak_err <= 0.4
        ok &= good
        say(f'  {name:<11} shape r={r:.4f}  rmse={rmse:.4f}  '
            f'peak-lag error={peak_err:.2f}s  {"PASS" if good else "FAIL"}')

    say('\n' + '=' * 70)
    say('SELF-TEST 2: sign and timing are not flipped')
    say('=' * 70)
    say(f'  planted A peak at {LAG_S[np.argmax(truth[0])]:+.2f}s, '
        f'recovered {LAG_S[np.argmax(K[0])]:+.2f}s')
    say(f'  planted B trough at {LAG_S[np.argmin(truth[1])]:+.2f}s, '
        f'recovered {LAG_S[np.argmin(K[1])]:+.2f}s')
    sign_ok = K[0].max() > 0 and K[1].min() < 0
    ok &= sign_ok
    say(f'  A positive and B negative: {"PASS" if sign_ok else "FAIL"}')

    say('\n' + '=' * 70)
    say('SELF-TEST 3: circularly shifted design predicts nothing')
    say('=' * 70)
    split = n // 2
    Xtr = build_design([ev_a, ev_b, cov], n)
    m = Ridge(alpha=1e3).fit(Xtr[:split], y[:split])
    r2_true = r2_score_manual(y[split:], m.predict(Xtr[split:]))

    sh = n // 3
    Xsh = build_design([np.roll(ev_a, sh), np.roll(ev_b, sh), np.roll(cov, sh)], n)
    m2 = Ridge(alpha=1e3).fit(Xsh[:split], y[:split])
    r2_null = r2_score_manual(y[split:], m2.predict(Xsh[split:]))
    null_ok = r2_true > 0.3 and abs(r2_null) < 0.05
    ok &= null_ok
    say(f'  held-out R2, true alignment  = {r2_true:+.4f}')
    say(f'  held-out R2, shifted design  = {r2_null:+.4f}')
    say(f'  {"PASS" if null_ok else "FAIL"}')

    say('\n' + '=' * 70)
    say('SELF-TEST 4: collinearity check on real predictor structure')
    say('=' * 70)
    # Occupancy boxcars sum to 1 at every frame, so [boxcars | intercept] is
    # rank deficient. The boxcars alone are full rank -- the collinearity only
    # shows up once the intercept is included, which is why we must test that.
    occ = np.zeros((n, 3), np.float32)
    lab = rng.integers(0, 3, n)
    for i in range(3):
        occ[:, i] = (lab == i)
    occ_int = np.column_stack([occ, np.ones(n, np.float32)])
    ons_int = np.column_stack([ev_a, ev_b, cov, np.ones(n, np.float32)])
    r_occ = np.linalg.matrix_rank(occ_int)
    r_ons = np.linalg.matrix_rank(ons_int)
    say(f'  [occupancy boxcars | intercept]: rank {r_occ} of 4 columns '
        f'-> {"RANK DEFICIENT as expected" if r_occ == 3 else "unexpected"}')
    say(f'  [onset impulses    | intercept]: rank {r_ons} of 4 columns '
        f'-> {"full rank" if r_ons == 4 else "RANK DEFICIENT"}')
    coll_ok = (r_occ == 3) and (r_ons == 4)
    say(f'  -> onsets used, and they are safe with an intercept. '
        f'{"PASS" if coll_ok else "FAIL"}')
    ok &= coll_ok

    say('\n' + ('ALL SELF-TESTS PASSED' if ok else '*** SELF-TEST FAILURE ***'))
    return ok


# --------------------------------------------------------------------------
# Real data
# --------------------------------------------------------------------------
def zscore_nan(x):
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x)
    mu, sd = np.nanmean(x[finite]), np.nanstd(x[finite])
    out = (x - mu) / (sd if sd > 0 else 1.0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def bout_onsets(preds, b, min_bout=MIN_BOUT, min_prior=MIN_PRIOR):
    """Start frames of genuine bouts of behaviour b."""
    m = (preds == b).astype(np.int8)
    d = np.diff(np.concatenate(([0], m, [0])))
    out = []
    for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        if e - s < min_bout or s < min_prior:
            continue
        if m[s - min_prior:s].any():
            continue
        out.append(s)
    return np.array(out, int)


def session_predictors(entry):
    """9 behaviour BOUT-onset impulse trains + z-scored velocity and SSD."""
    preds = entry['merged'].astype(int)
    n = preds.size
    onsets = []
    for b in range(len(BEHAVIORS)):
        x = np.zeros(n, np.float32)
        idx = bout_onsets(preds, b)
        if len(idx):
            x[idx] = 1.0
        onsets.append(x)
    vel = zscore_nan(entry['topcam']['velocity'])
    ssd = zscore_nan(np.nanmax(np.vstack([entry['SSD']['cam1'],
                                          entry['SSD']['cam2']]), axis=0))
    return onsets + [vel, ssd], n


def collect_sessions(CMT, trials):
    """(pathway, mouse) -> list of (predictors, y) per session."""
    rm_mouse = {m for _, m in photom_exclusions['remove_mouse']}
    rm_hem = {(m, h) for _, m, h in photom_exclusions['remove_hemisphere']}
    rm_trial = {(m, t) for _, m, t in photom_exclusions['remove_trial']}
    out = defaultdict(list)
    for c in CMT:
        for m in CMT[c]:
            if m in rm_mouse:
                continue
            pathway = 'dSPN' if get_pathway(c, m) == 'drd1' else 'iSPN'
            for t in CMT[c][m]:
                if t not in trials or (m, t) in rm_trial:
                    continue
                e = CMT[c][m][t]
                if 'photom' not in e:
                    continue
                hems = [h for h in ('left', 'right')
                        if isinstance(e['photom'].get(h), dict) and (m, h) not in rm_hem]
                if not hems:
                    continue
                y = np.nanmean(np.vstack([e['photom'][h]['z'] for h in hems]), axis=0)
                px, n = session_predictors(e)
                if len(y) != n or not np.isfinite(y).all():
                    continue
                out[(pathway, m)].append((px, y.astype(np.float64), t))
    return out


def fit_mouse(sessions, shift_null=False, rng=None):
    """Leave-one-session-out CV R2 plus kernels fitted on all sessions."""
    designs, ys = [], []
    for px, y, _ in sessions:
        if shift_null:
            k = int(rng.integers(5 * 60 * FPS, len(y) - 5 * 60 * FPS))
            px = [np.roll(x, k) for x in px]
        designs.append(build_design(px, len(y)))
        ys.append(y)

    r2s = []
    if len(designs) >= 2:
        for i in range(len(designs)):
            Xtr = np.vstack([d for j, d in enumerate(designs) if j != i])
            ytr = np.concatenate([y for j, y in enumerate(ys) if j != i])
            mdl = Ridge(alpha=1e3).fit(Xtr, ytr)
            r2s.append(r2_score_manual(ys[i], mdl.predict(designs[i])))
            del Xtr, ytr
    X = np.vstack(designs)
    Y = np.concatenate(ys)
    from sklearn.linear_model import RidgeCV
    final = RidgeCV(alphas=ALPHAS).fit(X, Y)
    K = unpack(final.coef_, len(PREDICTORS))
    del designs, X, Y
    return K, np.array(r2s, float), float(final.alpha_)


def cluster_permutation_ind(A, B, n_perm=10000, cluster_alpha=0.05, seed=0):
    """Independent-samples cluster permutation across kernel lags.

    Shape-aware, so unlike a peak-picking summary it cannot manufacture a sign
    difference by selecting the positive lobe of one group and the negative
    lobe of the other on a biphasic kernel. Group labels are permuted.
    """
    A, B = np.asarray(A, float), np.asarray(B, float)
    nA, nB = len(A), len(B)
    if nA < 2 or nB < 2:
        return []
    pooled = np.vstack([A, B])
    dfree = nA + nB - 2
    thr = stats.t.ppf(1 - cluster_alpha / 2, dfree)

    def tstat(idx):
        a, b = pooled[idx[:nA]], pooled[idx[nA:]]
        sa, sb = a.var(0, ddof=1), b.var(0, ddof=1)
        sp = np.sqrt(((nA - 1) * sa + (nB - 1) * sb) / dfree)
        denom = sp * np.sqrt(1 / nA + 1 / nB)
        return np.divide(a.mean(0) - b.mean(0), denom,
                         out=np.zeros(pooled.shape[1]), where=denom > 0)

    def clusters_of(t):
        idx = np.where(np.abs(t) > thr)[0]
        if idx.size == 0:
            return []
        return [c for c in np.split(idx, np.where(np.diff(idx) != 1)[0] + 1) if c.size]

    base = np.arange(nA + nB)
    t_obs = tstat(base)
    obs = clusters_of(t_obs)
    masses = [float(np.sum(np.abs(t_obs[c]))) for c in obs]

    rng = np.random.default_rng(seed)
    null = np.zeros(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(base)
        tc = clusters_of(tstat(perm))
        if tc:
            null[i] = max(np.sum(np.abs(tstat(perm)[c])) for c in tc)
    return [(c, float((np.sum(null >= m) + 1) / (n_perm + 1)))
            for c, m in zip(obs, masses)]


def event_triggered_average(sessions, b_index):
    """Raw ETA for one behaviour onset -- an independent cross-check on the kernel."""
    segs = []
    for px, y, _ in sessions:
        idx = np.where(px[b_index] > 0)[0]
        for i in idx:
            a, z = i + LAG_MIN, i + LAG_MAX + 1
            if a >= 0 and z <= len(y):
                seg = y[a:z]
                segs.append(seg - np.mean(seg[:abs(LAG_MIN) // 2 or 1]))
    if not segs:
        return None
    full = np.mean(segs, axis=0)
    return full[(LAGS - LAG_MIN)]



def bout_counts(sessions):
    """Bouts per behaviour across a mouse's sessions, for reporting."""
    n = np.zeros(len(BEHAVIORS), int)
    for px, _, _ in sessions:
        for b in range(len(BEHAVIORS)):
            n[b] += int(px[b].sum())
    return n


def main():
    setup_style()
    os.makedirs(OUT, exist_ok=True)
    print('Self-test before touching real data...')
    if not selftest(verbose=False):
        print('SELF-TEST FAILED -- aborting'); sys.exit(1)
    print('  passed.\n')

    CMT = data_io.load_CMT(flatten=True)
    S = {cond: collect_sessions(CMT, trials) for cond, trials in CONDITIONS.items()}
    del CMT

    sink = open(os.path.join(OUT, 'statistics.txt'), 'w')
    sink.write('Photometry encoding model (FIR kernels)\n' + '=' * 70 + '\n')
    sink.write(f'lags {LAG_MIN/FPS:+.1f}s to {LAG_MAX/FPS:+.1f}s, '
               f'step {LAG_STEP/FPS:.2f}s ({len(LAGS)} lags)\n')
    sink.write(f'events: BOUT onsets (>= {MIN_BOUT/FPS:.0f}s long, following '
               f'>= {MIN_PRIOR/FPS:.0f}s of not doing it), plus {COVARIATES}\n')
    sink.write('Counting every label change instead diluted the locked transient\n')
    sink.write('3-10x, since ~94% of label changes are sub-second fragments.\n')
    sink.write('Ridge, alpha by generalised CV; held-out R2 = leave-one-session-out.\n')
    sink.write(f'Conditions kept separate; {BASELINE_COND} is the naive baseline and\n')
    sink.write('salineOnly is saline after a cocaine history.\n')

    rng = np.random.default_rng(0)
    K_all = {}                                  # [cond][(pathway, mouse)] = kernels
    for cond in CONDITIONS:
        K_all[cond] = {}
        sink.write('\n' + '=' * 70 + f'\n{cond.upper()} SESSIONS\n' + '=' * 70 + '\n')
        sink.write(f'  {"pathway":<8}{"mouse":<10}{"n sess":>7}{"held-out R2":>13}'
                   f'{"shifted null":>14}{"alpha":>9}\n')
        rows = []
        for (pathway, mouse), s in sorted(S[cond].items()):
            if len(s) < 1:
                continue
            K, r2, alpha = fit_mouse(s)
            _, r2n, _ = fit_mouse(s, shift_null=True, rng=rng)
            K_all[cond][(pathway, mouse)] = K
            # salineOnly is a single session per mouse, so leave-one-session-out
            # CV is undefined there; kernels are still fitted, R2 is just absent.
            has_cv = r2.size > 0
            if has_cv:
                rows.append((np.nanmean(r2), np.nanmean(r2n)))
            r2_txt = f'{np.nanmean(r2):>13.4f}' if has_cv else f'{"n/a":>13}'
            r2n_txt = f'{np.nanmean(r2n):>14.4f}' if has_cv else f'{"n/a":>14}'
            sink.write(f'  {pathway:<8}{mouse:<10}{len(s):>7}{r2_txt}{r2n_txt}{alpha:>9.0f}\n')
            print(f'  {cond:<11}{pathway} {mouse}: {len(s)} sessions, '
                  + (f'R2={np.nanmean(r2):+.4f} (null {np.nanmean(r2n):+.4f})'
                     if has_cv else 'single session, no CV'))
        if len(rows) >= 3:
            a = np.array([r[0] for r in rows]); b = np.array([r[1] for r in rows])
            t_, p_ = stats.ttest_rel(a, b)
            sink.write(f'\n  mean held-out R2 {a.mean():+.4f} vs shifted null '
                       f'{b.mean():+.4f}; t={t_:.2f}, p={p_:.2g}, N={len(a)} mice\n')

        # bout counts actually driving the fit
        sink.write(f'\n  bouts per behaviour (all mice pooled)\n')
        tot = np.zeros(len(BEHAVIORS), int)
        for s in S[cond].values():
            tot += bout_counts(s)
        for b, name in enumerate(BEHAVIORS):
            flag = '   <- too few, kernel unreliable' if tot[b] < 30 else ''
            sink.write(f'    {name:<16}{tot[b]:>7}{flag}\n')

    # ---------------- kernels figure, one per condition ----------------
    for cond in CONDITIONS:
        res = K_all[cond]
        if not res:
            continue
        fig = plt.figure(figsize=(210 * mm, 150 * mm))
        gs = gridspec.GridSpec(3, 4, figure=fig, left=0.08, right=0.98,
                               top=0.90, bottom=0.08, wspace=0.35, hspace=0.6)
        sink.write('\n' + '=' * 70 + f'\nKERNEL COMPARISON dSPN vs iSPN â€” {cond}\n')
        sink.write('=' * 70 + '\n')
        sink.write('Signed epoch means, not peak-picking. Cluster permutation over\n')
        sink.write('lags is corrected across lags; 11 predictors -> Bonferroni 0.0045.\n')
        sink.write(f'  {"predictor":<16}{"early dSPN":>11}{"early iSPN":>11}{"p_early":>9}'
                   f'{"late dSPN":>11}{"late iSPN":>11}{"p_late":>9}   clusters\n')

        for i, name in enumerate(PREDICTORS):
            ax = fig.add_subplot(gs[i // 4, i % 4])
            early = (LAG_S >= 0) & (LAG_S <= 0.5)
            late = (LAG_S > 0.5) & (LAG_S <= 2.0)
            stacks, ep = {}, {}
            for pathway, colr in (('dSPN', '#d73027'), ('iSPN', '#721515')):
                sel = [K[i] for (p, m), K in res.items() if p == pathway]
                if len(sel) < 2:
                    continue
                stack = np.vstack(sel)
                stacks[pathway] = stack
                mean, sem = stack.mean(0), stats.sem(stack, axis=0)
                ax.plot(LAG_S, mean, color=colr, lw=1.1, label=pathway)
                ax.fill_between(LAG_S, mean - sem, mean + sem, color=colr, alpha=0.22, lw=0)
                ep[pathway] = (stack[:, early].mean(1), stack[:, late].mean(1))
            if len(stacks) == 2:
                _, p_e = stats.ttest_ind(ep['dSPN'][0], ep['iSPN'][0])
                _, p_l = stats.ttest_ind(ep['dSPN'][1], ep['iSPN'][1])
                cl = cluster_permutation_ind(stacks['dSPN'], stacks['iSPN'], seed=i)
                sig = [(c, p) for c, p in cl if p < 0.05]
                cl_txt = ('none' if not sig else
                          '; '.join(f'{LAG_S[c[0]]:+.1f}..{LAG_S[c[-1]]:+.1f}s p={p:.4f}'
                                    for c, p in sig))
                for c, p in sig:
                    ax.axvspan(LAG_S[c[0]], LAG_S[c[-1]], color='k', alpha=0.07, lw=0)
                sink.write(f'  {name:<16}{ep["dSPN"][0].mean():>11.4f}'
                           f'{ep["iSPN"][0].mean():>11.4f}{p_e:>9.4f}'
                           f'{ep["dSPN"][1].mean():>11.4f}{ep["iSPN"][1].mean():>11.4f}'
                           f'{p_l:>9.4f}   {cl_txt}\n')
            ax.axvline(0, color='k', ls='--', lw=0.6)
            ax.axhline(0, color='gray', lw=0.5)
            ax.set_title(name, fontsize=7, fontweight='bold')
            ax.set_xlabel('Lag (s)', fontsize=6)
            if i % 4 == 0:
                ax.set_ylabel(r'$\beta$ (z per bout)', fontsize=6)
            if i == 0:
                ax.legend(fontsize=5.5, frameon=False)
            ax.tick_params(labelsize=6)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

        fig.suptitle(f'Encoding kernels, bout onsets with movement regressed out '
                     f'â€” {cond} sessions', fontsize=9, fontweight='bold')
        save_fig(fig, OUT, f'encoding_kernels_{cond}')
        print(f'  figure -> encoding_kernels_{cond}')

    # ------------- each condition vs naive saline, paired within mouse -------------
    for other in [c for c in CONDITIONS if c != BASELINE_COND]:
        sink.write('\n' + '=' * 70 + f'\n{other.upper()} vs {BASELINE_COND.upper()} '
                   f'kernels (paired within mouse)\n' + '=' * 70 + '\n')
        sink.write('Both windows are tested: some effects (e.g. floor licking) live in\n')
        sink.write('the early transient, so a late-only comparison would miss them.\n')
        sink.write(f'  {"predictor":<16}{"pathway":<7}{"window":<7}{other:>11}'
                   f'{BASELINE_COND:>10}{"diff":>10}{"t":>8}{"p":>9}{"N":>4}\n')
        windows = (('early', (LAG_S >= 0) & (LAG_S <= 0.5)),
                   ('late', (LAG_S > 0.5) & (LAG_S <= 2.0)))
        for i, name in enumerate(PREDICTORS):
            for pathway in ('dSPN', 'iSPN'):
                ka = {m: K for (p, m), K in K_all[other].items() if p == pathway}
                kb = {m: K for (p, m), K in K_all[BASELINE_COND].items() if p == pathway}
                shared = sorted(set(ka) & set(kb))
                if len(shared) < 4:
                    continue
                for wname, w in windows:
                    a = np.array([ka[m][i][w].mean() for m in shared])
                    b = np.array([kb[m][i][w].mean() for m in shared])
                    if max(abs(a.mean()), abs(b.mean())) < 0.02:
                        continue
                    t_, p_ = stats.ttest_rel(a, b)
                    sink.write(f'  {name:<16}{pathway:<7}{wname:<7}{a.mean():>11.4f}'
                               f'{b.mean():>10.4f}{a.mean()-b.mean():>10.4f}{t_:>8.2f}'
                               f'{p_:>9.4f}{len(shared):>4}'
                               f'{"  *" if p_ < 0.05 else ""}\n')

    # ---------------- cross-check: kernel vs raw ETA ----------------
    sink.write('\nCross-check: model kernel vs raw bout-triggered average (cocaine)\n')
    for bname in ('Floor licking', 'Grooming'):
        bi = BEHAVIORS.index(bname)
        for pathway in ('dSPN', 'iSPN'):
            ks, es = [], []
            for (p, m), s in sorted(S['cocaine'].items()):
                if p != pathway or (p, m) not in K_all['cocaine']:
                    continue
                eta = event_triggered_average(s, bi)
                if eta is not None:
                    ks.append(K_all['cocaine'][(p, m)][bi]); es.append(eta)
            if len(ks) >= 2:
                km, em = np.mean(ks, 0), np.mean(es, 0)
                sink.write(f'  {bname:<16}{pathway}: corr(kernel, BTA) = '
                           f'{np.corrcoef(km, em)[0,1]:+.3f}  '
                           f'(kernel peak {LAG_S[np.argmax(np.abs(km))]:+.2f}s, '
                           f'BTA peak {LAG_S[np.argmax(np.abs(em))]:+.2f}s)\n')
    sink.close()
    print(f'\nDone. Outputs in {OUT}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    main()


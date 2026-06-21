#!/usr/bin/env python3
"""
TysonNNSE_Linspace50Jacknife.py

NNSE for Tyson cell-cycle model:
  - 50 bins on linspace(1, 50)
  - 2500 stabilisation steps after convergence
  - Jackknife stability over N_RUNS independent full runs
  - Random sampling timed to match NNSE wall time
  - Overlayed CDFs on log-scale plot
  - All figures saved to ./NNSE_linspace_figures/
"""

import os, copy, time
import multiprocessing as mp
from collections import namedtuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ============================================================================
# CONFIGURATION
# ============================================================================

JACKKNIFE   = True
N_RUNS      = 5       # jackknife: number of independent full runs
SEED0       = 0

MAX_STEPS            = 2500    # total steps to run (first half = burn-in)
N_Vec                = 50      # number of bins / parameter vectors
MIN_FILLED_POSITIONS = 9999    # disabled — fallback burn-in at MAX_STEPS//2 always used
STABILIZATION_STEPS  = 2500    # unused (MIN_FILLED_POSITIONS never triggers)
SIGMA        = 0.01
K_INITIAL    = 1
T_START, T_END, N_TIME_POINTS = 0.0, 500.0, 501

# 50 bins: linspace(1, 50) → 51 boundaries; override last as sentinel
bin_thresholds       = np.linspace(1, 50, N_Vec + 1).copy()
bin_thresholds[-1]   = 1000.0

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "NNSE_linspace_figures")

# ============================================================================
# BASE PARAMETERS (Tyson 1991 cell-cycle model)
# ============================================================================

p0 = {
    "k1_aa_over_CT": 0.015, "k2": 0.0,       "k3_CT": 200.0,
    "k4":  180.0,            "k4prime": 0.018, "k5_minusP": 0.0,
    "k6":  1.0,              "k7":  0.6,       "k8_minusP": 100.0,
    "k9":  50.0,             "CT":  1.0,
}
param_names = ["k1_aa_over_CT", "k3_CT", "k4", "k4prime", "k6", "k7"]
p0_vec      = np.array([p0[name] for name in param_names])
n_params    = len(param_names)
CT          = p0["CT"]
t_eval      = np.linspace(T_START, T_END, N_TIME_POINTS)

# Set in main() before forking workers
_ref_t  = None
_ref_YT = None
_ref_M  = None

# ============================================================================
# MODEL & OBJECTIVE
# ============================================================================

def _ode_rhs(t, x, p):
    C2, CP, pM, M, Y, YP = x
    k3 = p["k3_CT"] / p["CT"]
    k1 = p["k1_aa_over_CT"] * p["CT"]
    FM = p["k4prime"] + p["k4"] * (M / p["CT"])**2
    return np.array([
        p["k6"]*M - p["k8_minusP"]*C2 + p["k9"]*CP,
        -k3*CP*Y + p["k8_minusP"]*C2 - p["k9"]*CP,
        k3*CP*Y - pM*FM + p["k5_minusP"]*M,
        pM*FM - p["k5_minusP"]*M - p["k6"]*M,
        k1 - p["k2"]*Y - k3*CP*Y,
        p["k6"]*M - p["k7"]*YP,
    ])

def _simulate(p_dict):
    y0  = np.array([0.9, 0.05, 0.0, 0.005, 0.3, 0.0])
    sol = solve_ivp(lambda t, x: _ode_rhs(t, x, p_dict),
                    (t_eval[0], t_eval[-1]), y0,
                    method='BDF', t_eval=t_eval, rtol=1e-6, atol=1e-8)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.t, sol.y

def _obs(Y):
    C2, CP, pM, M, Y_, YP = Y
    YT = Y_ + YP + pM + M
    return YT / CT, M / CT

def sim(P_vec):
    p = copy.deepcopy(p0)
    for i, name in enumerate(param_names):
        p[name] = P_vec[i]
    try:
        t, y       = _simulate(p)
        YT, M_arr  = _obs(y)
        YT0i = np.interp(t, _ref_t, _ref_YT)
        M0i  = np.interp(t, _ref_t, _ref_M)
        trap = getattr(np, 'trapezoid', None) or getattr(np, 'trapz', np.trapezoid)
        return float(trap((YT - YT0i)**2 + (M_arr - M0i)**2, t))
    except Exception:
        return np.inf

# ============================================================================
# WORKERS (must be picklable for pool.map)
# ============================================================================

def _mutate_worker(args):
    i, xi, fxi = args
    if xi is None:
        return (i, None, None)
    u  = (xi / (2.0 * p0_vec) + np.random.normal(0, SIGMA, n_params)) % 1.0
    xp = 2.0 * p0_vec * u
    fp = sim(xp)
    if fp > bin_thresholds[i]:
        return (i, xi.copy(), fxi)
    return (i, xp, fp)

def _rand_worker(_):
    x  = 2.0 * p0_vec * np.random.uniform(0, 1, n_params)
    fx = sim(x)
    return fx if np.isfinite(fx) else None

# ============================================================================
# TYSON FUNC: mutate → accept/reject → permute → fill vacancies
# ============================================================================

def TysonFunc(X_list, fX_list, pool=None):
    n = len(X_list)

    # Mutate (parallel or serial)
    if pool is not None:
        items = [(i, X_list[i], fX_list[i]) for i in range(n) if X_list[i] is not None]
        rd    = {idx: (x, fx) for idx, x, fx in pool.map(_mutate_worker, items)} if items else {}
        Xp  = [rd[i][0] if i in rd else None for i in range(n)]
        fXp = [rd[i][1] if i in rd else None for i in range(n)]
    else:
        Xp, fXp = [], []
        for i in range(n):
            _, x, fx = _mutate_worker((i, X_list[i], fX_list[i]))
            Xp.append(x); fXp.append(fx)

    # Permute: bubble values upward
    v, fv = list(Xp), list(fXp)
    empty_before = {i for i in range(n) if v[i] is None}
    swaps = []
    for i in range(n - 1, 0, -1):
        if fv[i] is not None and fv[i] <= bin_thresholds[i - 1]:
            if v[i] is not None and v[i - 1] is not None:
                v[i], v[i-1] = v[i-1].copy(), v[i].copy()
            else:
                v[i], v[i-1] = v[i-1], v[i]
            fv[i], fv[i-1] = fv[i-1], fv[i]
            swaps.append((i, i - 1))

    # Fill newly vacated positions
    newly_empty = {i for i in range(n) if v[i] is None} - empty_before
    for ep in newly_empty:
        for _ in range(1000):
            xn = 2.0 * p0_vec * np.random.uniform(0, 1, n_params)
            fn = sim(xn)
            placed = False
            for pos in range(n):
                if fn <= bin_thresholds[pos]:
                    if v[pos] is None or fn < fv[pos]:
                        v[pos]  = xn
                        fv[pos] = fn
                        placed  = True
                        break
            if placed:
                break

    return v, fv, fXp, swaps

# ============================================================================
# SINGLE NNSE RUN
# ============================================================================

def run_one_nnse(seed=None, pool=None, verbose=True, collect_all_fX=False,
                 print_interval=50):
    """
    Returns (volume_ratios, wall_time, final_fX [, all_fX_flat], steps)
      volume_ratios : ndarray (N_Vec,)   V[i-1]/V[i] for i=1..N_Vec
      wall_time     : float seconds
      final_fX      : list of finite f(x) values in the last state
      all_fX_flat   : list of every finite f(x) seen (only if collect_all_fX)
      steps         : int total steps executed
    """
    if seed is not None:
        np.random.seed(int(seed))

    n = N_Vec
    X_list, fX_list = [None]*n, [None]*n

    # Initialise worst K positions, avoiding the two worst bins
    for idx in range(n - K_INITIAL, n):
        for attempt in range(100):
            u  = np.random.uniform(0, 1, n_params)
            xi = 2.0 * p0_vec * u
            fi = sim(xi)
            bp = next((pos for pos in range(n) if fi <= bin_thresholds[pos]), n - 1)
            if bp < n - 2 or attempt == 99:
                X_list[idx]  = xi
                fX_list[idx] = fi
                break

    conv_step   = None
    rem_stab    = None
    BURN_FB     = MAX_STEPS // 2
    swap_cnt    = np.zeros(n + 1)
    opp_cnt     = np.zeros(n + 1)
    burned_in   = False
    all_fX_flat = [] if collect_all_fX else None

    t0 = time.time()
    t_last_print = t0
    step = 0
    while step < MAX_STEPS:
        X_list, fX_list, _, swaps = TysonFunc(X_list, fX_list, pool=pool)

        if collect_all_fX:
            all_fX_flat.extend(
                fx for fx in fX_list if fx is not None and np.isfinite(fx))

        filled = sum(fx is not None for fx in fX_list)

        if verbose and step > 0 and step % print_interval == 0:
            elapsed   = time.time() - t0
            phase     = "stab" if conv_step is not None else "fill"
            stab_done = STABILIZATION_STEPS - rem_stab if rem_stab is not None else 0
            print(f"  step {step:5d}  filled {filled:2d}/{N_Vec}  "
                  f"phase={phase}  stab={stab_done}/{STABILIZATION_STEPS}  "
                  f"elapsed={elapsed/60:.1f}min", flush=True)
            t_last_print = time.time()

        if conv_step is None and filled >= MIN_FILLED_POSITIONS:
            conv_step  = step
            rem_stab   = STABILIZATION_STEPS
            burned_in  = True
            swap_cnt[:] = opp_cnt[:] = 0
            if verbose:
                print(f"  Converged at step {step+1} ({filled}/{n} filled)")

        if rem_stab is not None:
            rem_stab -= 1
            if rem_stab == 0:
                step += 1
                break

        eff_burn = conv_step if conv_step is not None else BURN_FB
        if step >= eff_burn:
            if not burned_in:
                swap_cnt[:] = opp_cnt[:] = 0
                burned_in = True
            swap_set = set(swaps)
            for i in range(1, n):
                if fX_list[i] is not None and fX_list[i - 1] is not None:
                    opp_cnt[i]  += 1
                    if (i, i - 1) in swap_set:
                        swap_cnt[i] += 1

        step += 1

    wall_time = time.time() - t0

    vr = np.full(n + 1, np.nan)
    for i in range(1, n + 1):
        if opp_cnt[i] > 0:
            vr[i] = swap_cnt[i] / opp_cnt[i]

    final_fX = [fx for fx in fX_list if fx is not None and np.isfinite(fx)]

    if collect_all_fX:
        return vr[1:], wall_time, final_fX, all_fX_flat, step
    return vr[1:], wall_time, final_fX, step

# ============================================================================
# RANDOM SAMPLING (wall-time budget)
# ============================================================================

def random_sampling_timed(budget_s, pool=None):
    samples = []
    t0    = time.time()
    batch = max(N_Vec, 20)
    while time.time() - t0 < budget_s:
        if pool is not None:
            res = pool.map(_rand_worker, range(batch))
        else:
            res = [_rand_worker(None) for _ in range(batch)]
        samples.extend(r for r in res if r is not None)
    return np.array(samples)

# ============================================================================
# JACKKNIFE
# ============================================================================

JKResult = namedtuple('JKResult', [
    'names', 'theta_hat', 'bias', 'se', 'rel_bias', 'rel_se',
    'ok_rel_bias', 'ok_rel_se', 'runs', 'final_fX_per_run'
])

def run_jackknife(n_runs, seed0, pool):
    print(f"\n=== Jackknife: {n_runs} independent runs ===")
    t_all        = time.time()
    all_ratios   = []
    all_final_fX = []

    for i in range(n_runs):
        seed = seed0 + i
        print(f"  [{i+1}/{n_runs}] seed={seed} ...", end=' ', flush=True)
        vr, wt, ffX, _ = run_one_nnse(seed=seed, pool=pool, verbose=False)
        all_ratios.append(vr)
        all_final_fX.append(ffX)
        elapsed = time.time() - t_all
        eta     = elapsed / (i + 1) * (n_runs - i - 1)
        print(f"done {wt:.0f}s | ETA {eta:.0f}s")

    runs      = np.array(all_ratios)   # (n_runs, N_Vec)
    theta_hat = np.nanmean(runs, axis=0)
    n_r       = len(runs)
    jk_ests   = np.array([np.nanmean(np.delete(runs, i, 0), axis=0) for i in range(n_r)])
    bias      = (n_r - 1) * (np.nanmean(jk_ests, axis=0) - theta_hat)
    se        = np.sqrt((n_r - 1) * np.nanmean((jk_ests - theta_hat)**2, axis=0))
    safe_th   = np.where(np.abs(theta_hat) > 1e-12, theta_hat, np.nan)
    rel_bias  = np.abs(bias / safe_th)
    rel_se    = np.abs(se   / safe_th)
    names     = [f"V[{i-1}]/V[{i}]" for i in range(1, N_Vec + 1)]

    return JKResult(names, theta_hat, bias, se, rel_bias, rel_se,
                    rel_bias < 0.05, rel_se < 0.05, runs, all_final_fX)

# ============================================================================
# HELPER: assign f(x) value to bin index
# ============================================================================

def _bin_idx(fx):
    """Return the first bin index j such that fx <= bin_thresholds[j], else N_Vec."""
    for j in range(N_Vec + 1):
        if fx <= bin_thresholds[j]:
            return j
    return N_Vec

# ============================================================================
# PLOTTING
# ============================================================================

def _save(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_cdf_overlay(nnse_all_fX, random_fX, jk=None):
    """
    Overlayed empirical CDFs evaluated at each bin threshold.
    x-axis = bin threshold value (log scale).
    Left panel: linear y; right panel: log y.

    nnse_all_fX : all f(x) values visited during the main NNSE run
    random_fX   : ndarray of random-sampling f(x) values
    jk          : JKResult — per-run CDFs from jackknife (thin background lines)
    """
    thresholds = bin_thresholds[:-1]   # drop sentinel

    def ecdf_at(fX_arr, ts):
        arr = np.asarray(fX_arr)
        arr = arr[np.isfinite(arr)]
        return np.array([np.mean(arr <= t) for t in ts])

    nnse_arr = np.asarray(nnse_all_fX)
    nnse_arr = nnse_arr[np.isfinite(nnse_arr)]
    rand_arr = random_fX[np.isfinite(random_fX)]

    nnse_cdf = ecdf_at(nnse_arr, thresholds)
    rand_cdf = ecdf_at(rand_arr, thresholds)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, yscale in zip(axes, ['linear', 'log']):
        # Jackknife per-run CDFs (thin background)
        if jk is not None:
            for run_fX in jk.final_fX_per_run:
                cdf = ecdf_at(run_fX, thresholds)
                v   = (cdf > 0) if yscale == 'log' else slice(None)
                ax.plot(thresholds[v], cdf[v],
                        lw=0.8, alpha=0.25, color='steelblue')

        # Random
        v = (rand_cdf > 0) if yscale == 'log' else slice(None)
        ax.plot(thresholds[v], rand_cdf[v], '--', lw=2, color='coral',
                label=f'Random  (n={len(rand_arr):,})')

        # NNSE (all visited states)
        v2 = (nnse_cdf > 0) if yscale == 'log' else slice(None)
        ax.plot(thresholds[v2], nnse_cdf[v2], '-', lw=2.5, color='navy',
                label=f'NNSE visited  (n={len(nnse_arr):,})')

        ax.set_xscale('log')
        if yscale == 'log':
            ax.set_yscale('log')

        ax.set_xlabel('Bin threshold  (log scale)', fontsize=12)
        ax.set_ylabel(f'CDF  ({yscale} scale)', fontsize=12)
        ax.set_title(f'CDF — log x / {yscale} y', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, which='both')

        for t in thresholds:
            ax.axvline(t, color='lightgray', lw=0.4, zorder=0)

    fig.suptitle('NNSE vs Random: CDF as function of bin thresholds  '
                 f'[linspace(1,50), {N_Vec} bins]',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, 'cdf_overlay_logx.png')


def plot_volume_ratios_comparison(vr_nnse, rand_vr):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    positions = np.arange(N_Vec)
    w = 0.4

    valid = np.isfinite(vr_nnse) & np.isfinite(rand_vr)
    ax = axes[0]
    ax.bar(positions[valid] - w/2, vr_nnse[valid], w, label='NNSE',   alpha=0.7, color='steelblue')
    ax.bar(positions[valid] + w/2, rand_vr[valid],  w, label='Random', alpha=0.7, color='coral')
    ax.set_xlabel('Position index i', fontsize=12)
    ax.set_ylabel('V[i−1]/V[i]  (log)', fontsize=12)
    ax.set_title('Volume ratios: NNSE vs Random', fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    ax.plot(positions[np.isfinite(vr_nnse)], vr_nnse[np.isfinite(vr_nnse)],
            'o-', lw=1.5, ms=5, color='steelblue', label='NNSE')
    ax.plot(positions[np.isfinite(rand_vr)], rand_vr[np.isfinite(rand_vr)],
            's--', lw=1.5, ms=5, color='coral', label='Random')
    ax.set_xlabel('Position index i', fontsize=12)
    ax.set_ylabel('V[i−1]/V[i]', fontsize=12)
    ax.set_title('Volume ratios (line)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Volume ratios  [{N_Vec} linspace bins]', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, 'volume_ratios_comparison.png')


def plot_jackknife_results(jk):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    positions = np.arange(len(jk.theta_hat))

    ax = axes[0]
    ax.errorbar(positions, jk.theta_hat, yerr=jk.se,
                fmt='o-', capsize=4, ms=5, lw=1.5, color='steelblue', label='Mean ± SE')
    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('Volume ratio', fontsize=12)
    ax.set_title(f'Jackknife volume ratios  (N={len(jk.runs)} runs)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    ax = axes[1]
    ax.bar(positions, jk.rel_se, color='coral', alpha=0.7, edgecolor='black', lw=0.5)
    ax.axhline(0.05, color='red', ls='--', lw=2, label='5% threshold')
    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('Relative SE', fontsize=12)
    ax.set_title('Jackknife: relative standard error', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[2]
    n_show = min(8, jk.runs.shape[1])
    colors = plt.cm.viridis(np.linspace(0, 1, n_show))
    for i in range(n_show):
        vals  = jk.runs[:, i]
        valid = ~np.isnan(vals)
        ax.plot(np.where(valid)[0], vals[valid], 'o-', ms=5,
                color=colors[i], alpha=0.8, label=f'V[{i}]/V[{i+1}]')
    ax.set_xlabel('Run index', fontsize=12)
    ax.set_ylabel('Volume ratio', fontsize=12)
    ax.set_title(f'Run-to-run variability  (first {n_show} positions)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Jackknife Stability Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, 'jackknife_results.png')


def plot_comparison_2x2(nnse_final_fX, random_fX, vr_nnse, rand_vr):
    """2×2 summary: bin counts, volume ratios, CDF, best-per-bin."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    w = 0.35

    # Bin counts
    nnse_counts = np.zeros(N_Vec + 1, dtype=int)
    for fx in nnse_final_fX:
        if np.isfinite(fx):
            nnse_counts[_bin_idx(fx)] += 1

    rand_counts = np.zeros(N_Vec + 1, dtype=int)
    for fx in random_fX:
        if np.isfinite(fx):
            rand_counts[_bin_idx(fx)] += 1

    bi = np.arange(N_Vec + 1)
    ax = axes[0, 0]
    ax.bar(bi - w/2, nnse_counts, w, label='NNSE',   alpha=0.7, color='steelblue')
    ax.bar(bi + w/2, rand_counts, w, label='Random', alpha=0.7, color='coral')
    ax.set_yscale('log')
    ax.set_xlabel('Bin index'); ax.set_ylabel('Count')
    ax.set_title('Bin occupancy'); ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    # Volume ratios
    positions = np.arange(N_Vec)
    valid = np.isfinite(vr_nnse) & np.isfinite(rand_vr)
    ax = axes[0, 1]
    ax.bar(positions[valid] - w/2, vr_nnse[valid], w, label='NNSE',   alpha=0.7, color='steelblue')
    ax.bar(positions[valid] + w/2, rand_vr[valid],  w, label='Random', alpha=0.7, color='coral')
    ax.set_yscale('log')
    ax.set_xlabel('Position'); ax.set_ylabel('V[i−1]/V[i]')
    ax.set_title('Volume ratios'); ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    # CDF (log x)
    ax = axes[1, 0]
    nf = np.sort([fx for fx in nnse_final_fX if np.isfinite(fx) and fx > 0])
    rf = np.sort(random_fX[np.isfinite(random_fX) & (random_fX > 0)])
    if len(nf):
        ax.semilogx(nf, np.arange(1, len(nf)+1)/len(nf),
                    '-',  lw=2.5, color='steelblue', label=f'NNSE final  (n={len(nf)})')
    if len(rf):
        ax.semilogx(rf, np.arange(1, len(rf)+1)/len(rf),
                    '--', lw=2,   color='coral',     label=f'Random  (n={len(rf):,})')
    ax.set_xlabel('f(x)  (log)'); ax.set_ylabel('CDF')
    ax.set_title('CDF comparison'); ax.legend(); ax.grid(True, alpha=0.3, which='both')

    # Best f per bin
    def best_per_bin(fX_arr):
        best = np.full(N_Vec + 1, np.nan)
        for j in range(N_Vec + 1):
            lo = bin_thresholds[j-1] if j > 0 else 0.0
            hi = bin_thresholds[j]
            vals = [fx for fx in fX_arr if np.isfinite(fx) and lo < fx <= hi]
            if vals:
                best[j] = min(vals)
        return best

    nb = best_per_bin(fx for fx in nnse_final_fX if fx is not None)
    rb = best_per_bin(random_fX)
    ax = axes[1, 1]
    ax.semilogy(bi, nb, 'o-',  ms=5, lw=1.5, color='steelblue', label='NNSE')
    ax.semilogy(bi, rb, 's--', ms=5, lw=1.5, color='coral',     label='Random')
    ax.set_xlabel('Bin index'); ax.set_ylabel('Best f(x) in bin')
    ax.set_title('Best value per bin'); ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle(f'NNSE vs Random — {N_Vec} linspace bins  (1→50)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, 'comparison_2x2.png')


# ============================================================================
# MAIN
# ============================================================================

def main():
    global _ref_t, _ref_YT, _ref_M

    os.makedirs(FIGURES_DIR, exist_ok=True)
    print(f"Output directory : {FIGURES_DIR}")
    print(f"Bins             : linspace(1, 50, {N_Vec+1})  N_Vec={N_Vec}")
    print(f"Steps            : MAX={MAX_STEPS}  STAB={STABILIZATION_STEPS}\n")

    # Reference simulation — set globals BEFORE forking workers
    print("Running reference simulation (p0)...")
    _ref_t, y_ref = _simulate(p0)
    _ref_YT, _ref_M = _obs(y_ref)
    print("✓ Reference done\n")

    ctx  = mp.get_context('fork')
    pool = ctx.Pool(processes=mp.cpu_count())
    print(f"Worker pool: {mp.cpu_count()} cores\n")

    try:
        # ---- Main NNSE run ----
        print("=== Main NNSE run ===")
        vr_main, wt_main, final_fX, all_fX, steps = run_one_nnse(
            seed=42, pool=pool, verbose=True, collect_all_fX=True)
        print(f"✓ Done in {wt_main:.1f}s  ({steps} steps)  "
              f"{np.sum(~np.isnan(vr_main))}/{N_Vec} valid volume ratios\n")

        # ---- Random sampling (same wall time) ----
        print(f"=== Random sampling  ({wt_main:.1f}s budget) ===")
        random_fX = random_sampling_timed(wt_main, pool=pool)
        print(f"✓ {len(random_fX):,} random samples\n")

        # Volume ratios from random sampling
        rand_counts = np.zeros(N_Vec + 1, dtype=int)
        for fx in random_fX:
            rand_counts[_bin_idx(fx)] += 1
        rand_vr = np.full(N_Vec, np.nan)
        for i in range(N_Vec):
            if rand_counts[i + 1] > 0:
                rand_vr[i] = rand_counts[i] / rand_counts[i + 1]

        # ---- Jackknife ----
        jk = None
        if JACKKNIFE:
            jk = run_jackknife(N_RUNS, SEED0, pool)
            print(f"\n✓ Jackknife done: "
                  f"OK bias {np.sum(jk.ok_rel_bias)}/{N_Vec}  "
                  f"OK SE {np.sum(jk.ok_rel_se)}/{N_Vec}\n")

        # ---- Plots ----
        print("=== Saving plots ===")
        plot_cdf_overlay(all_fX, random_fX, jk=jk)
        plot_volume_ratios_comparison(vr_main, rand_vr)
        plot_comparison_2x2(final_fX, random_fX, vr_main, rand_vr)
        if jk is not None:
            plot_jackknife_results(jk)
        print(f"\n✓ All plots saved to {FIGURES_DIR}")

    finally:
        pool.close()
        pool.join()


if __name__ == '__main__':
    main()

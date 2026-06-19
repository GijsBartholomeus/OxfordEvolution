#!/usr/bin/env python3
"""
Run MCMC for geometric d=8..22 with snapshot saving, then compute
Hamming-distance connected correlation G(ℓ) and Landau order parameter m₁²
from each equilibrated snapshot.

Diagnostics:
  G(ℓ) = q(ℓ) − q_rand
    q(ℓ)   = fraction of same-phenotype pairs sampled at Hamming distance ℓ
    q_rand = Σ_a n_a(n_a−1) / [N(N−1)]
  ξ fitted from G(ℓ) ~ A exp(−ℓ/ξ)

  m₁²   = Σ_{a,j} M_{aj}²
  m₁w²  = Σ_{a,j} M_{aj}² / f_a   (weighted, boosts rare classes)
    M_{aj} = (1/N) Σ_x (−1)^{x_j} φ_a(x),  φ_a(x) = δ(σ_x, a)

Outputs: runs/diagnostics/geometric_d{d}_diag/diag_summary.tsv
"""
import json, shutil, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))

import run_hamming_sims as base
from run_d24_parallel import run_single_temp
from run_dseries import counts_geometric, make_geom_grid

DIAG_DIR  = ROOT / "runs" / "diagnostics"
PLOTS_DIR = ROOT / "plots" / "mergedheatplots"

D_LIST    = [8, 10, 12, 14, 16, 18, 20, 22]
N_WORKERS = 12
THIN      = 1
N_PAIRS   = 10_000   # random pairs sampled per distance for G(ℓ)

# Sweep counts: enough to equilibrate, short enough to be fast
DIAG_SWEEPS = {8: 1000, 10: 1000, 12: 800, 14: 600,
               16: 400,  18: 300,  20: 200, 22: 150}
DIAG_BURN   = {d: s // 2 for d, s in DIAG_SWEEPS.items()}

INIT = "ordered"


# ── diagnostic computation ────────────────────────────────────────────────────

def compute_diagnostics(omega, d, counts, rng=None):
    """
    omega  : (N,) uint8 array of phenotype labels 0..Q-1
    counts : (Q,) expected class counts
    Returns: dict with q_rand, q_ell (list), G_ell (list), xi, m1_sq, m1w_sq
    """
    if rng is None:
        rng = np.random.default_rng()

    N = 1 << d
    Q = len(counts)
    f = counts.astype(float) / counts.sum()

    # ── G(ℓ): Hamming-distance connected correlation ──────────────────────
    n_a = np.bincount(omega.astype(int), minlength=Q).astype(float)
    q_rand = float(np.dot(n_a, n_a - 1) / (N * (N - 1)))

    q_ell = np.zeros(d)
    for ell in range(1, d + 1):
        # Sample N_PAIRS random pairs at Hamming distance ell.
        # For each pair: pick x uniformly, then flip exactly ell distinct bits.
        noise  = rng.random((N_PAIRS, d))
        perm   = np.argsort(noise, axis=1)          # (N_PAIRS, d)
        bits   = perm[:, :ell]                       # (N_PAIRS, ell) — bit positions to flip
        shifts = np.zeros(N_PAIRS, dtype=np.int64)
        for k in range(ell):
            shifts |= (np.int64(1) << bits[:, k].astype(np.int64))
        x_idx  = rng.integers(0, N, N_PAIRS, dtype=np.int64)
        y_idx  = x_idx ^ shifts
        q_ell[ell - 1] = float(np.mean(omega[x_idx] == omega[y_idx]))

    G_ell = q_ell - q_rand

    # Fit ξ from G(ℓ) ~ A exp(−ℓ/ξ).
    # Use only the initial consecutive positive hump of G to avoid fitting noise
    # at large ℓ where G fluctuates around 0.
    xi = float("nan")
    ell_arr = np.arange(1, d + 1, dtype=float)
    try:
        # 3-sigma noise floor based on sampling statistics
        sigma_G   = np.sqrt(max(q_rand, 1e-6) * (1 - max(q_rand, 1e-6)) / N_PAIRS)
        threshold = max(3.0 * sigma_G, 5e-3)

        # Walk along ℓ=1,2,... keeping only the initial run above threshold
        n_good = 0
        for g in G_ell:
            if g > threshold:
                n_good += 1
            else:
                break

        if n_good >= 3 and G_ell[0] > threshold:
            popt, _ = curve_fit(
                lambda x, A, xi_: A * np.exp(-x / xi_),
                ell_arr[:n_good], G_ell[:n_good],
                p0=[max(G_ell[0], 1e-4), 2.0],
                maxfev=2000,
                bounds=([0, 0.05], [1.0, float(d)]),
            )
            xi_fit = float(popt[1])
            if xi_fit < 0.95 * d:
                xi = xi_fit
    except Exception:
        pass

    # ── m₁²: Landau order parameter ──────────────────────────────────────
    # u_j(x) = (−1)^{x_j} = 1 − 2·bit_j(x)
    # M_{aj} = (1/N) Σ_{x: σ_x=a} u_j(x)
    genotypes = np.arange(N, dtype=np.uint32)
    bit_pos   = np.arange(d, dtype=np.uint32)
    u = (1 - 2 * ((genotypes[:, None] >> bit_pos[None, :]) & 1)).astype(np.int8)

    group_sum = np.zeros((Q, d), dtype=np.float64)
    for j in range(d):
        group_sum[:, j] = np.bincount(omega.astype(int),
                                      weights=u[:, j].astype(float),
                                      minlength=Q)
    M = group_sum / N
    M_sq = M ** 2

    m1_sq = float(M_sq.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = np.where(f > 0, 1.0 / f, 0.0)
        m1w_sq = float((M_sq * weight[:, None]).sum())

    return {
        "q_rand": q_rand,
        "q_ell":  q_ell.tolist(),
        "G_ell":  G_ell.tolist(),
        "xi":     xi,
        "m1_sq":  m1_sq,
        "m1w_sq": m1w_sq,
    }


# ── per-d run ──────────────────────────────────────────────────────────────────

def run_diag_case(d):
    counts  = counts_geometric(d)
    lam1    = d - 2
    temps   = make_geom_grid(lam1)
    n_temps = len(temps)
    Q       = len(counts)
    N       = 1 << d
    sweeps  = DIAG_SWEEPS[d]
    burn    = DIAG_BURN[d]
    seed    = 9200 + d

    name    = f"geometric_d{d}_diag"
    run_dir = DIAG_DIR / name

    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    counts_path = run_dir / "counts.tsv"
    base.write_counts(counts_path, counts)

    print(f"\n{'='*55}\n{name}  d={d}  N={N:,}  Q={Q}  {n_temps} temps  "
          f"{sweeps}+{burn} sweeps", flush=True)

    # ── run all temperatures in parallel with save_flag=True ──────────────
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {
            pool.submit(run_single_temp,
                        gi, float(temps[gi]), True,
                        run_dir, counts_path,
                        d, sweeps, burn, THIN, seed): gi
            for gi in range(n_temps)
        }
        done = 0
        for fut in as_completed(futures):
            gi, wdir, rt, rc = fut.result()
            done += 1
            if done % 12 == 0 or done == n_temps:
                print(f"  [{done:3d}/{n_temps}] last T*={temps[gi]:.4f}  {rt:.1f}s", flush=True)

    wall_time = time.perf_counter() - wall_start
    print(f"  Sims done in {wall_time:.0f}s.  Computing diagnostics...", flush=True)

    # ── compute diagnostics from snapshots ────────────────────────────────
    np_dtype = np.dtype(base.dtype_for_q(Q))
    rng      = np.random.default_rng(seed=42)
    results  = []

    for gi in range(n_temps):
        wdir      = run_dir / f"worker_{gi:04d}"
        snap_path = wdir / "omega_T000.bin"

        if not snap_path.exists():
            print(f"  WARNING: no snapshot for gi={gi} T*={temps[gi]:.4f}", flush=True)
            results.append({"temp_index": gi, "temp": float(temps[gi]),
                            "temp_over_lam1": float(temps[gi] / lam1),
                            "q_rand": float("nan"), "xi": float("nan"),
                            "m1_sq": float("nan"), "m1w_sq": float("nan")})
            continue

        omega = np.fromfile(snap_path, dtype=np_dtype)
        diag  = compute_diagnostics(omega, d, counts, rng=rng)

        row = {
            "temp_index":    gi,
            "temp":          float(temps[gi]),
            "temp_over_lam1": float(temps[gi] / lam1),
            "q_rand":        diag["q_rand"],
            "xi":            diag["xi"],
            "m1_sq":         diag["m1_sq"],
            "m1w_sq":        diag["m1w_sq"],
        }
        for ell, (q_v, G_v) in enumerate(zip(diag["q_ell"], diag["G_ell"]), 1):
            row[f"q_{ell}"] = q_v
            row[f"G_{ell}"] = G_v

        results.append(row)

    # ── save diagnostics TSV ──────────────────────────────────────────────
    if results:
        cols = list(results[0].keys())
        with (run_dir / "diag_summary.tsv").open("w") as f:
            f.write("\t".join(cols) + "\n")
            for r in results:
                f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")

    diag_time = time.perf_counter() - wall_start - wall_time
    print(f"  Diagnostics done in {diag_time:.0f}s  →  {run_dir/'diag_summary.tsv'}",
          flush=True)
    return results


# ── recompute from existing snapshots (no MCMC) ───────────────────────────────

def recompute_diag_case(d):
    """Recompute diagnostics from existing omega snapshots; skip if no snapshots found."""
    counts  = counts_geometric(d)
    lam1    = d - 2
    temps   = make_geom_grid(lam1)
    n_temps = len(temps)
    Q       = len(counts)

    name    = f"geometric_d{d}_diag"
    run_dir = DIAG_DIR / name

    if not run_dir.exists():
        print(f"  {name}: directory not found, skipping.", flush=True)
        return []

    np_dtype = np.dtype(base.dtype_for_q(Q))
    rng      = np.random.default_rng(seed=42)
    results  = []

    print(f"\n{'='*55}\nRecomputing {name}  d={d}  {n_temps} temps", flush=True)

    for gi in range(n_temps):
        wdir      = run_dir / f"worker_{gi:04d}"
        snap_path = wdir / "omega_T000.bin"

        if not snap_path.exists():
            print(f"  WARNING: no snapshot for gi={gi}", flush=True)
            results.append({"temp_index": gi, "temp": float(temps[gi]),
                            "temp_over_lam1": float(temps[gi] / lam1),
                            "q_rand": float("nan"), "xi": float("nan"),
                            "m1_sq": float("nan"), "m1w_sq": float("nan")})
            continue

        omega = np.fromfile(snap_path, dtype=np_dtype)
        diag  = compute_diagnostics(omega, d, counts, rng=rng)

        row = {
            "temp_index":    gi,
            "temp":          float(temps[gi]),
            "temp_over_lam1": float(temps[gi] / lam1),
            "q_rand":        diag["q_rand"],
            "xi":            diag["xi"],
            "m1_sq":         diag["m1_sq"],
            "m1w_sq":        diag["m1w_sq"],
        }
        for ell, (q_v, G_v) in enumerate(zip(diag["q_ell"], diag["G_ell"]), 1):
            row[f"q_{ell}"] = q_v
            row[f"G_{ell}"] = G_v

        results.append(row)

    if results:
        cols = list(results[0].keys())
        with (run_dir / "diag_summary.tsv").open("w") as f:
            f.write("\t".join(cols) + "\n")
            for r in results:
                f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")

    print(f"  Updated: {run_dir / 'diag_summary.tsv'}", flush=True)
    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    base.compile_core()

    for d in D_LIST:
        run_diag_case(d)

    print("\nAll diagnostic runs done.")


def recompute_all():
    """Recompute diagnostics for all finished d values (no MCMC)."""
    for d in D_LIST:
        recompute_diag_case(d)
    print("\nRecompute done.")


if __name__ == "__main__":
    import sys
    if "--recompute" in sys.argv:
        recompute_all()
    else:
        main()

#!/usr/bin/env python3
"""
Uniform-distribution diagnostic runs: same G(ℓ)/m₁²/m₁w² diagnostics
as run_geom_diagnostics.py but for the uniform phenotype distribution.

Uniform: Q = 2^(d/2) equal classes, each with N/Q = 2^(d/2) genotypes.
Ground-state predictions:
  m₁²   = d / (2Q)            → small, shrinks with d
  m₁w²  = d / 2               → grows with d, same for all Q at fixed d
  q_rand = 1/Q                 → shrinks with d

Outputs: runs/diagnostics/uniform_d{d}_diag/diag_summary.tsv
"""
import shutil, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))

import run_hamming_sims as base
from run_d24_parallel import run_single_temp
from run_dseries import make_uniform_grid
from run_geom_diagnostics import compute_diagnostics   # reuse from geometric

DIAG_DIR  = ROOT / "runs" / "diagnostics"
PLOTS_DIR = ROOT / "plots" / "mergedheatplots"

D_LIST    = [8, 10, 12, 14, 16, 18, 20]   # d=22 excluded: N=4M cache thrash
N_WORKERS = 12
THIN      = 1
INIT      = "ordered"

DIAG_SWEEPS = {8: 1000, 10: 1000, 12: 800, 14: 600,
               16: 400,  18: 300,  20: 200}
DIAG_BURN   = {d: s // 2 for d, s in DIAG_SWEEPS.items()}


def run_diag_case(d):
    counts  = base.counts_uniform(d)
    lam1    = d - 2
    temps   = make_uniform_grid(lam1)
    n_temps = len(temps)
    Q       = len(counts)
    N       = 1 << d
    sweeps  = DIAG_SWEEPS[d]
    burn    = DIAG_BURN[d]
    seed    = 8300 + d

    name    = f"uniform_d{d}_diag"
    run_dir = DIAG_DIR / name

    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    counts_path = run_dir / "counts.tsv"
    base.write_counts(counts_path, counts)

    print(f"\n{'='*55}\n{name}  d={d}  N={N:,}  Q={Q}  {n_temps} temps  "
          f"{sweeps}+{burn} sweeps", flush=True)

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
                print(f"  [{done:3d}/{n_temps}] last T*={temps[gi]:.4f}  {rt:.1f}s",
                      flush=True)

    wall_time = time.perf_counter() - wall_start
    print(f"  Sims done in {wall_time:.0f}s.  Computing diagnostics...", flush=True)

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
            "temp_index":     gi,
            "temp":           float(temps[gi]),
            "temp_over_lam1": float(temps[gi] / lam1),
            "q_rand":         diag["q_rand"],
            "xi":             diag["xi"],
            "m1_sq":          diag["m1_sq"],
            "m1w_sq":         diag["m1w_sq"],
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

    print(f"  Saved: {run_dir / 'diag_summary.tsv'}", flush=True)
    return results


def main():
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    base.compile_core()
    for d in D_LIST:
        run_diag_case(d)
    print("\nUniform diagnostic runs done.")


if __name__ == "__main__":
    main()

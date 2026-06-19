#!/usr/bin/env python3
"""
Zipf + Uniform d=24 runs at Mohanty (2021) sweep count.
Continuation after geometric finished with 350 sweeps.

Grids same as run_d24_parallel.py:
  Zipf    : T* 0.1..8.0  step 0.1  (80 pts)
  Uniform : T* 0.05..3.0 step 0.05 (60 pts)
"""
import json, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
import run_hamming_sims as base
import run_d24_production as prod
# reuse helpers from run_d24_parallel
from run_d24_parallel import (run_single_temp, merge_workers,
                               make_linear_grid, plot_run_linear, run_case)
# run_d24_parallel.merge_workers now writes absolute paths in snapshots.tsv (bug fixed)

REPO      = Path("/home/gijs/Documents/Thesis-MSc")
SWEEPS    = 100   # Mohanty exact
BURN      = 50
THIN      = 1
N_WORKERS = 12
INIT      = "ordered"

# Monkey-patch the globals that run_case() reads from its module
import run_d24_parallel as _par
_par.SWEEPS = SWEEPS
_par.BURN   = BURN


def main():
    compiler_flags = base.compile_core()
    print(f"\nCompiled: {compiler_flags}", flush=True)
    print(f"Zipf + Uniform at Mohanty sweeps ({SWEEPS}+{BURN})", flush=True)
    results = []

    # ── Zipf ──────────────────────────────────────────────────────────────────
    counts_z = base.counts_zipf(24, 4)
    h_z, _   = base.landau_h_factor(counts_z)
    tl_z     = 22.0 * h_z
    temps_z, _ = make_linear_grid(0.1, 8.0, 0.1, [])
    sidx_z = list(range(len(temps_z)))   # save all 80 temps (~2.5 GB, uint16)
    m = run_case("d24_zipf_ordered", 24, counts_z, seed=4100,
                 temps=temps_z, save_indices=sidx_z,
                 description="Zipf d=24, ordered init, linear grid 0.1-8 δ0.1. Mohanty sweeps. All temps saved.")
    if m: results.append(m)

    # ── Uniform ───────────────────────────────────────────────────────────────
    counts_u = base.counts_uniform(24)
    h_u, _   = base.landau_h_factor(counts_u)
    tl_u     = 22.0 * h_u
    temps_u, _ = make_linear_grid(0.05, 3.0, 0.05, [])
    sidx_u = list(range(len(temps_u)))   # save all 60 temps (~1.9 GB, uint16)
    m = run_case("d24_uniform_ordered", 24, counts_u, seed=3100,
                 temps=temps_u, save_indices=sidx_u,
                 description="Uniform d=24, ordered init, linear grid 0.05-3 δ0.05. Mohanty sweeps. All temps saved.")
    if m: results.append(m)

    report = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
              "root": str(ROOT), "cases": results}
    (ROOT/"run_report_zipf_uniform_mohanty.json").write_text(json.dumps(report, indent=2))

    print("\n" + "="*60 + "\nSUMMARY")
    for r in results:
        print(f"  {r['name']}: C_S peak {r['observed_heat_peak_over_landau']:.2f}x T*_Landau,"
              f" wall {r['wall_time_seconds']/60:.0f} min")


if __name__ == "__main__":
    main()

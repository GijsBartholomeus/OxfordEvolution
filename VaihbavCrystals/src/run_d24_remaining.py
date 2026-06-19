#!/usr/bin/env python3
"""
Run remaining d=24 cases:
  - d24_zipf_classbits4        (fresh run, same window)
  - d24_geometric_25phenotypes  (fresh run, same window)
  - d24_uniform_wide            (new wide logarithmic grid to find the actual transition)

The uniform case had its window entirely in the ordered regime (E at hottest point
was 87x the high-T random expectation). The wide grid spans 0.4 to 320 x T*_Landau
to locate the actual transition.
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
import run_hamming_sims as base
import run_d24_production as prod

REPO = Path("/home/gijs/Documents/Thesis-MSc")
SWEEPS = 60
BURN = 20
THIN = 1


def run_case(name, d, counts, seed, temps, save_indices, description=""):
    run_dir = ROOT / "runs" / name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    counts_path = run_dir / "counts.tsv"
    temps_path = run_dir / "temps.tsv"
    base.write_counts(counts_path, counts)
    base.write_temps(temps_path, temps, save_indices)

    k = 2
    z = d
    lambda_1 = d * (k - 1) - k
    h_f, mu_f = base.landau_h_factor(counts)
    t_landau = lambda_1 * h_f

    metadata = {
        "name": name,
        "description": description,
        "d": d, "k": k,
        "N": int(1 << d), "z": z,
        "counts_summary": base.summarize_counts(counts),
        "counts_file": str(counts_path),
        "frequencies_summary": {
            "min": float(counts.min() / counts.sum()),
            "max": float(counts.max() / counts.sum()),
            "sum_f_squared": float(np.sum((counts / counts.sum()) ** 2)),
        },
        "lambda_1": lambda_1, "h_f": float(h_f), "mu_f": float(mu_f),
        "T_star_Landau": float(t_landau),
        "temperature_grid_hot_to_cold": temps.tolist(),
        "selected_snapshot_indices": save_indices,
        "selected_snapshot_temperatures": temps[save_indices].tolist(),
        "sweeps": SWEEPS, "burn_in": BURN, "thin": THIN,
        "rng_seed": seed, "initialization": "random",
        "compiler_flags": "",
        "git_commit_Thesis_MSc": base.git_commit(REPO),
        "heat_capacity_convention": "C_S(T*) = Var(S) / (N * T*^2), post-burn-in thinned sweeps.",
        "high_temperature_random_E_expectation": base.high_temp_expectation(counts),
        "snapshot_dtype_expected": base.dtype_for_q(len(counts)),
        "notes": (
            "Hot-to-cold annealing, 60 sweeps per temperature, 20 burn-in. "
            "Stationarity checked by first/second post-burn-in half comparison."
        ),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    cmd = [
        str(base.BIN),
        "--d", str(d),
        "--counts", str(counts_path),
        "--temps", str(temps_path),
        "--out", str(run_dir),
        "--sweeps", str(SWEEPS),
        "--burn", str(BURN),
        "--thin", str(THIN),
        "--seed", str(seed),
        "--init", "random",
        "--validate-trials", "0",
    ]
    print(f"\n{'='*60}", flush=True)
    print(f"Starting: {name}", flush=True)
    print(f"  T*_Landau = {t_landau:.6g}", flush=True)
    print(f"  Q = {len(counts)} phenotypes, {SWEEPS} sweeps, {len(temps)} temps", flush=True)
    print(f"  Temp range: {temps[-1]:.4g} (cold) to {temps[0]:.4g} (hot)", flush=True)

    start = time.perf_counter()
    proc = subprocess.run(cmd, text=True, capture_output=True)
    runtime = time.perf_counter() - start
    (run_dir / "stdout.txt").write_text(proc.stdout)
    (run_dir / "stderr.txt").write_text(proc.stderr)
    print(proc.stderr.strip(), flush=True)

    if proc.returncode != 0:
        print(f"FAILED: {name}", file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        return None

    summary = np.genfromtxt(run_dir / "summary.tsv", names=True, delimiter="\t")
    if summary.shape == ():
        summary = np.asarray([summary])
    peak_idx = int(np.argmax(summary["heat_capacity_S"]))

    diagnostics = prod.stationarity_diagnostics(run_dir, BURN)
    (run_dir / "stationarity_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    max_half_diff = max((d["abs_half_difference"] for d in diagnostics), default=0.0)

    metadata.update({
        "runtime_seconds": runtime,
        "counts_preservation_snapshot_failures": base.validate_counts_preserved(run_dir, counts),
        "observed_heat_peak_temperature": float(summary["temp"][peak_idx]),
        "observed_heat_peak_over_landau": float(summary["temp"][peak_idx] / t_landau),
        "stationarity_diagnostics_file": str(run_dir / "stationarity_diagnostics.json"),
        "stationarity_max_half_diff_E": float(max_half_diff),
    })
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    base.plot_run(run_dir, t_landau, save_indices, name.replace("_", " "))

    print(f"  Done in {runtime/60:.1f} min", flush=True)
    print(f"  Peak C_S at T={metadata['observed_heat_peak_temperature']:.4g}"
          f"  ({metadata['observed_heat_peak_over_landau']:.3f} x T*_Landau)", flush=True)
    print(f"  Stationarity max half-diff E = {max_half_diff:.4g}", flush=True)
    return metadata


def make_grid(t_landau, factors, save_factors):
    temps = np.unique(t_landau * np.asarray(factors))[::-1]  # hot to cold
    save_targets = t_landau * np.asarray(save_factors)
    save_indices = sorted({int(np.argmin(np.abs(temps - t))) for t in save_targets})
    return temps, save_indices


def counts_geometric_d24():
    return np.asarray([1 << e for e in range(23, 0, -1)] + [1, 1], dtype=np.uint64)


def main():
    compiler_flags = base.compile_core()
    print(f"Compiled: {compiler_flags}", flush=True)

    results = []

    # --- d24_zipf_classbits4 --------------------------------------------------
    counts_z = base.counts_zipf(24, 4)
    h_z, mu_z = base.landau_h_factor(counts_z)
    t_l_z = 22.0 * h_z
    factors_z  = [2.5, 1.60, 1.30, 1.10, 1.00, 0.90, 0.70, 0.50, 0.40]
    save_z     = [2.5, 1.10, 1.00, 0.90, 0.40]
    temps_z, sidx_z = make_grid(t_l_z, factors_z, save_z)
    m = run_case(
        name="d24_zipf_classbits4",
        d=24, counts=counts_z, seed=4001,
        temps=temps_z, save_indices=sidx_z,
        description="Binary d=24 Zipf-like counts: class_bits=4, Q=65535 phenotypes.",
    )
    if m: results.append(m)

    # --- d24_geometric_25phenotypes -------------------------------------------
    counts_g = counts_geometric_d24()
    h_g, mu_g = base.landau_h_factor(counts_g)
    t_l_g = 22.0 * h_g
    factors_g  = [2.5, 1.60, 1.30, 1.10, 1.00, 0.90, 0.70, 0.50, 0.40]
    save_g     = [2.5, 1.10, 1.00, 0.90, 0.40]
    temps_g, sidx_g = make_grid(t_l_g, factors_g, save_g)
    m = run_case(
        name="d24_geometric_25phenotypes",
        d=24, counts=counts_g, seed=5001,
        temps=temps_g, save_indices=sidx_g,
        description="Binary d=24 geometric counts: [2^23,...,2,1,1], Q=25 phenotypes.",
    )
    if m: results.append(m)

    # --- d24_uniform WIDE GRID ------------------------------------------------
    # Previous run (2.5x window): E_hot=-0.021, random expectation=-0.000244 (87x off)
    # => window entirely in ordered phase. Use wide log grid to find transition.
    counts_u = base.counts_uniform(24)
    h_u, mu_u = base.landau_h_factor(counts_u)
    t_l_u = 22.0 * h_u
    # Span 0.4x to 320x T*_Landau on a log scale — should bracket the real transition
    log_factors = np.logspace(np.log10(0.4), np.log10(320.0), 15)
    save_u = [0.4, 1.0, 2.5, 40.0, 320.0]  # cold, near-Landau, 2.5x, high, very hot
    temps_u, sidx_u = make_grid(t_l_u, log_factors.tolist(), save_u)
    print(f"\nUniform wide grid: {len(temps_u)} temps from"
          f" {temps_u[-1]:.4g} to {temps_u[0]:.4g}", flush=True)
    m = run_case(
        name="d24_uniform_wide",
        d=24, counts=counts_u, seed=3002,
        temps=temps_u, save_indices=sidx_u,
        description=(
            "d=24 uniform, wide log grid (0.4–320 x T*_Landau) to locate transition. "
            "Narrow-window run showed window entirely in ordered regime."
        ),
    )
    if m: results.append(m)

    # --- summary --------------------------------------------------------------
    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "root": str(ROOT),
        "cases": results,
    }
    (ROOT / "run_report_remaining.json").write_text(json.dumps(report, indent=2))
    print("\n" + "="*60 + "\nSUMMARY")
    for r in results:
        print(f"\n  {r['name']}")
        print(f"    T*_Landau = {r['T_star_Landau']:.5g}")
        print(f"    heat peak = {r['observed_heat_peak_temperature']:.4g}"
              f"  ({r['observed_heat_peak_over_landau']:.3f} x T*_Landau)")
        print(f"    runtime   = {r['runtime_seconds']/60:.1f} min")


if __name__ == "__main__":
    main()

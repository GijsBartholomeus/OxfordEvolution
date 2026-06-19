#!/usr/bin/env python3
"""
Re-run zipf and uniform with 100 sweeps / 30 burn-in, 9 temperature points each.
Zipf: hot end extended to 10x T*_Landau to reach the disordered phase.
Uniform: 9 points focused on the interesting region (0.002 to 1.72).
Target wall-clock: ~5 hours total.
"""
import json, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
import run_hamming_sims as base
import run_d24_production as prod

REPO = Path("/home/gijs/Documents/Thesis-MSc")
SWEEPS = 100
BURN   = 30
THIN   = 1


def run_case(name, d, counts, seed, temps, save_indices, description=""):
    run_dir = ROOT / "runs" / name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    counts_path = run_dir / "counts.tsv"
    temps_path  = run_dir / "temps.tsv"
    base.write_counts(counts_path, counts)
    base.write_temps(temps_path, temps, save_indices)

    k = 2; z = d; lambda_1 = d*(k-1)-k
    h_f, mu_f = base.landau_h_factor(counts)
    t_landau  = lambda_1 * h_f

    metadata = {
        "name": name, "description": description,
        "d": d, "k": k, "N": int(1<<d), "z": z,
        "counts_summary": base.summarize_counts(counts),
        "counts_file": str(counts_path),
        "frequencies_summary": {
            "min": float(counts.min()/counts.sum()),
            "max": float(counts.max()/counts.sum()),
            "sum_f_squared": float(np.sum((counts/counts.sum())**2)),
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
        "heat_capacity_convention": "C_S(T*) = Var(S)/(N*T*^2), post-burn-in thinned sweeps.",
        "high_temperature_random_E_expectation": base.high_temp_expectation(counts),
        "snapshot_dtype_expected": base.dtype_for_q(len(counts)),
        "notes": f"Hot-to-cold, {SWEEPS} sweeps, {BURN} burn-in. ~5-hr target run.",
    }
    (run_dir/"metadata.json").write_text(json.dumps(metadata, indent=2))

    cmd = [str(base.BIN),
           "--d", str(d), "--counts", str(counts_path),
           "--temps", str(temps_path), "--out", str(run_dir),
           "--sweeps", str(SWEEPS), "--burn", str(BURN), "--thin", str(THIN),
           "--seed", str(seed), "--init", "random", "--validate-trials", "0"]

    print(f"\n{'='*60}\nStarting: {name}", flush=True)
    print(f"  T*_Landau={t_landau:.4g}  Q={len(counts)}  "
          f"{SWEEPS} sweeps  {len(temps)} temps", flush=True)
    print(f"  {temps[-1]:.4g} (cold) → {temps[0]:.4g} (hot)", flush=True)

    start = time.perf_counter()
    proc  = subprocess.run(cmd, text=True, capture_output=True)
    runtime = time.perf_counter() - start
    (run_dir/"stdout.txt").write_text(proc.stdout)
    (run_dir/"stderr.txt").write_text(proc.stderr)
    print(proc.stderr.strip(), flush=True)

    if proc.returncode != 0:
        print(f"FAILED: {proc.stderr}", file=sys.stderr)
        return None

    summary = np.genfromtxt(run_dir/"summary.tsv", names=True, delimiter="\t")
    if summary.shape == (): summary = np.asarray([summary])
    peak_idx   = int(np.argmax(summary["heat_capacity_S"]))
    diagnostics = prod.stationarity_diagnostics(run_dir, BURN)
    (run_dir/"stationarity_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    max_hd = max((d["abs_half_difference"] for d in diagnostics), default=0.0)

    metadata.update({
        "runtime_seconds": runtime,
        "counts_preservation_snapshot_failures": base.validate_counts_preserved(run_dir, counts),
        "observed_heat_peak_temperature": float(summary["temp"][peak_idx]),
        "observed_heat_peak_over_landau":  float(summary["temp"][peak_idx]/t_landau),
        "stationarity_max_half_diff_E": float(max_hd),
    })
    (run_dir/"metadata.json").write_text(json.dumps(metadata, indent=2))
    base.plot_run(run_dir, t_landau, save_indices, name.replace("_", " "))

    print(f"  Done in {runtime/60:.1f} min", flush=True)
    print(f"  C_S peak T={metadata['observed_heat_peak_temperature']:.4g}"
          f" ({metadata['observed_heat_peak_over_landau']:.2f}x T*_Landau)", flush=True)
    print(f"  stationarity max half-diff = {max_hd:.4g}", flush=True)
    return metadata


def make_grid(t_landau, factors, save_factors):
    temps = np.unique(t_landau * np.asarray(factors))[::-1]
    save_targets = t_landau * np.asarray(save_factors)
    save_indices = sorted({int(np.argmin(np.abs(temps-t))) for t in save_targets})
    return temps, save_indices


def main():
    compiler_flags = base.compile_core()
    print(f"Compiled: {compiler_flags}", flush=True)
    results = []

    # --- Zipf: extend to 10x T*_Landau, 9 temps ----------------------------
    counts_z = base.counts_zipf(24, 4)
    h_z, _   = base.landau_h_factor(counts_z)
    t_l_z    = 22.0 * h_z
    # 9 points hot→cold, now reaching 10x to enter disordered phase
    factors_z = [10.0, 5.0, 2.5, 1.6, 1.1, 1.0, 0.9, 0.6, 0.4]
    save_z    = [10.0, 1.1, 1.0, 0.9, 0.4]
    temps_z, sidx_z = make_grid(t_l_z, factors_z, save_z)
    m = run_case("d24_zipf_classbits4", 24, counts_z, 4002,
                 temps_z, sidx_z,
                 "Zipf d=24, extended hot end to 10x T*_Landau, 100 sweeps.")
    if m: results.append(m)

    # --- Uniform: 9 points across the wide region ---------------------------
    counts_u = base.counts_uniform(24)
    h_u, _   = base.landau_h_factor(counts_u)
    t_l_u    = 22.0 * h_u
    # 9 log-spaced points from 0.4x to 320x — covers full sigmoid seen in wide run
    log_factors = np.logspace(np.log10(0.4), np.log10(320.0), 9)
    save_u      = [0.4, 1.0, 10.0, 100.0, 320.0]
    temps_u, sidx_u = make_grid(t_l_u, log_factors.tolist(), save_u)
    m = run_case("d24_uniform_wide", 24, counts_u, 3003,
                 temps_u, sidx_u,
                 "Uniform d=24 wide grid, 9 log-spaced temps 0.4-320x T*_Landau, 100 sweeps.")
    if m: results.append(m)

    report = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
              "root": str(ROOT), "cases": results}
    (ROOT/"run_report_more_sweeps.json").write_text(json.dumps(report, indent=2))
    print("\n" + "="*60 + "\nSUMMARY")
    for r in results:
        print(f"  {r['name']}: peak at {r['observed_heat_peak_over_landau']:.2f}x T*_Landau,"
              f" {r['runtime_seconds']/60:.0f} min")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Production d=24 runs for uniform, Zipf (classbits4), and geometric distributions.
Uses hot-to-cold annealing, 60 sweeps per temperature, 20 sweep burn-in.
Produces energy/heat-capacity plots with Landau T* marker and five snapshot temps.
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

import run_hamming_sims as base

ROOT = Path(__file__).resolve().parents[1]
REPO = Path("/home/gijs/Documents/Thesis-MSc")

SWEEPS = 60
BURN = 20
THIN = 1


def counts_geometric_d24():
    return np.asarray([1 << e for e in range(23, 0, -1)] + [1, 1], dtype=np.uint64)


def stationarity_diagnostics(run_dir, burn):
    ts_path = run_dir / "timeseries.tsv"
    if not ts_path.exists():
        return []
    ts = np.genfromtxt(ts_path, names=True, delimiter="\t")
    if ts.shape == ():
        ts = np.asarray([ts])
    out = []
    for idx in sorted(set(ts["temp_index"].astype(int))):
        block = ts[ts["temp_index"].astype(int) == idx]
        post = block[block["sweep"] > burn]
        if len(post) < 4:
            continue
        half = len(post) // 2
        first_half = post[:half]
        second_half = post[half:]
        mean1 = float(np.mean(first_half["E"]))
        mean2 = float(np.mean(second_half["E"]))
        sd = float(np.std(post["E"], ddof=1)) if len(post) > 1 else 0.0
        out.append({
            "temp_index": int(idx),
            "temp": float(post["temp"][0]),
            "post_burn_samples": int(len(post)),
            "mean_E_first_half": mean1,
            "mean_E_second_half": mean2,
            "abs_half_difference": abs(mean2 - mean1),
            "post_burn_sd_E": sd,
            "half_difference_in_sd_units": abs(mean2 - mean1) / sd if sd > 0 else None,
        })
    return out


def make_temp_grid(t_landau, factors, save_factors):
    temps = np.unique(t_landau * np.asarray(factors))[::-1]  # hot to cold
    save_targets = t_landau * np.asarray(save_factors)
    save_indices = sorted({int(np.argmin(np.abs(temps - t))) for t in save_targets})
    return temps, save_indices


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
        "d": d,
        "k": k,
        "N": int(1 << d),
        "z": z,
        "counts_summary": base.summarize_counts(counts),
        "counts_file": str(counts_path),
        "frequencies_summary": {
            "min": float(counts.min() / counts.sum()),
            "max": float(counts.max() / counts.sum()),
            "sum_f_squared": float(np.sum((counts / counts.sum()) ** 2)),
        },
        "lambda_1": lambda_1,
        "h_f": float(h_f),
        "mu_f": float(mu_f),
        "T_star_Landau": float(t_landau),
        "temperature_grid_hot_to_cold": temps.tolist(),
        "selected_snapshot_indices": save_indices,
        "selected_snapshot_temperatures": temps[save_indices].tolist(),
        "sweeps": SWEEPS,
        "burn_in": BURN,
        "thin": THIN,
        "rng_seed": seed,
        "initialization": "random",
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
    proc = __import__("subprocess").run(cmd, text=True, capture_output=True)
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

    diagnostics = stationarity_diagnostics(run_dir, BURN)
    (run_dir / "stationarity_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))

    max_half_diff = max((d["abs_half_difference"] for d in diagnostics), default=0.0)

    metadata.update({
        "compiler_flags": "",
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
    print(f"  Peak heat capacity at T* = {metadata['observed_heat_peak_temperature']:.4g}"
          f"  ({metadata['observed_heat_peak_over_landau']:.3f} x T*_Landau)", flush=True)
    print(f"  Stationarity max half-diff E = {max_half_diff:.4g}", flush=True)
    return metadata


def main():
    compiler_flags = base.compile_core()
    print(f"Compiled with: {compiler_flags}", flush=True)

    # ---- case 1: d=24 uniform -----------------------------------------------
    counts_u = base.counts_uniform(24)
    _, mu_u = base.landau_h_factor(counts_u)
    t_l_u = 22.0 / mu_u
    # factors relative to T*_Landau, hot to cold
    factors_u = [2.5, 1.60, 1.30, 1.10, 1.00, 0.90, 0.70, 0.50, 0.40]
    save_u    = [2.5, 1.10, 1.00, 0.90, 0.40]
    temps_u, sidx_u = make_temp_grid(t_l_u, factors_u, save_u)
    meta_u = run_case(
        name="d24_uniform",
        d=24,
        counts=counts_u,
        seed=3001,
        temps=temps_u,
        save_indices=sidx_u,
        description="Binary d=24 uniform phenotype counts: Q=4096, each size 4096.",
    )

    # ---- case 2: d=24 Zipf classbits=4 ---------------------------------------
    counts_z = base.counts_zipf(24, 4)
    _, mu_z = base.landau_h_factor(counts_z)
    t_l_z = 22.0 / mu_z
    factors_z = [2.5, 1.60, 1.30, 1.10, 1.00, 0.90, 0.70, 0.50, 0.40]
    save_z    = [2.5, 1.10, 1.00, 0.90, 0.40]
    temps_z, sidx_z = make_temp_grid(t_l_z, factors_z, save_z)
    meta_z = run_case(
        name="d24_zipf_classbits4",
        d=24,
        counts=counts_z,
        seed=4001,
        temps=temps_z,
        save_indices=sidx_z,
        description="Binary d=24 Zipf-like counts: class_bits=4, Q=65535 phenotypes.",
    )

    # ---- case 3: d=24 geometric (25 phenotypes) ------------------------------
    counts_g = counts_geometric_d24()
    _, mu_g = base.landau_h_factor(counts_g)
    t_l_g = 22.0 / mu_g
    factors_g = [2.5, 1.60, 1.30, 1.10, 1.00, 0.90, 0.70, 0.50, 0.40]
    save_g    = [2.5, 1.10, 1.00, 0.90, 0.40]
    temps_g, sidx_g = make_temp_grid(t_l_g, factors_g, save_g)
    meta_g = run_case(
        name="d24_geometric_25phenotypes",
        d=24,
        counts=counts_g,
        seed=5001,
        temps=temps_g,
        save_indices=sidx_g,
        description="Binary d=24 geometric counts: [2^23, 2^22, ..., 2, 1, 1], Q=25 phenotypes.",
    )

    # ---- summary report ------------------------------------------------------
    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "root": str(ROOT),
        "cases": [m for m in [meta_u, meta_z, meta_g] if m is not None],
    }
    (ROOT / "run_report.json").write_text(json.dumps(report, indent=2))
    print("\n" + "="*60)
    print("PRODUCTION RUN SUMMARY")
    print("="*60)
    for m in report["cases"]:
        print(f"\n  {m['name']}")
        print(f"    T*_Landau = {m['T_star_Landau']:.6g}")
        print(f"    heat peak = {m['observed_heat_peak_temperature']:.4g}"
              f"  ({m['observed_heat_peak_over_landau']:.3f} x T*_Landau)")
        print(f"    runtime   = {m['runtime_seconds']/60:.1f} min")
        print(f"    snapshot failures: {m['counts_preservation_snapshot_failures']}")
    print()


if __name__ == "__main__":
    main()

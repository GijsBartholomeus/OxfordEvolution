#!/usr/bin/env python3
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import run_hamming_sims as base


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "d24_geometric_25phenotypes"


def counts_geometric_d24():
    return np.asarray([1 << e for e in range(23, 0, -1)] + [1, 1], dtype=np.uint64)


def stationarity_diagnostics(run_dir, burn):
    ts = np.genfromtxt(run_dir / "timeseries.tsv", names=True, delimiter="\t")
    if ts.shape == ():
        ts = np.asarray([ts])
    out = []
    for idx in sorted(set(ts["temp_index"].astype(int))):
        block = ts[ts["temp_index"].astype(int) == idx]
        post = block[block["sweep"] > burn]
        if len(post) < 4:
            continue
        half = len(post) // 2
        first = post[:half]
        second = post[half:]
        mean1 = float(np.mean(first["E"]))
        mean2 = float(np.mean(second["E"]))
        sd = float(np.std(post["E"], ddof=1)) if len(post) > 1 else 0.0
        out.append({
            "temp_index": idx,
            "temp": float(post["temp"][0]),
            "post_burn_samples": int(len(post)),
            "mean_E_first_half": mean1,
            "mean_E_second_half": mean2,
            "abs_half_difference": abs(mean2 - mean1),
            "post_burn_sd_E": sd,
            "half_difference_in_sd_units": abs(mean2 - mean1) / sd if sd > 0 else None,
        })
    return out


def main():
    repo = Path("/home/gijs/Documents/Thesis-MSc")
    compiler_flags = base.compile_core()
    counts = counts_geometric_d24()
    d = 24
    k = 2
    lambda_1 = d * (k - 1) - k
    h_f, mu_f = base.landau_h_factor(counts)
    t_landau = lambda_1 * h_f

    if RUN_DIR.exists():
        import shutil
        shutil.rmtree(RUN_DIR)
    RUN_DIR.mkdir(parents=True)

    temps = t_landau * np.asarray([2.5, 1.6, 1.3, 1.1, 1.0, 0.9, 0.7, 0.5])
    save_targets = t_landau * np.asarray([2.5, 1.1, 1.0, 0.9, 0.5])
    save_indices = sorted({int(np.argmin(np.abs(temps - t))) for t in save_targets})
    counts_path = RUN_DIR / "counts.tsv"
    temps_path = RUN_DIR / "temps.tsv"
    base.write_counts(counts_path, counts)
    base.write_temps(temps_path, temps, save_indices)

    sweeps = 60
    burn = 20
    thin = 1
    seed = 5001
    metadata = {
        "name": "d24_geometric_25phenotypes",
        "description": "Phenotype counts [2^23, 2^22, ..., 2, 1, 1].",
        "d": d,
        "k": k,
        "N": int(1 << d),
        "z": d,
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
        "sweeps": sweeps,
        "burn_in": burn,
        "thin": thin,
        "rng_seed": seed,
        "initialization": "random",
        "compiler_flags": compiler_flags,
        "git_commit_Thesis_MSc": base.git_commit(repo),
        "heat_capacity_convention": "C_S(T*) = Var(S) / (N * T*^2), using post-burn-in thinned sweeps.",
        "high_temperature_random_E_expectation": base.high_temp_expectation(counts),
        "snapshot_dtype_expected": base.dtype_for_q(len(counts)),
        "notes": "Hot-to-cold annealing, 60 sweeps per temperature. Stationarity is checked by comparing first and second post-burn-in halves.",
    }
    (RUN_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    cmd = [
        str(base.BIN),
        "--d", str(d),
        "--counts", str(counts_path),
        "--temps", str(temps_path),
        "--out", str(RUN_DIR),
        "--sweeps", str(sweeps),
        "--burn", str(burn),
        "--thin", str(thin),
        "--seed", str(seed),
        "--init", "random",
        "--validate-trials", "0",
    ]
    start = time.perf_counter()
    proc = subprocess.run(cmd, text=True, capture_output=True)
    runtime = time.perf_counter() - start
    (RUN_DIR / "stdout.txt").write_text(proc.stdout)
    (RUN_DIR / "stderr.txt").write_text(proc.stderr)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(proc.returncode)

    summary = np.genfromtxt(RUN_DIR / "summary.tsv", names=True, delimiter="\t")
    peak_idx = int(np.argmax(summary["heat_capacity_S"]))
    diagnostics = stationarity_diagnostics(RUN_DIR, burn)
    (RUN_DIR / "stationarity_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    metadata.update({
        "runtime_seconds": runtime,
        "counts_preservation_snapshot_failures": base.validate_counts_preserved(RUN_DIR, counts),
        "observed_heat_peak_temperature": float(summary["temp"][peak_idx]),
        "observed_heat_peak_over_landau": float(summary["temp"][peak_idx] / t_landau),
        "stationarity_diagnostics_file": str(RUN_DIR / "stationarity_diagnostics.json"),
    })
    (RUN_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    base.plot_run(RUN_DIR, t_landau, save_indices, "d24 geometric 25 phenotypes")
    print(json.dumps({
        "run_dir": str(RUN_DIR),
        "runtime_seconds": runtime,
        "T_star_Landau": float(t_landau),
        "observed_heat_peak_temperature": metadata["observed_heat_peak_temperature"],
        "snapshot_failures": metadata["counts_preservation_snapshot_failures"],
        "stationarity_max_half_diff_E": max(d["abs_half_difference"] for d in diagnostics),
    }, indent=2))


if __name__ == "__main__":
    main()

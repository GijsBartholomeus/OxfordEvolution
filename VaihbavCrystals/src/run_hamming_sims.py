#!/usr/bin/env python3
import argparse
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "hamming_mcmc.c"
BIN = ROOT / "src" / "hamming_mcmc"
RUNS = ROOT / "runs"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def git_commit(repo):
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return None


def compile_core():
    flags = ["-O3", "-std=c11", "-march=native", "-Wall", "-Wextra"]
    cmd = ["cc", *flags, str(SRC), "-lm", "-o", str(BIN)]
    subprocess.run(cmd, check=True)
    return " ".join(flags)


def counts_uniform(d):
    q = 1 << (d // 2)
    return np.full(q, 1 << (d // 2), dtype=np.uint64)


def counts_zipf(d, class_bits):
    vals = []
    for r in range(1 << class_bits):
        vals.extend([1 << (d - class_bits - r)] * (1 << r))
    return np.asarray(vals, dtype=np.uint64)


def landau_h_factor(counts):
    counts = counts.astype(np.float64)
    n = counts.sum()
    f = counts / n
    diag = 1.0 / f
    diag.sort()
    if len(diag) == 1:
        return 0.0, math.inf
    if diag[0] == diag[1]:
        mu = diag[0]
    else:
        lo = np.nextafter(diag[0], diag[1])
        hi = np.nextafter(diag[1], diag[0])

        def secular(x):
            return np.sum(1.0 / (diag - x))

        for _ in range(120):
            mid = 0.5 * (lo + hi)
            if secular(mid) > 0:
                lo = mid
            else:
                hi = mid
        mu = 0.5 * (lo + hi)
    return 1.0 / mu, mu


def temp_grid(t_landau, mode, d):
    if d == 24:
        factors = np.array([0.40, 0.70, 0.90, 1.00, 1.10, 1.40, 2.50])
    else:
        factors = np.array([0.35, 0.50, 0.65, 0.80, 0.90, 0.97, 1.00,
                            1.03, 1.10, 1.25, 1.50, 2.00, 3.00])
    temps = np.unique(np.asarray(t_landau * factors, dtype=float))
    save_targets = np.asarray(t_landau * np.array([0.40, 0.90, 1.00, 1.10, 2.50]))
    save_indices = sorted({int(np.argmin(np.abs(temps - target))) for target in save_targets})
    return temps, save_indices


def write_counts(path, counts):
    np.savetxt(path, counts, fmt="%d")


def write_temps(path, temps, save_indices):
    with path.open("w") as f:
        for i, t in enumerate(temps):
            f.write(f"{t:.17g}\t{1 if i in save_indices else 0}\n")


def high_temp_expectation(counts):
    f = counts.astype(np.float64) / float(counts.sum())
    return -float(np.sum(f * f))


def dtype_for_q(q):
    if q <= 256:
        return "uint8"
    if q <= 65536:
        return "uint16"
    return "uint32"


def summarize_counts(counts):
    return {
        "Q": int(len(counts)),
        "total": int(counts.sum()),
        "min": int(counts.min()),
        "max": int(counts.max()),
        "unique_sizes": int(len(np.unique(counts))),
    }


def plot_run(run_dir, t_landau, save_indices, title):
    summary = np.genfromtxt(run_dir / "summary.tsv", names=True, delimiter="\t")
    if summary.shape == ():
        summary = np.asarray([summary])
    temps = summary["temp"]
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.0), sharex=True)
    axes[0].plot(temps, summary["mean_E"], marker="o", lw=1.4)
    axes[0].axvline(t_landau, color="black", ls="--", lw=1.0, label="Landau")
    for idx in save_indices:
        axes[0].axvline(temps[idx], color="tab:gray", alpha=0.25, lw=0.9)
    axes[0].set_ylabel("mean E = -2S/(Nz)")
    axes[0].legend(frameon=False)
    axes[0].set_title(title)

    axes[1].plot(temps, summary["heat_capacity_S"], marker="o", lw=1.4, color="tab:red")
    axes[1].axvline(t_landau, color="black", ls="--", lw=1.0)
    for idx in save_indices:
        axes[1].axvline(temps[idx], color="tab:gray", alpha=0.25, lw=0.9)
    axes[1].set_ylabel("C_S = Var(S)/(N T*^2)")
    axes[1].set_xlabel("T*")
    axes[1].set_xscale("log")
    fig.tight_layout()
    fig.savefig(run_dir / "energy_heat.png", dpi=180)
    fig.savefig(run_dir / "energy_heat.pdf")
    plt.close(fig)


def validate_counts_preserved(run_dir, counts):
    manifest = run_dir / "snapshots.tsv"
    if not manifest.exists():
        return []
    problems = []
    with manifest.open() as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            path, dtype = Path(parts[2]), parts[3]
            arr = np.fromfile(path, dtype=np.dtype(dtype))
            got = np.bincount(arr.astype(np.int64), minlength=len(counts))
            if not np.array_equal(got.astype(np.uint64), counts):
                problems.append(str(path))
    return problems


def run_case(name, d, counts, sweeps, burn, thin, seed, init, compiler_flags, validate_trials, repo):
    run_dir = RUNS / name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    counts_path = run_dir / "counts.tsv"
    temps_path = run_dir / "temps.tsv"
    write_counts(counts_path, counts)

    k = 2
    z = d
    lambda_1 = d * (k - 1) - k
    h_f, mu_f = landau_h_factor(counts)
    t_landau = lambda_1 * h_f
    temps, save_indices = temp_grid(t_landau, name, d)
    write_temps(temps_path, temps, save_indices)

    metadata = {
        "name": name,
        "d": d,
        "k": k,
        "N": int(1 << d),
        "z": z,
        "counts_summary": summarize_counts(counts),
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
        "temperature_grid": temps.tolist(),
        "selected_snapshot_indices": save_indices,
        "selected_snapshot_temperatures": temps[save_indices].tolist(),
        "sweeps": sweeps,
        "burn_in": burn,
        "thin": thin,
        "rng_seed": seed,
        "initialization": init,
        "compiler_flags": compiler_flags,
        "git_commit_Thesis_MSc": git_commit(repo),
        "heat_capacity_convention": "C_S(T*) = Var(S) / (N * T*^2), using post-burn-in thinned sweeps.",
        "high_temperature_random_E_expectation": high_temp_expectation(counts),
        "snapshot_dtype_expected": dtype_for_q(len(counts)),
        "notes": "d=24 runs are short production pilots; do not treat stationarity as established.",
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    cmd = [
        str(BIN),
        "--d", str(d),
        "--counts", str(counts_path),
        "--temps", str(temps_path),
        "--out", str(run_dir),
        "--sweeps", str(sweeps),
        "--burn", str(burn),
        "--thin", str(thin),
        "--seed", str(seed),
        "--init", init,
        "--validate-trials", str(validate_trials),
    ]
    start = time.perf_counter()
    proc = subprocess.run(cmd, text=True, capture_output=True)
    runtime = time.perf_counter() - start
    (run_dir / "stdout.txt").write_text(proc.stdout)
    (run_dir / "stderr.txt").write_text(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed with code {proc.returncode}: {proc.stderr}")

    metadata["runtime_seconds"] = runtime
    metadata["counts_preservation_snapshot_failures"] = validate_counts_preserved(run_dir, counts)
    summary = np.genfromtxt(run_dir / "summary.tsv", names=True, delimiter="\t")
    if summary.shape == ():
        summary = np.asarray([summary])
    peak_idx = int(np.argmax(summary["heat_capacity_S"]))
    metadata["observed_heat_peak_temperature"] = float(summary["temp"][peak_idx])
    metadata["observed_heat_peak_over_landau"] = float(summary["temp"][peak_idx] / t_landau)
    metadata["final_metadata_written_after_run"] = True
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    plot_run(run_dir, t_landau, save_indices, name)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use shorter d=12 runs and minimal d=24 pilots.")
    parser.add_argument("--skip-big", action="store_true")
    parser.add_argument("--repo", default="/home/gijs/Documents/Thesis-MSc")
    args = parser.parse_args()

    RUNS.mkdir(exist_ok=True)
    compiler_flags = compile_core()

    if args.quick:
        small_sweeps, small_burn = 800, 200
        big_sweeps, big_burn = 1, 0
    else:
        small_sweeps, small_burn = 2500, 600
        big_sweeps, big_burn = 2, 0

    cases = [
        ("d12_uniform", 12, counts_uniform(12), small_sweeps, small_burn, 1, 1001, 1000),
        ("d12_zipf_classbits3", 12, counts_zipf(12, 3), small_sweeps, small_burn, 1, 2001, 1000),
    ]
    if not args.skip_big:
        cases.extend([
            ("d24_uniform", 24, counts_uniform(24), big_sweeps, big_burn, 1, 3001, 0),
            ("d24_zipf_classbits4", 24, counts_zipf(24, 4), big_sweeps, big_burn, 1, 4001, 0),
        ])

    all_meta = []
    for name, d, counts, sweeps, burn, thin, seed, validate_trials in cases:
        all_meta.append(run_case(
            name=name,
            d=d,
            counts=counts,
            sweeps=sweeps,
            burn=burn,
            thin=thin,
            seed=seed,
            init="random",
            compiler_flags=compiler_flags,
            validate_trials=validate_trials,
            repo=Path(args.repo),
        ))

    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "root": str(ROOT),
        "cases": all_meta,
    }
    (ROOT / "run_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "root": str(ROOT),
        "cases": [
            {
                "name": m["name"],
                "runtime_seconds": m["runtime_seconds"],
                "T_star_Landau": m["T_star_Landau"],
                "observed_heat_peak_temperature": m["observed_heat_peak_temperature"],
                "snapshot_failures": m["counts_preservation_snapshot_failures"],
            }
            for m in all_meta
        ],
    }, indent=2))


if __name__ == "__main__":
    main()

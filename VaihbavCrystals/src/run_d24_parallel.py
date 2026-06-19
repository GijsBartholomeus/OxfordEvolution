#!/usr/bin/env python3
"""
Parallel d=24 runs matching Mohanty (2021) setup.

Each temperature is run as an independent subprocess (--init ordered),
enabling 12-way parallelism on the local CPU. ~9-10 hours for all three cases.

Grids (linear, Mohanty density δT*=0.1):
  Geometric : T* 0.1..15.0  step 0.1   (150 pts) -- Mohanty's exact distribution
  Zipf      : T* 0.1..8.0   step 0.1   ( 80 pts) -- transition ~1.1-1.7
  Uniform   : T* 0.05..3.0  step 0.05  ( 60 pts) -- transition ~1.3 (T*_c_MF)
"""
import json, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
import run_hamming_sims as base
import run_d24_production as prod

REPO      = Path("/home/gijs/Documents/Thesis-MSc")
SWEEPS    = 350
BURN      = 50
THIN      = 1
N_WORKERS = 12   # logical CPUs to use in parallel
INIT      = "ordered"


def run_single_temp(global_idx, temp, save_flag, base_dir, counts_path,
                    d, sweeps, burn, thin, seed):
    """Run the C binary for one temperature; return (global_idx, worker_dir, runtime, rc)."""
    worker_dir = base_dir / f"worker_{global_idx:04d}"
    worker_dir.mkdir(parents=True, exist_ok=True)

    temps_path = worker_dir / "temps.tsv"
    with temps_path.open("w") as f:
        f.write(f"{temp:.17g}\t{int(save_flag)}\n")

    cmd = [str(base.BIN),
           "--d", str(d), "--counts", str(counts_path),
           "--temps", str(temps_path), "--out", str(worker_dir),
           "--sweeps", str(sweeps), "--burn", str(burn), "--thin", str(thin),
           "--seed", str(seed + global_idx), "--init", INIT,
           "--validate-trials", "0"]

    start = time.perf_counter()
    proc  = subprocess.run(cmd, text=True, capture_output=True)
    rt    = time.perf_counter() - start

    (worker_dir / "stdout.txt").write_text(proc.stdout)
    (worker_dir / "stderr.txt").write_text(proc.stderr)
    return global_idx, worker_dir, rt, proc.returncode


def merge_workers(run_dir, temps, save_indices, n_temps):
    """Merge per-temperature worker outputs into run_dir summary/timeseries/snapshots."""
    summary_rows  = []
    ts_rows       = []
    snap_rows     = []
    snap_header   = None
    omega_counter = 0

    for gi in range(n_temps):
        wdir = run_dir / f"worker_{gi:04d}"

        # --- summary ---
        s = np.genfromtxt(wdir / "summary.tsv", names=True, delimiter="\t")
        if s.shape == ():
            s = np.asarray([s])
        row = {n: float(s[n][0]) for n in s.dtype.names}
        row["temp_index"] = gi
        row["temp"]       = temps[gi]
        summary_rows.append(row)

        # --- timeseries ---
        ts = np.genfromtxt(wdir / "timeseries.tsv", names=True, delimiter="\t")
        if ts.shape == ():
            ts = np.asarray([ts])
        for r in ts:
            ts_rows.append({n: r[n] for n in ts.dtype.names} | {"temp_index": gi})

        # --- snapshots (omega files) ---
        snap_path = wdir / "snapshots.tsv"
        if snap_path.exists():
            lines = snap_path.read_text().strip().split("\n")
            if snap_header is None:
                snap_header = lines[0]
            for line in lines[1:]:
                parts = line.split("\t")
                src = wdir / parts[2]
                if src.exists():
                    dst_name = f"omega_T{omega_counter:03d}.bin"
                    shutil.copy(src, run_dir / dst_name)
                    parts[0] = str(gi)
                    parts[2] = str((run_dir / dst_name).resolve())
                    snap_rows.append("\t".join(parts))
                    omega_counter += 1

    # Write merged summary.tsv
    if summary_rows:
        cols = list(summary_rows[0].keys())
        with (run_dir / "summary.tsv").open("w") as f:
            f.write("\t".join(cols) + "\n")
            for r in summary_rows:
                f.write("\t".join(str(r[c]) for c in cols) + "\n")

    # Write merged timeseries.tsv
    if ts_rows:
        cols = list(ts_rows[0].keys())
        with (run_dir / "timeseries.tsv").open("w") as f:
            f.write("\t".join(cols) + "\n")
            for r in ts_rows:
                f.write("\t".join(str(r[c]) for c in cols) + "\n")

    # Write merged snapshots.tsv
    if snap_rows and snap_header:
        with (run_dir / "snapshots.tsv").open("w") as f:
            f.write(snap_header + "\n")
            for r in snap_rows:
                f.write(r + "\n")


def run_case(name, d, counts, seed, temps, save_indices, description=""):
    run_dir = ROOT / "runs" / name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    counts_path = run_dir / "counts.tsv"
    base.write_counts(counts_path, counts)

    k = 2; z = d; lambda_1 = d*(k-1)-k
    h_f, mu_f   = base.landau_h_factor(counts)
    t_landau    = lambda_1 * h_f
    n_temps     = len(temps)
    save_set    = set(save_indices)

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
        "temperature_grid": temps.tolist(),
        "selected_snapshot_indices": save_indices,
        "selected_snapshot_temperatures": temps[save_indices].tolist(),
        "sweeps": SWEEPS, "burn_in": BURN, "thin": THIN,
        "rng_seed_base": seed, "initialization": INIT,
        "n_parallel_workers": N_WORKERS,
        "git_commit_Thesis_MSc": base.git_commit(REPO),
        "heat_capacity_convention": "C_S(T*) = Var(S)/(N*T*^2), post-burn-in sweeps.",
        "high_temperature_random_E_expectation": base.high_temp_expectation(counts),
        "snapshot_dtype_expected": base.dtype_for_q(len(counts)),
        "notes": (f"Ordered init, {SWEEPS} sweeps, {BURN} burn-in. "
                  f"Each T* independent, parallelised across {N_WORKERS} workers."),
    }
    (run_dir/"metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"\n{'='*60}\n{name}", flush=True)
    print(f"  T*_Landau={t_landau:.4g}  Q={len(counts)}  "
          f"{n_temps} temps  {SWEEPS}+{BURN} sweeps  {N_WORKERS} workers", flush=True)
    print(f"  T* range: {temps[-1]:.4g} .. {temps[0]:.4g}", flush=True)

    wall_start  = time.perf_counter()
    runtimes    = {}
    failed      = []

    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {
            pool.submit(run_single_temp,
                        gi, float(temps[gi]), gi in save_set,
                        run_dir, counts_path,
                        d, SWEEPS, BURN, THIN, seed): gi
            for gi in range(n_temps)
        }
        done = 0
        for fut in as_completed(futures):
            gi, wdir, rt, rc = fut.result()
            runtimes[gi] = rt
            done += 1
            status = "ok" if rc == 0 else f"FAIL(rc={rc})"
            print(f"  [{done:3d}/{n_temps}] T*={temps[gi]:.3f}  {rt:.0f}s  {status}",
                  flush=True)
            if rc != 0:
                failed.append(gi)

    wall_time = time.perf_counter() - wall_start
    if failed:
        print(f"  WARNING: {len(failed)} temps failed: {failed}", flush=True)

    print(f"  Merging {n_temps} workers...", flush=True)
    merge_workers(run_dir, temps, save_indices, n_temps)

    summary = np.genfromtxt(run_dir/"summary.tsv", names=True, delimiter="\t")
    if summary.shape == ():
        summary = np.asarray([summary])

    diagnostics = prod.stationarity_diagnostics(run_dir, BURN)
    (run_dir/"stationarity_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    max_hd = max((d["abs_half_difference"] for d in diagnostics), default=0.0)

    peak_idx = int(np.argmax(summary["heat_capacity_S"]))
    metadata.update({
        "wall_time_seconds":  wall_time,
        "per_temp_runtimes":  {str(k): v for k, v in runtimes.items()},
        "failed_temp_indices": failed,
        "counts_preservation_snapshot_failures":
            base.validate_counts_preserved(run_dir, counts),
        "observed_heat_peak_temperature":
            float(summary["temp"][peak_idx]),
        "observed_heat_peak_over_landau":
            float(summary["temp"][peak_idx]/t_landau),
        "stationarity_max_half_diff_E": float(max_hd),
    })
    (run_dir/"metadata.json").write_text(json.dumps(metadata, indent=2))

    plot_run_linear(run_dir, t_landau, save_indices,
                    name.replace("_", " ") + "  (ordered init, linear grid)")

    print(f"  Wall time: {wall_time/60:.1f} min", flush=True)
    print(f"  C_S peak T*={metadata['observed_heat_peak_temperature']:.4g}"
          f"  ({metadata['observed_heat_peak_over_landau']:.2f}× T*_Landau)", flush=True)
    print(f"  Stationarity max half-diff/E: {max_hd:.4g}", flush=True)
    return metadata


def make_linear_grid(t_start, t_end, step, save_targets):
    n = round((t_end - t_start) / step) + 1
    temps = np.linspace(t_end, t_start, n)        # hot → cold, exactly n points
    temps = np.round(temps, 10)
    save_indices = sorted({int(np.argmin(np.abs(temps - t))) for t in save_targets})
    return temps, save_indices


def plot_run_linear(run_dir, t_landau, save_indices, title):
    """Like base.plot_run but with linear x-axis (for linear T* grids)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summary = np.genfromtxt(run_dir / "summary.tsv", names=True, delimiter="\t")
    if summary.shape == ():
        summary = np.asarray([summary])
    temps = summary["temp"]
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.0), sharex=True)
    axes[0].plot(temps, summary["mean_E"], marker="o", ms=3, lw=1.4)
    axes[0].axvline(t_landau, color="black", ls="--", lw=1.0, label=f"T*_Landau={t_landau:.3g}")
    for idx in save_indices:
        axes[0].axvline(temps[idx], color="tab:gray", alpha=0.4, lw=0.9)
    axes[0].set_ylabel("mean E = -2S/(Nz)")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].set_title(title)
    axes[1].plot(temps, summary["heat_capacity_S"], marker="o", ms=3, lw=1.4, color="tab:red")
    axes[1].axvline(t_landau, color="black", ls="--", lw=1.0)
    for idx in save_indices:
        axes[1].axvline(temps[idx], color="tab:gray", alpha=0.4, lw=0.9)
    axes[1].set_ylabel("C_S = Var(S)/(N T*²)")
    axes[1].set_xlabel("T*  (linear)")
    fig.tight_layout()
    fig.savefig(run_dir / "energy_heat.png", dpi=180)
    fig.savefig(run_dir / "energy_heat.pdf")
    plt.close(fig)


def counts_geometric_d24():
    # f_0=1, f_i=2^(i-1) i=1..24 — identical to Mohanty (2021) k=2,d=24
    c = [1] + [1 << (i-1) for i in range(1, 25)]
    return np.array(c, dtype=np.uint64)


def main():
    compiler_flags = base.compile_core()
    print(f"Compiled: {compiler_flags}", flush=True)
    results = []

    # ── Geometric: matches Mohanty (2021) k=2, d=24 exactly ─────────────────
    # T* 0.1..15.0, δT*=0.1, 150 temps  ≈4.3 h wall-clock
    counts_g = counts_geometric_d24()
    temps_g, sidx_g = make_linear_grid(
        0.1, 15.0, 0.1,
        save_targets=[1.0, 3.5, 5.5, 7.5, 13.0])   # cold, below, at T*_Landau, above, random
    m = run_case("d24_geometric_ordered", 24, counts_g, seed=5100,
                 temps=temps_g, save_indices=sidx_g,
                 description="Geometric d=24, ordered init, linear grid 0.1-15 δ0.1. Matches Mohanty (2021) density.")
    if m: results.append(m)

    # ── Zipf: transition ~1.1-1.7, T*_c_MF~3.1 ──────────────────────────────
    # T* 0.1..8.0, δT*=0.1, 80 temps  ≈2.9 h wall-clock
    counts_z = base.counts_zipf(24, 4)
    h_z, _   = base.landau_h_factor(counts_z)
    tl_z     = 22.0 * h_z
    temps_z, sidx_z = make_linear_grid(
        0.1, 8.0, 0.1,
        save_targets=[0.35, tl_z, 1.5, 3.0, 7.0])
    m = run_case("d24_zipf_ordered", 24, counts_z, seed=4100,
                 temps=temps_z, save_indices=sidx_z,
                 description="Zipf d=24, ordered init, linear grid 0.1-8 δ0.1.")
    if m: results.append(m)

    # ── Uniform: transition ~T*_c_MF=1.32 ────────────────────────────────────
    # T* 0.05..3.0, δT*=0.05, 60 temps  ≈2.1 h wall-clock
    counts_u = base.counts_uniform(24)
    h_u, _   = base.landau_h_factor(counts_u)
    tl_u     = 22.0 * h_u
    temps_u, sidx_u = make_linear_grid(
        0.05, 3.0, 0.05,
        save_targets=[0.05, 0.8, 1.35, 2.0, 3.0])
    m = run_case("d24_uniform_ordered", 24, counts_u, seed=3100,
                 temps=temps_u, save_indices=sidx_u,
                 description="Uniform d=24, ordered init, linear grid 0.05-3 δ0.05.")
    if m: results.append(m)

    report = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
              "root": str(ROOT), "cases": results}
    (ROOT/"run_report_parallel.json").write_text(json.dumps(report, indent=2))

    print("\n" + "="*60 + "\nSUMMARY")
    for r in results:
        print(f"  {r['name']}: C_S peak {r['observed_heat_peak_over_landau']:.2f}× T*_Landau,"
              f" wall {r['wall_time_seconds']/60:.0f} min")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
d-series runs for geometric, uniform, and zipf distributions.

Geometric : d = 8..22 step 2  (counts_geometric scales naturally, Q = d+1)
Uniform   : d = 8..22 step 2  (counts_uniform scales naturally, Q = 2^(d//2))
Zipf cb=4 : d = 20, 22 only   (min count = 2^(d-19); invalid below d=20)

All d-series run dirs land in runs/dseries/.
Overlay plots land in plots/mergedheatplots/.

Grids (two-zone in T*/lambda_1 space):
  Geometric: dense around T*/lam1~0.22  (d=24 empirical peak 4.9)
  Uniform  : dense around T*/lam1~0.13  (d=24 empirical peak 2.5-3.3)
  Zipf cb=4: dense around T*/lam1~0.12  (d=24 empirical peak 1.9-3.5)
"""
import json, shutil, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
import run_hamming_sims as base
import run_d24_production as prod
from run_d24_parallel import run_single_temp, merge_workers, plot_run_linear

REPO      = Path("/home/gijs/Documents/Thesis-MSc")
SWEEPS    = 100
BURN      = 50
THIN      = 1
N_WORKERS = 12
INIT      = "ordered"

DSERIES_DIR     = ROOT / "runs" / "dseries"
MERGEDPLOTS_DIR = ROOT / "plots" / "mergedheatplots"

D_SERIES_GEOM   = [8, 10, 12, 14, 16, 18, 20, 22]
D_SERIES_UNIF   = [8, 10, 12, 14, 16, 18, 20, 22]
D_SERIES_ZIPF   = [20, 22]   # class_bits=4 needs d>=20


def counts_geometric(d):
    return np.array([1] + [1 << (i - 1) for i in range(1, d + 1)], dtype=np.uint64)


def make_geom_grid(lam1):
    """Grid aligned with the plot window T*/lam1 in [0.05, 0.40].
    Sparse below the transition, dense through the peak (~0.18-0.28)."""
    cold   = np.arange(0.05, 0.10 + 1e-9, 0.025)   # 0.05, 0.075, 0.10
    dense  = np.arange(0.10, 0.40 + 1e-9, 0.007)    # 0.10 … 0.40  (~44 pts)
    ratios = np.unique(np.round(np.concatenate([cold, dense]), 8))
    return np.sort(ratios * lam1)[::-1]


def make_uniform_grid(lam1):
    """Two-zone grid for uniform: dense around T*/lam1~0.13 (d=24 peak 2.5-3.3)."""
    cold   = np.arange(0.01, 0.09 + 1e-9, 0.02)
    dense  = np.arange(0.09, 0.25 + 1e-9, 0.005)
    hot    = np.array([0.30, 0.40, 0.55, 0.70])
    ratios = np.unique(np.concatenate([cold, dense, hot]))
    return np.sort(np.round(ratios, 8) * lam1)[::-1]


def make_zipf_grid(lam1):
    """Two-zone grid for zipf cb=4: dense around T*/lam1~0.12 (d=24 peak 1.9-3.5)."""
    cold   = np.arange(0.01, 0.08 + 1e-9, 0.02)
    dense  = np.arange(0.08, 0.23 + 1e-9, 0.005)
    hot    = np.array([0.30, 0.40, 0.55, 0.70])
    ratios = np.unique(np.concatenate([cold, dense, hot]))
    return np.sort(np.round(ratios, 8) * lam1)[::-1]


def run_case(name, d, counts, seed, temps):
    run_dir = DSERIES_DIR / name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    counts_path = run_dir / "counts.tsv"
    base.write_counts(counts_path, counts)

    lam1     = d - 2
    h_f, _   = base.landau_h_factor(counts)
    t_landau = lam1 * h_f
    n_temps  = len(temps)

    metadata = {
        "name": name, "d": d, "k": 2, "N": int(1 << d),
        "Q": len(counts), "lambda_1": lam1,
        "h_f": float(h_f), "T_star_Landau": float(t_landau),
        "temperature_grid": temps.tolist(),
        "sweeps": SWEEPS, "burn_in": BURN, "initialization": INIT,
        "rng_seed_base": seed,
        "git_commit_Thesis_MSc": base.git_commit(REPO),
        "heat_capacity_convention": "C_S = Var(S)/(N*T*^2)",
        "high_T_random_E": base.high_temp_expectation(counts),
        "snapshot_dtype": base.dtype_for_q(len(counts)),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"\n{'='*55}\n{name}", flush=True)
    print(f"  d={d}  N={1<<d:,}  Q={len(counts)}  lam1={lam1}"
          f"  T*_Landau={t_landau:.3g}  {n_temps} temps", flush=True)

    wall_start = time.perf_counter()
    runtimes, failed = {}, []

    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {
            pool.submit(run_single_temp,
                        gi, float(temps[gi]), False,
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
            if done % 12 == 0 or done == n_temps:
                print(f"  [{done:3d}/{n_temps}] last T*={temps[gi]:.3f}"
                      f"  {rt:.0f}s  {status}", flush=True)
            if rc != 0:
                failed.append(gi)

    wall_time = time.perf_counter() - wall_start
    print(f"  Merging {n_temps} workers...", flush=True)
    merge_workers(run_dir, temps, [], n_temps)

    summary = np.genfromtxt(run_dir / "summary.tsv", names=True, delimiter="\t")
    if summary.shape == ():
        summary = np.asarray([summary])

    diagnostics = prod.stationarity_diagnostics(run_dir, BURN)
    (run_dir / "stationarity_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2))
    max_hd = max((d2["abs_half_difference"] for d2 in diagnostics), default=0.0)

    peak_idx = int(np.argmax(summary["heat_capacity_S"]))
    metadata.update({
        "wall_time_seconds": wall_time,
        "failed_temp_indices": failed,
        "observed_peak_T_star": float(summary["temp"][peak_idx]),
        "observed_peak_over_lam1": float(summary["temp"][peak_idx] / lam1),
        "stationarity_max_half_diff_E": float(max_hd),
        "counts_preservation_snapshot_failures": [],
    })
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    plot_run_linear(run_dir, t_landau, [],
                    f"{name}  (d={d}, Q={len(counts)}, {SWEEPS}+{BURN} sweeps)")

    print(f"  Wall: {wall_time/60:.1f} min  peak T*/lam1="
          f"{metadata['observed_peak_over_lam1']:.3f}", flush=True)
    return metadata


def _find_run_dir(dist_name, d):
    """Look in dseries/ first, fall back to runs/ root (for in-progress runs)."""
    new = DSERIES_DIR / f"{dist_name}_d{d}_ordered"
    old = ROOT / "runs" / f"{dist_name}_d{d}_ordered"
    return new if new.exists() else old


def overlay_plot(dist_name, run_dir_d24, d_list, name):
    """Overlay E and C_S vs T*/lambda_1 for d=24 plus all d in d_list."""
    out_path = MERGEDPLOTS_DIR / name
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    cmap  = plt.cm.viridis
    all_d = sorted(set([24] + list(d_list)))   # ascending so low-d = dark, high-d = bright
    n     = len(all_d)
    # clip to 0.05–0.90 to avoid the very dark purple and very bright yellow extremes
    colors = {d: cmap(0.05 + 0.85 * i / max(n - 1, 1)) for i, d in enumerate(all_d)}

    for d in sorted(all_d, reverse=True):
        rdir = Path(run_dir_d24) if d == 24 else _find_run_dir(dist_name, d)
        if not (rdir / "summary.tsv").exists():
            print(f"  overlay: skipping d={d} (no summary.tsv at {rdir})", flush=True)
            continue
        meta = json.loads((rdir / "metadata.json").read_text())
        lam1 = meta.get("lambda_1", d - 2)
        summ = np.genfromtxt(rdir / "summary.tsv", names=True, delimiter="\t")
        if summ.shape == ():
            summ = np.asarray([summ])
        x  = summ["temp"] / lam1
        lw = 1.8 if d == 24 else 1.2
        axes[0].plot(x, summ["mean_E"],          color=colors[d], lw=lw, label=f"d={d}")
        axes[1].plot(x, summ["heat_capacity_S"], color=colors[d], lw=lw, label=f"d={d}")

    for ax in axes:
        ax.axvline(1.0, color="gray", ls="--", lw=0.8, alpha=0.6, label="T*_Landau")
        ax.legend(fontsize=8, frameon=False, ncol=2)

    axes[0].set_ylabel("mean E = −2S/(Nz)")
    axes[0].set_title(f"{dist_name} d-series  ({SWEEPS}+{BURN} sweeps, ordered init)")
    axes[1].set_ylabel("C_S = Var(S)/(N T*²)")
    axes[1].set_xlabel("T* / λ₁  (λ₁ = d − 2)")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=180)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  Overlay saved: {out_path.with_suffix('.png')}", flush=True)


def run_geometric_series():
    print("\n" + "="*55)
    print("GEOMETRIC d-series")
    results = []
    for d in D_SERIES_GEOM:
        counts = counts_geometric(d)
        lam1   = d - 2
        temps  = make_geom_grid(lam1)
        m = run_case(f"geometric_d{d}_ordered", d, counts, seed=5200 + d, temps=temps)
        if m:
            results.append(m)
    overlay_plot("geometric", ROOT / "runs" / "d24_geometric_ordered",
                 D_SERIES_GEOM, "geometric_dseries_overlay")
    return results


def run_uniform_series():
    print("\n" + "="*55)
    print("UNIFORM d-series")
    results = []
    for d in D_SERIES_UNIF:
        counts = base.counts_uniform(d)
        lam1   = d - 2
        temps  = make_uniform_grid(lam1)
        m = run_case(f"uniform_d{d}_ordered", d, counts, seed=3200 + d, temps=temps)
        if m:
            results.append(m)
    overlay_plot("uniform", ROOT / "runs" / "d24_uniform_ordered",
                 D_SERIES_UNIF, "uniform_dseries_overlay")
    return results


def run_zipf_series():
    print("\n" + "="*55)
    print("ZIPF d-series  (class_bits=4, d>=20 only)")
    results = []
    for d in D_SERIES_ZIPF:
        counts = base.counts_zipf(d, 4)
        lam1   = d - 2
        temps  = make_zipf_grid(lam1)
        m = run_case(f"zipf_cb4_d{d}_ordered", d, counts, seed=4200 + d, temps=temps)
        if m:
            results.append(m)
    overlay_plot("zipf_cb4", ROOT / "runs" / "d24_zipf_ordered",
                 D_SERIES_ZIPF, "zipf_dseries_overlay")
    return results


def main():
    DSERIES_DIR.mkdir(parents=True, exist_ok=True)
    MERGEDPLOTS_DIR.mkdir(parents=True, exist_ok=True)
    base.compile_core()

    all_results = []
    all_results += run_geometric_series()
    all_results += run_uniform_series()
    all_results += run_zipf_series()

    report = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
              "cases": all_results}
    (ROOT / "run_report_dseries.json").write_text(json.dumps(report, indent=2))
    print("\nAll d-series done.")


if __name__ == "__main__":
    main()

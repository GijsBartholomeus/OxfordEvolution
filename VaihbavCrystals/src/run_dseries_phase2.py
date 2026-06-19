#!/usr/bin/env python3
"""
Phase-2 d-series: rerun d=8..20 with sweep count scaled to total moves ≈ 100×N_20.
d=22 is already smooth at 100 sweeps — load existing data for the overlay.
Produces normalized C_S overlay (C_S / max C_S) for d=8..22, no d=24.

Distributions:
  Geometric : d = 8..22   (d=22 from phase-1)
  Uniform   : d = 8..22   (d=22 from phase-1)
  Zipf cb=4 : d = 20, 22  (d=22 from phase-1; d=20 rerun)
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
from run_dseries import (counts_geometric, make_geom_grid, make_uniform_grid,
                         make_zipf_grid, DSERIES_DIR, MERGEDPLOTS_DIR)

REPO      = Path("/home/gijs/Documents/Thesis-MSc")
BURN      = 50
THIN      = 1
N_WORKERS = 12
INIT      = "ordered"

D_RERUN_GEOM  = [8, 10, 12, 14, 16, 18, 20]   # d=22 not rerun — use phase-1
D_RERUN_UNIF  = [8, 10, 12, 14, 16, 18, 20]
D_RERUN_ZIPF  = [20]                            # d=22 not rerun — class_bits=4 valid


def sweeps_for_d(d):
    """Scale sweeps so total MCMC moves ≈ 100 × N at d=20."""
    n_ref = 1 << 20
    n     = 1 << d
    return max(100, min(10000, 100 * n_ref // n))


def run_case(name, d, counts, seed, temps):
    sweeps  = sweeps_for_d(d)
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
        "sweeps": sweeps, "burn_in": BURN, "initialization": INIT,
        "rng_seed_base": seed,
        "git_commit_Thesis_MSc": base.git_commit(REPO),
        "heat_capacity_convention": "C_S = Var(S)/(N*T*^2)",
        "high_T_random_E": base.high_temp_expectation(counts),
        "snapshot_dtype": base.dtype_for_q(len(counts)),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"\n{'='*55}\n{name}", flush=True)
    print(f"  d={d}  N={1<<d:,}  Q={len(counts)}  sweeps={sweeps}  {n_temps} temps",
          flush=True)

    wall_start = time.perf_counter()
    failed = []

    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {
            pool.submit(run_single_temp,
                        gi, float(temps[gi]), False,
                        run_dir, counts_path,
                        d, sweeps, BURN, THIN, seed): gi
            for gi in range(n_temps)
        }
        done = 0
        for fut in as_completed(futures):
            gi, wdir, rt, rc = fut.result()
            done += 1
            if done % 12 == 0 or done == n_temps:
                print(f"  [{done:3d}/{n_temps}]  {rt:.0f}s", flush=True)
            if rc != 0:
                failed.append(gi)

    wall_time = time.perf_counter() - wall_start
    merge_workers(run_dir, temps, [], n_temps)

    summary = np.genfromtxt(run_dir / "summary.tsv", names=True, delimiter="\t")
    if summary.shape == ():
        summary = np.asarray([summary])

    peak_idx = int(np.argmax(summary["heat_capacity_S"]))
    metadata.update({
        "wall_time_seconds": wall_time,
        "failed_temp_indices": failed,
        "observed_peak_T_star": float(summary["temp"][peak_idx]),
        "observed_peak_over_lam1": float(summary["temp"][peak_idx] / lam1),
        "counts_preservation_snapshot_failures": [],
    })
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    plot_run_linear(run_dir, t_landau, [],
                    f"{name}  (d={d}, {sweeps}+{BURN} sweeps)")

    print(f"  Wall: {wall_time/60:.1f} min  peak T*/lam1="
          f"{metadata['observed_peak_over_lam1']:.3f}", flush=True)
    return metadata


def _find_dir(dist_name, d):
    """Find run dir: dseries/ first, then runs/ root as fallback."""
    new = DSERIES_DIR / f"{dist_name}_d{d}_ordered"
    old = ROOT / "runs" / f"{dist_name}_d{d}_ordered"
    return new if new.exists() else old


def overlay_normalized(dist_name, d_list, out_stem):
    """Two-panel overlay: E and C_S/max(C_S) vs T*/lambda_1 for d in d_list."""
    all_d   = sorted(d_list)
    n       = len(all_d)
    cmap    = plt.cm.viridis
    colors  = {d: cmap(0.05 + 0.85 * i / max(n - 1, 1)) for i, d in enumerate(all_d)}

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    for d in sorted(all_d, reverse=True):
        rdir = _find_dir(dist_name, d)
        if not (rdir / "summary.tsv").exists():
            print(f"  overlay: skipping d={d} (missing)", flush=True)
            continue
        meta  = json.loads((rdir / "metadata.json").read_text())
        lam1  = meta.get("lambda_1", d - 2)
        sw    = meta.get("sweeps", "?")
        summ  = np.genfromtxt(rdir / "summary.tsv", names=True, delimiter="\t")
        if summ.shape == ():
            summ = np.asarray([summ])
        x    = summ["temp"] / lam1
        cs   = summ["heat_capacity_S"]
        cs_n = cs / cs.max() if cs.max() > 0 else cs
        lw   = 1.5
        axes[0].plot(x, summ["mean_E"], color=colors[d], lw=lw, label=f"d={d} ({sw}sw)")
        axes[1].plot(x, cs_n,           color=colors[d], lw=lw, label=f"d={d}")

    for ax in axes:
        ax.legend(fontsize=8, frameon=False, ncol=2)

    axes[0].set_ylabel("mean E = −2S/(Nz)")
    axes[0].set_title(
        f"{dist_name} d-series  d=8..22  (ordered init, scaled sweeps)")
    axes[1].set_ylabel("C_S / max(C_S)")
    axes[1].set_xlabel("T* / λ₁  (λ₁ = d − 2)")
    fig.tight_layout()
    out = MERGEDPLOTS_DIR / out_stem
    fig.savefig(out.with_suffix(".png"), dpi=180)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  Normalized overlay: {out.with_suffix('.png')}", flush=True)


def run_geometric_phase2():
    print("\n" + "="*55 + "\nGEOMETRIC phase-2")
    for d in D_RERUN_GEOM:
        counts = counts_geometric(d)
        run_case(f"geometric_d{d}_ordered", d, counts,
                 seed=5200 + d, temps=make_geom_grid(d - 2))
    overlay_normalized("geometric",
                       D_RERUN_GEOM + [22],          # d=22 from phase-1
                       "geometric_dseries_normalized")


def run_uniform_phase2():
    print("\n" + "="*55 + "\nUNIFORM phase-2")
    for d in D_RERUN_UNIF:
        counts = base.counts_uniform(d)
        run_case(f"uniform_d{d}_ordered", d, counts,
                 seed=3200 + d, temps=make_uniform_grid(d - 2))
    overlay_normalized("uniform",
                       D_RERUN_UNIF + [22],
                       "uniform_dseries_normalized")


def run_zipf_phase2():
    print("\n" + "="*55 + "\nZIPF phase-2 (class_bits=4, d>=20)")
    for d in D_RERUN_ZIPF:
        counts = base.counts_zipf(d, 4)
        run_case(f"zipf_cb4_d{d}_ordered", d, counts,
                 seed=4200 + d, temps=make_zipf_grid(d - 2))
    overlay_normalized("zipf_cb4",
                       D_RERUN_ZIPF + [22],
                       "zipf_dseries_normalized")


def main():
    DSERIES_DIR.mkdir(parents=True, exist_ok=True)
    MERGEDPLOTS_DIR.mkdir(parents=True, exist_ok=True)
    base.compile_core()

    run_geometric_phase2()
    run_uniform_phase2()
    run_zipf_phase2()

    print("\nPhase-2 done. Plots in:", MERGEDPLOTS_DIR)


if __name__ == "__main__":
    main()

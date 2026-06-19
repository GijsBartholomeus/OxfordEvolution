#!/usr/bin/env python3
"""Rerun d=8..14 with high sweep counts to smooth jagged curves."""
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
from run_d24_parallel import run_single_temp, merge_workers, plot_run_linear
from run_dseries import (counts_geometric, make_geom_grid, make_uniform_grid,
                         DSERIES_DIR, MERGEDPLOTS_DIR)

REPO      = Path("/home/gijs/Documents/Thesis-MSc")
BURN      = 50
THIN      = 1
N_WORKERS = 12
INIT      = "ordered"

SMOOTH_SWEEPS = {8: 1_000_000, 10: 300_000, 12: 80_000, 14: 20_000}


def run_case(name, d, counts, seed, temps):
    sweeps  = SMOOTH_SWEEPS[d]
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

    print(f"\n{'='*55}\n{name}  sweeps={sweeps:,}", flush=True)

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

    summ = np.genfromtxt(run_dir / "summary.tsv", names=True, delimiter="\t")
    if summ.shape == ():
        summ = np.asarray([summ])
    peak_idx = int(np.argmax(summ["heat_capacity_S"]))
    metadata.update({
        "wall_time_seconds": wall_time,
        "failed_temp_indices": failed,
        "observed_peak_T_star": float(summ["temp"][peak_idx]),
        "observed_peak_over_lam1": float(summ["temp"][peak_idx] / lam1),
        "counts_preservation_snapshot_failures": [],
    })
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    plot_run_linear(run_dir, t_landau, [],
                    f"{name}  ({sweeps:,}+{BURN} sweeps)")
    print(f"  Wall: {wall_time/60:.1f} min  peak T*/lam1="
          f"{metadata['observed_peak_over_lam1']:.3f}", flush=True)


def find_dir(dist, d):
    p = DSERIES_DIR / f"{dist}_d{d}_ordered"
    return p if p.exists() else ROOT / "runs" / f"{dist}_d{d}_ordered"


def make_normalized_overlay(dist_name, d_list, out_stem):
    all_d  = sorted(d_list)
    n      = len(all_d)
    cmap   = plt.cm.viridis
    colors = {d: cmap(0.05 + 0.85*i/max(n-1,1)) for i,d in enumerate(all_d)}
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    for d in sorted(all_d, reverse=True):
        rdir = find_dir(dist_name, d)
        if not (rdir / "summary.tsv").exists():
            print(f"  skip d={d}"); continue
        meta = json.loads((rdir / "metadata.json").read_text())
        lam1 = meta.get("lambda_1", d - 2)
        sw   = meta.get("sweeps", "?")
        summ = np.genfromtxt(rdir / "summary.tsv", names=True, delimiter="\t")
        if summ.shape == ():
            summ = np.asarray([summ])
        x    = summ["temp"] / lam1
        cs   = summ["heat_capacity_S"]
        cs_n = cs / cs.max()
        axes[0].plot(x, summ["mean_E"], color=colors[d], lw=1.5,
                     label=f"d={d} ({sw:,}sw)" if isinstance(sw,int) else f"d={d} ({sw}sw)")
        axes[1].plot(x, cs_n,           color=colors[d], lw=1.5, label=f"d={d}")

    axes[0].set_ylabel("mean E = −2S/(Nz)")
    axes[0].set_title(f"{dist_name} d=8..22  normalized C_S  (ordered init)")
    axes[0].legend(fontsize=7, frameon=False, ncol=2)
    axes[1].set_ylabel("C_S / max(C_S)")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("T* / λ₁  (λ₁ = d − 2)")
    axes[1].legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    out = MERGEDPLOTS_DIR / out_stem
    fig.savefig(out.with_suffix(".png"), dpi=180)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  Saved: {out}.png", flush=True)


def main():
    base.compile_core()

    print("=== GEOMETRIC small-d smooth rerun ===")
    for d in sorted(SMOOTH_SWEEPS):
        run_case(f"geometric_d{d}_ordered", d, counts_geometric(d),
                 seed=5200+d, temps=make_geom_grid(d-2))
    make_normalized_overlay("geometric", list(range(8,24,2)),
                            "geometric_dseries_normalized")

    print("\n=== UNIFORM small-d smooth rerun ===")
    for d in sorted(SMOOTH_SWEEPS):
        run_case(f"uniform_d{d}_ordered", d, base.counts_uniform(d),
                 seed=3200+d, temps=make_uniform_grid(d-2))
    make_normalized_overlay("uniform", list(range(8,24,2)),
                            "uniform_dseries_normalized")

    print("\nDone.")


if __name__ == "__main__":
    main()

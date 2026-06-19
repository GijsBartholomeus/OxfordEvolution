#!/usr/bin/env python3
"""Post-process already-completed d24_uniform run: generate stationarity diagnostics, update metadata, plot."""
import json
from pathlib import Path
import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
import run_hamming_sims as base
import run_d24_production as prod

RUN_DIR = ROOT / "runs" / "d24_uniform"
BURN = 20

meta = json.loads((RUN_DIR / "metadata.json").read_text())
t_landau = meta["T_star_Landau"]
save_indices = meta["selected_snapshot_indices"]

diagnostics = prod.stationarity_diagnostics(RUN_DIR, BURN)
(RUN_DIR / "stationarity_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))

summary = np.genfromtxt(RUN_DIR / "summary.tsv", names=True, delimiter="\t")
peak_idx = int(np.argmax(summary["heat_capacity_S"]))
max_half_diff = max((d["abs_half_difference"] for d in diagnostics), default=0.0)

meta.update({
    "counts_preservation_snapshot_failures": base.validate_counts_preserved(
        RUN_DIR, np.loadtxt(RUN_DIR / "counts.tsv", dtype=np.uint64)
    ),
    "observed_heat_peak_temperature": float(summary["temp"][peak_idx]),
    "observed_heat_peak_over_landau": float(summary["temp"][peak_idx] / t_landau),
    "stationarity_diagnostics_file": str(RUN_DIR / "stationarity_diagnostics.json"),
    "stationarity_max_half_diff_E": float(max_half_diff),
    "postprocessed": True,
})
(RUN_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))

base.plot_run(RUN_DIR, t_landau, save_indices, "d24 uniform (Q=4096)")
print("Plot written to", RUN_DIR / "energy_heat.png")
print(f"Heat peak at T={meta['observed_heat_peak_temperature']:.5g} ({meta['observed_heat_peak_over_landau']:.3f} x T*_Landau)")
print(f"Stationarity max half-diff E = {max_half_diff:.4g}")
print()
print("NOTE: E at hottest temp = -0.0212 vs high-T random = -0.000244")
print("Window is entirely in the ordered regime. Transition is above our grid.")

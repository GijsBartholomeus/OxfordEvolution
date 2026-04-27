from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import roadrunner

from wsbw_pipeline import (
    DIVERGENCE_CAP_FACTOR,
    PLOTS,
    RESULTS,
    SPECS,
    encode_signal,
    prepare_models,
)


def reset_initials(rr, base_initials):
    for sid, val in base_initials.items():
        try:
            rr.setValue(f"init({sid})", val)
        except Exception:
            pass


def simulate_trace(rr, spec, defaults, base_initials, rng, divergence_cap):
    reset_initials(rr, base_initials)
    rr.resetAll()
    if spec.setup:
        spec.setup(rr)
    for pid, val in defaults.items():
        rr.setValue(pid, val)
    for pid, val in defaults.items():
        rr.setValue(pid, val * rng.choice([0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]))
    if spec.warmup:
        spec.warmup(rr)
    rr.selections = ["time", spec.output]
    result = np.asarray(rr.simulate(0, spec.t_end, spec.npoints), dtype=float)
    t = result[:, 0]
    y = result[:, 1]
    if not np.all(np.isfinite(y)):
        return None
    if divergence_cap is not None and np.any(np.abs(y) > divergence_cap):
        return None
    if np.any(np.abs(y) > 1e9):
        return None
    mask = (t >= spec.coarse_start) & (t <= spec.coarse_start + spec.coarse_duration)
    bits = encode_signal(t[mask], y[mask], 50)
    return t, y, bits


def pick_targets(data):
    phenos = sorted(data["phenotypes"], key=lambda p: (p["complexity"], -p["count"]))
    low = phenos[0]
    high = phenos[-1]
    return low, high


def find_representatives(spec, audit, data, seed, max_draws=5000):
    low, high = pick_targets(data)
    targets = {"low": low["encoding"], "high": high["encoding"]}
    found = {}

    rr = roadrunner.RoadRunner(audit[spec.key]["promoted_sbml"])
    if spec.setup:
        spec.setup(rr)
    defaults = {pid: float(rr.getValue(pid)) for pid in audit[spec.key]["free_parameters"]}
    base_initials = {}
    for sid in rr.model.getFloatingSpeciesIds():
        try:
            base_initials[sid] = float(rr.getValue(f"init({sid})"))
        except Exception:
            base_initials[sid] = float(rr.getValue(sid))

    reset_initials(rr, base_initials)
    rr.resetAll()
    if spec.setup:
        spec.setup(rr)
    for pid, val in defaults.items():
        rr.setValue(pid, val)
    if spec.warmup:
        spec.warmup(rr)
    rr.selections = ["time", spec.output]
    wt_result = np.asarray(rr.simulate(0, spec.t_end, spec.npoints), dtype=float)
    divergence_cap = DIVERGENCE_CAP_FACTOR * max(float(np.max(np.abs(wt_result[:, 1]))), 1e-12)

    rng = random.Random(seed)
    for _ in range(max_draws):
        try:
            trace = simulate_trace(rr, spec, defaults, base_initials, rng, divergence_cap)
        except Exception:
            continue
        if trace is None:
            continue
        t, y, bits = trace
        for name, target in targets.items():
            if name not in found and bits == target:
                found[name] = {
                    "time": t,
                    "signal": y,
                    "encoding": bits,
                    "complexity": low["complexity"] if name == "low" else high["complexity"],
                    "count": low["count"] if name == "low" else high["count"],
                }
        if len(found) == 2:
            break
    return found, low, high


def plot_representatives(seed=42):
    audit = prepare_models()
    fig_low, axes_low = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    fig_high, axes_high = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    report = {}

    for idx, spec in enumerate(SPECS):
        data = json.loads((RESULTS / f"{spec.key}_complexity_frequency.json").read_text())
        found, low, high = find_representatives(spec, audit, data, seed + idx)
        report[spec.key] = {
            "low_target": {"complexity": low["complexity"], "count": low["count"], "found": "low" in found},
            "high_target": {"complexity": high["complexity"], "count": high["count"], "found": "high" in found},
        }
        for name, fig, axes, target in [
            ("low", fig_low, axes_low, low),
            ("high", fig_high, axes_high, high),
        ]:
            ax = axes.ravel()[idx]
            if name in found:
                t = found[name]["time"]
                y = found[name]["signal"]
                mask = (t >= spec.coarse_start) & (t <= spec.coarse_start + spec.coarse_duration)
                t = t[mask] - spec.coarse_start
                y = y[mask]
                if spec.t_end > 1000:
                    t = t / 60.0
                    xlabel = "time in phenotype window (min)"
                else:
                    xlabel = "time in phenotype window"
                ax.plot(t, y, lw=1.5)
                ax.set_xlabel(xlabel)
            else:
                ax.text(0.5, 0.5, "not found on replay", ha="center", va="center")
            ax.set_title(f"{spec.label}\nK={target['complexity']:.1f}, n={target['count']}", fontsize=10)
            ax.set_ylabel(spec.output)
            ax.grid(alpha=0.25)

    low_out = PLOTS / "oscillatory_subset_low_complexity_representatives_trough_windows.png"
    high_out = PLOTS / "oscillatory_subset_high_complexity_representatives_trough_windows.png"
    fig_low.savefig(low_out, dpi=220)
    fig_high.savefig(high_out, dpi=220)
    fig_low.savefig(PLOTS / "oscillatory_subset_low_complexity_representatives.png", dpi=220)
    fig_high.savefig(PLOTS / "oscillatory_subset_high_complexity_representatives.png", dpi=220)
    (RESULTS / "representative_trace_report.json").write_text(json.dumps(report, indent=2))
    print(low_out)
    print(high_out)


if __name__ == "__main__":
    plot_representatives()

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import libsbml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import roadrunner


ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
SOURCES = ROOT / "sources" / "Dataset1"
PROMOTED = ROOT / "models_promoted"
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"
for path in (PROMOTED, RESULTS, PLOTS):
    path.mkdir(exist_ok=True)

MULTIPLIERS = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
DIVERGENCE_CAP_FACTOR = 100.0


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    source_dir: str
    sbml: Path
    output: str
    t_end: float
    npoints: int
    coarse_start: float
    coarse_duration: float
    promoted_sbml: Path | None = None
    setup: Callable[[roadrunner.RoadRunner], None] | None = None
    warmup: Callable[[roadrunner.RoadRunner], None] | None = None


def set_if_exists(rr: roadrunner.RoadRunner, key: str, value: float) -> None:
    try:
        rr.setValue(key, value)
    except Exception:
        pass


def setup_chen(rr):
    set_if_exists(rr, "PE", 0.698687)
    set_if_exists(rr, "CDC15", 0.6565)
    set_if_exists(rr, "CDC15i", 0.3435)


def warmup_leloup(rr):
    rr.simulate(0, 72, 500)
    for sid in rr.model.getFloatingSpeciesIds():
        rr.setValue(f"init({sid})", rr.getValue(sid))
    rr.reset()


def warmup_locke(rr):
    rr.simulate(0, 24 * 10, 1200)
    for sid in rr.model.getFloatingSpeciesIds():
        rr.setValue(f"init({sid})", rr.getValue(sid))
    rr.reset()


def warmup_ueda(rr):
    rr.simulate(0, 20 * 24, 2000)
    for sid in rr.model.getFloatingSpeciesIds():
        rr.setValue(f"init({sid})", rr.getValue(sid))
    rr.reset()


SPECS = [
    ModelSpec("chen2004", "Chen 2004", "Chen_2004", MODELS / "BIOMD0000000056.xml", "CLB2", 418.948, 1001, 14.140, 404.808, setup=setup_chen),
    ModelSpec("kholodenko2000", "Kholodenko 2000", "Kholodenko_2000", MODELS / "BIOMD0000000010.xml", "MKK_PP", 6526.107, 1201, 1301.750, 5224.357),
    ModelSpec("leloup1999", "Leloup 1999", "Leloup_1999", MODELS / "BIOMD0000000021.xml", "Cn", 131.371, 1001, 34.795, 96.576, warmup=warmup_leloup),
    ModelSpec("locke2005", "Locke 2005", "Locke_2005", MODELS / "BIOMD0000000055.xml", "cXn", 106.272, 1001, 10.272, 96.000, warmup=warmup_locke),
    ModelSpec("ueda2001", "Ueda 2001", "Ueda_2001", MODELS / "BIOMD0000000022.xml", "CCc", 88.713, 1001, 2.269, 86.444, warmup=warmup_ueda),
    ModelSpec("vilar2002", "Vilar 2002", "Vilar_2002", MODELS / "BIOMD0000000035.xml", "C", 116.960, 1201, 14.705, 102.255),
]


def hessian_keys(spec: ModelSpec) -> list[str]:
    path = SOURCES / spec.source_dir / "hessian_keys.dat"
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def canonical_key(key: str) -> str:
    return re.sub(r"_\d+$", "", key)


def local_parameter_values(sbml: Path) -> dict[str, list[tuple[str, float]]]:
    doc = libsbml.readSBML(str(sbml))
    model = doc.getModel()
    values: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for rxn in model.getListOfReactions():
        kl = rxn.getKineticLaw()
        if kl is None:
            continue
        locals_list = []
        if hasattr(kl, "getListOfLocalParameters"):
            locals_list.extend(list(kl.getListOfLocalParameters()))
        if hasattr(kl, "getListOfParameters"):
            locals_list.extend(list(kl.getListOfParameters()))
        for par in locals_list:
            values[par.getId()].append((rxn.getId(), par.getValue()))
    return values


def promote_local_parameters(spec: ModelSpec) -> Path:
    keys = {canonical_key(k) for k in hessian_keys(spec)}
    out = PROMOTED / f"{spec.key}.xml"
    doc = libsbml.readSBML(str(spec.sbml))
    model = doc.getModel()

    existing_globals = {p.getId() for p in model.getListOfParameters()}
    local_values = local_parameter_values(spec.sbml)
    for pid in sorted(keys):
        if pid in existing_globals or pid not in local_values:
            continue
        vals = [v for _, v in local_values[pid]]
        if max(vals) - min(vals) > 1e-12:
            # Duplicate local IDs with different values cannot be safely promoted to one global.
            continue
        par = model.createParameter()
        par.setId(pid)
        par.setConstant(True)
        par.setValue(vals[0])

    for rxn in model.getListOfReactions():
        kl = rxn.getKineticLaw()
        if kl is None:
            continue
        for pid in sorted(keys):
            if pid in existing_globals or pid in {p.getId() for p in model.getListOfParameters()}:
                if hasattr(kl, "getLocalParameter") and kl.getLocalParameter(pid) is not None:
                    kl.removeLocalParameter(pid)
                if hasattr(kl, "getParameter") and kl.getParameter(pid) is not None:
                    kl.removeParameter(pid)

    libsbml.writeSBMLToFile(doc, str(out))
    return out


def candidate_parameter_ids(spec: ModelSpec, sbml: Path) -> tuple[list[str], dict[str, list[str]]]:
    rr = roadrunner.RoadRunner(str(sbml))
    if spec.setup:
        spec.setup(rr)
    if spec.key == "chen2004":
        resolved = []
        rejected: dict[str, list[str]] = defaultdict(list)
        for pid in rr.model.getGlobalParameterIds():
            try:
                val = float(rr.getValue(pid))
                rr.setValue(pid, val)
            except Exception:
                rejected["not_settable"].append(pid)
                continue
            low = pid.lower()
            if (low.endswith("t") and val in (0.0, 1.0)) or (
                low.startswith("d") and low.endswith("n")
            ) or ("flag" in low) or ("switch" in low) or val == 0.0 or pid in {"cell"} or (
                "total" in low and val in (0.0, 1.0)
            ):
                rejected["nonkinetic_or_switch_like"].append(pid)
            else:
                resolved.append(pid)
        return resolved, dict(rejected)

    global_ids = set(rr.model.getGlobalParameterIds())
    boundary_ids = set(rr.model.getBoundarySpeciesIds())
    keys = hessian_keys(spec)
    resolved = []
    rejected: dict[str, list[str]] = defaultdict(list)
    seen = set()

    for raw in keys:
        options = [raw, canonical_key(raw)]
        pid = next((opt for opt in options if opt in global_ids or opt in boundary_ids), None)
        if pid is None:
            rejected["not_settable"].append(raw)
            continue
        if pid in seen:
            continue
        seen.add(pid)
        try:
            val = float(rr.getValue(pid))
            rr.setValue(pid, val)
        except Exception:
            rejected["not_settable"].append(raw)
            continue
        low = pid.lower()
        if not math.isfinite(val):
            rejected["nonfinite"].append(pid)
        elif val == 0.0:
            rejected["zero"].append(pid)
        elif val in (0.0, 1.0) and (low in {"light", "emptyset"} or low.startswith("switch") or "flag" in low):
            rejected["binary_switch"].append(pid)
        elif low in {"n", "emptyset", "light", "time", "turntime"}:
            rejected["manual_exclude"].append(pid)
        else:
            resolved.append(pid)
    return resolved, dict(rejected)


def prepare_models() -> dict[str, dict]:
    audit = {}
    for spec in SPECS:
        sbml = promote_local_parameters(spec)
        params, rejected = candidate_parameter_ids(spec, sbml)
        audit[spec.key] = {
            "label": spec.label,
            "source_sbml": str(spec.sbml),
            "promoted_sbml": str(sbml),
            "output": spec.output,
            "free_parameter_count": len(params),
            "free_parameters": params,
            "rejected": rejected,
        }
    (RESULTS / "parameter_audit.json").write_text(json.dumps(audit, indent=2))
    return audit


def lz76_phrase_count(s: str) -> int:
    n = len(s)
    if n == 0:
        return 0
    i = 0
    c = 1
    k = 1
    while i + k <= n:
        if s[i : i + k] in s[: i + k - 1]:
            k += 1
            if i + k - 1 > n:
                c += 1
                break
        else:
            c += 1
            i += k
            k = 1
    return c


def clz(bits: str) -> float:
    if not bits:
        return 0.0
    if bits.count("0") == len(bits) or bits.count("1") == len(bits):
        return math.log2(len(bits))
    return math.log2(len(bits)) / 2.0 * (lz76_phrase_count(bits) + lz76_phrase_count(bits[::-1]))


def encode_signal(time: np.ndarray, signal: np.ndarray, nbins: int = 50) -> str:
    coarse_time = np.linspace(time[0], time[-1], nbins)
    coarse_signal = np.interp(coarse_time, time, signal)
    slopes = np.diff(coarse_signal) / np.diff(coarse_time)
    return "".join("1" if slope > 0 else "0" for slope in slopes)


def simulate_encoding(
    rr: roadrunner.RoadRunner,
    spec: ModelSpec,
    defaults: dict[str, float],
    base_initials: dict[str, float],
    wildtype: bool,
    rng: random.Random,
    divergence_cap: float | None = None,
):
    for sid, val in base_initials.items():
        try:
            rr.setValue(f"init({sid})", val)
        except Exception:
            pass
    rr.resetAll()
    if spec.setup:
        spec.setup(rr)
    for pid, val in defaults.items():
        rr.setValue(pid, val)
    if not wildtype:
        for pid, val in defaults.items():
            rr.setValue(pid, val * rng.choice(MULTIPLIERS))
    if spec.warmup:
        spec.warmup(rr)
    rr.selections = ["time", spec.output]
    result = rr.simulate(0, spec.t_end, spec.npoints)
    t = np.asarray(result[:, 0], dtype=float)
    y = np.asarray(result[:, 1], dtype=float)
    if not np.all(np.isfinite(y)):
        return None
    if divergence_cap is not None and np.any(np.abs(y) > divergence_cap):
        return None
    if np.any(np.abs(y) > 1e9):
        return None
    mask = (t >= spec.coarse_start) & (t <= spec.coarse_start + spec.coarse_duration)
    if not np.any(mask):
        return None
    bits = encode_signal(t[mask], y[mask], 50)
    return bits


def run_model(spec: ModelSpec, audit: dict, samples: int = 5000, seed: int = 1):
    sbml = Path(audit[spec.key]["promoted_sbml"])
    params = audit[spec.key]["free_parameters"]
    rng = random.Random(seed)
    counts = Counter()
    failures = 0
    rr = roadrunner.RoadRunner(str(sbml))
    if spec.setup:
        spec.setup(rr)
    defaults = {pid: float(rr.getValue(pid)) for pid in params}
    base_initials = {}
    for sid in rr.model.getFloatingSpeciesIds():
        try:
            base_initials[sid] = float(rr.getValue(f"init({sid})"))
        except Exception:
            base_initials[sid] = float(rr.getValue(sid))
    wt_bits = simulate_encoding(rr, spec, defaults, base_initials, wildtype=True, rng=rng)
    restore_initials = dict(base_initials)
    for sid, val in restore_initials.items():
        try:
            rr.setValue(f"init({sid})", val)
        except Exception:
            pass
    rr.resetAll()
    if spec.setup:
        spec.setup(rr)
    for pid, val in defaults.items():
        rr.setValue(pid, val)
    if spec.warmup:
        spec.warmup(rr)
    rr.selections = ["time", spec.output]
    wt_result = rr.simulate(0, spec.t_end, spec.npoints)
    wt_signal = np.asarray(wt_result[:, 1], dtype=float)
    wildtype_max_abs = float(np.max(np.abs(wt_signal)))
    divergence_cap = DIVERGENCE_CAP_FACTOR * max(wildtype_max_abs, 1e-12)
    for _ in range(samples):
        try:
            bits = simulate_encoding(
                rr,
                spec,
                defaults,
                base_initials,
                wildtype=False,
                rng=rng,
                divergence_cap=divergence_cap,
            )
            if bits is None:
                failures += 1
            else:
                counts[bits] += 1
        except Exception:
            failures += 1
    data = {
        "model": spec.key,
        "label": spec.label,
        "samples": samples,
        "successes": sum(counts.values()),
        "failures": failures,
        "wildtype_encoding": wt_bits,
        "wildtype_complexity": clz(wt_bits) if wt_bits else None,
        "wildtype_count": counts.get(wt_bits, 0) if wt_bits else 0,
        "wildtype_max_abs": wildtype_max_abs,
        "divergence_cap_factor": DIVERGENCE_CAP_FACTOR,
        "divergence_cap": divergence_cap,
        "time_window": {
            "t_end": spec.t_end,
            "coarse_start": spec.coarse_start,
            "coarse_duration": spec.coarse_duration,
        },
        "phenotypes": [{"encoding": enc, "count": n, "complexity": clz(enc)} for enc, n in counts.items()],
    }
    (RESULTS / f"{spec.key}_complexity_frequency.json").write_text(json.dumps(data))
    return data


def plot_complexity_frequency(all_data: list[dict], out: Path | None = None):
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, data, color in zip(axes.ravel(), all_data, colors):
        phenos = data["phenotypes"]
        successes = max(data["successes"], 1)
        xs = np.array([p["complexity"] for p in phenos])
        ys = np.array([p["count"] / successes for p in phenos])
        ax.scatter(xs, ys, s=10, color="black", alpha=0.55, linewidths=0)
        bins = defaultdict(list)
        for x, y in zip(xs, ys):
            bins[round(float(x), 1)].append(float(y))
        if len(bins) >= 2:
            bx = np.array(sorted(bins))
            upper = np.array([max(bins[x]) for x in bx])
            lower = np.array([min(bins[x]) for x in bx])
            ax.fill_between(bx, lower, upper, color=color, alpha=0.35)
            ax.plot(bx, upper, color=color, lw=1.5)
        if data["wildtype_encoding"]:
            wt_x = data["wildtype_complexity"]
            wt_y = max(data["wildtype_count"], 0.5) / successes
            ax.scatter([wt_x], [wt_y], color="red", s=34, zorder=4)
        ax.set_yscale("log")
        ax.set_title(f"{data['label']}\n{len(phenos)} phenotypes, {successes} successes", fontsize=10)
        ax.set_xlabel("K(x)")
        ax.set_ylabel("P(x)")
        ax.grid(alpha=0.25)
    if out is None:
        out = PLOTS / "oscillatory_subset_complexity_frequency_trough_windows.png"
    fig.savefig(out, dpi=220)
    legacy_out = PLOTS / "oscillatory_subset_complexity_frequency.png"
    if out != legacy_out:
        fig.savefig(legacy_out, dpi=220)
    return out


def main(samples: int = 1000, seed: int = 1):
    audit = prepare_models()
    all_data = []
    for idx, spec in enumerate(SPECS):
        print(f"Running {spec.label} with {audit[spec.key]['free_parameter_count']} free parameters")
        all_data.append(run_model(spec, audit, samples=samples, seed=seed + idx))
    out = plot_complexity_frequency(all_data)
    print(out)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    main(samples=args.samples, seed=args.seed)

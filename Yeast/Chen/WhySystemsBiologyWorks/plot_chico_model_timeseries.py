from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import roadrunner
from scipy.integrate import solve_ivp


ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
PLOTS = ROOT / "plots"
DATASET = ROOT / "sources" / "Dataset1"
PLOTS.mkdir(exist_ok=True)


def simulate_sbml(xml: Path, species: str, start: float, end: float, points: int = 1200, setup=None):
    rr = roadrunner.RoadRunner(str(xml))
    if setup:
        setup(rr)
    rr.selections = ["time", species]
    out = rr.simulate(start, end, points)
    return np.asarray(out[:, 0], dtype=float), np.asarray(out[:, 1], dtype=float)


def set_if_present(rr, var: str, value: float) -> None:
    try:
        rr[var] = value
    except Exception:
        pass


def lee_rhs(_t, y, W):
    X2, X3, X4, X9, X10, X11, X12 = y
    Dsh0, APC0, TCF0, GSK0 = 100.0, 100.0, 15.0, 50.0
    K7, K8, K16, K17 = 50.0, 120.0, 30.0, 1200.0
    k1, k2, k3, k4, k5 = 0.182, 1.82e-2, 5e-2, 0.267, 0.133
    k6, km6, k9, k10, k11, k13, k15 = 9.09e-2, 0.909, 206.0, 206.0, 0.417, 2.57e-4, 0.167
    v12, v14 = 0.423, 8.22e-5

    a = 1 + APC0 * K17 / (K7 * (K17 + X11))
    b = APC0 * K17 * X12 / (K7 * (K17 + X11) ** 2)
    c = k3 * X2 * X4 - k6 * GSK0 * APC0 * K17 * X12 / (K7 * (K17 + X11)) + km6 * X4 + v14 - k15 * X12
    d = 1 + X11 / K8
    e = X3 / K8
    f = k4 * X4 - k5 * X3 - k9 * X3 * X11 / K8 + k10 * X9
    g = 1 + X3 / K8 + TCF0 * K16 / (K16 + X11) ** 2 + APC0 * K17 / (K17 + X11) ** 2
    h = X11 / K8
    i = v12 - (k9 * X3 / K8 + k13) * X11

    denom = d * g - e * h
    rhs_x11 = (d * i - f * h) / denom
    rhs_x12 = (c + rhs_x11 * b) / a
    rhs_x3 = (e * i - f * g) / (e * h - d * g)

    return [
        k1 * W * (Dsh0 - X2) - k2 * X2,
        rhs_x3,
        -k3 * X2 * X4 - k4 * X4 + k5 * X3 + k6 * GSK0 * (K17 * X12 * APC0 / (K7 * (K17 + X11))) - km6 * X4,
        k9 * (X3 * X11 / K8) - k10 * X9,
        k10 * X9 - k11 * X10,
        rhs_x11,
        rhs_x12,
    ]


def simulate_lee():
    steady = solve_ivp(lambda t, y: lee_rhs(t, y, 0.0), (0, 1e5), np.zeros(7), method="LSODA", rtol=1e-8, atol=1e-10)
    y0 = steady.y[:, -1]
    t_eval = np.linspace(0, 16 * 60, 1200)
    sol = solve_ivp(lambda t, y: lee_rhs(t, y, math.exp(-t / 20.0)), (0, 16 * 60), y0, t_eval=t_eval, method="LSODA", rtol=1e-8, atol=1e-10)
    X11 = sol.y[5]
    return sol.t / 60.0, X11


def total_sasagawa(rr, pattern: str):
    ids = rr.model.getFloatingSpeciesIds()
    selected = [sid for sid in ids if pattern in sid]
    return selected


def simulate_total_species(xml: Path, species_ids: list[str], start: float, end: float, points: int = 1200, setup=None):
    rr = roadrunner.RoadRunner(str(xml))
    if setup:
        setup(rr)
    rr.selections = ["time"] + species_ids
    out = rr.simulate(start, end, points)
    return np.asarray(out[:, 0], dtype=float), np.asarray(out[:, 1:], dtype=float).sum(axis=1)


def main():
    entries = [
        ("1 Brown 2004", MODELS / "BIOMD0000000033.xml", "BRafActive", 0, 120, "Active B-Raf", lambda rr: (set_if_present(rr, "EGF", 0), set_if_present(rr, "NGF", 456000))),
        ("2 Chassagnole 2002", MODELS / "BIOMD0000000051.xml", "cpg", 0, 40, "2-phosphoglycerate", None),
        ("3 Chen 2004", MODELS / "BIOMD0000000056.xml", "CLB2", 0, 500, "CLB2", lambda rr: (set_if_present(rr, "PE", 0.698687), set_if_present(rr, "CDC15", 0.6565), set_if_present(rr, "CDC15i", 0.3435))),
        ("4 Edelstein 1996", MODELS / "BIOMD0000000002.xml", "ALL", 1e-5, 1e2, "Biliganded ACh receptor (ALL)", lambda rr: (set_if_present(rr, "B", 1), set_if_present(rr, "L", 1e-5))),
        ("5 Kholodenko 2000", MODELS / "BIOMD0000000010.xml", "MKK_PP", 0, 205 * 60, "MAP2K-PP / MKK-PP", None),
        ("7 Leloup 1999", MODELS / "BIOMD0000000021.xml", "Cn", 0, 96, "Nuclear PER-TIM", None),
        ("8 Locke 2005", MODELS / "BIOMD0000000055.xml", "cXn", 0, 96, "Nuclear X protein", None),
        ("10 Ueda 2001", MODELS / "BIOMD0000000022.xml", "CCc", 0, 72, "Cytoplasmic Clk-Cyc", None),
        ("11 Vilar 2002", MODELS / "BIOMD0000000035.xml", "C", 0, 200, "C protein", None),
    ]

    fig, axes = plt.subplots(4, 3, figsize=(15, 10), constrained_layout=True)
    axes = axes.ravel()
    report = []

    for ax, (title, xml, species, start, end, ylabel, setup) in zip(axes, entries):
        try:
            t, y = simulate_sbml(xml, species, start, end, setup=setup)
            if title.startswith("4 "):
                ax.set_xscale("log")
            if "Kholodenko" in title:
                t = t / 60.0
                ax.set_xlabel("time (min)")
            ax.plot(t, y, lw=1.7)
            ax.set_title(title, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=9)
            report.append(f"OK {title}: {species}, range {np.nanmin(y):.4g}..{np.nanmax(y):.4g}")
        except Exception as exc:
            ax.text(0.5, 0.5, f"failed\n{exc}", ha="center", va="center", fontsize=8)
            ax.set_title(title, fontsize=10)
            report.append(f"FAIL {title}: {exc}")

    ax = axes[len(entries)]
    try:
        t, y = simulate_lee()
        ax.plot(t, y, lw=1.7)
        ax.set_title("6 Lee 2003", fontsize=10)
        ax.set_ylabel("Non-active beta-catenin (X11)", fontsize=9)
        ax.set_xlabel("time (h)")
        report.append(f"OK 6 Lee 2003: X11, range {np.nanmin(y):.4g}..{np.nanmax(y):.4g}")
    except Exception as exc:
        ax.text(0.5, 0.5, f"failed\n{exc}", ha="center", va="center", fontsize=8)
        ax.set_title("6 Lee 2003", fontsize=10)
        report.append(f"FAIL 6 Lee 2003: {exc}")

    ax = axes[len(entries) + 1]
    try:
        ids = total_sasagawa(roadrunner.RoadRunner(str(MODELS / "BIOMD0000000049.xml")), "Ras_GDP")
        t, y = simulate_total_species(MODELS / "BIOMD0000000049.xml", ids, 0, 60 * 60)
        ax.plot(t / 60.0, y, lw=1.7)
        ax.set_title("9 Sasagawa 2005", fontsize=10)
        ax.set_ylabel("Ras-GDP total", fontsize=9)
        ax.set_xlabel("time (min)")
        report.append(f"OK 9 Sasagawa 2005: {'+'.join(ids)}, range {np.nanmin(y):.4g}..{np.nanmax(y):.4g}")
    except Exception as exc:
        ax.text(0.5, 0.5, f"failed\n{exc}", ha="center", va="center", fontsize=8)
        ax.set_title("9 Sasagawa 2005", fontsize=10)
        report.append(f"FAIL 9 Sasagawa 2005: {exc}")

    ax = axes[len(entries) + 2]
    try:
        t, y = simulate_sbml(DATASET / "Zak_2003" / "Model-l2v1.xml", "MA", 0, 48 * 60)
        ax.plot(t / 60.0, y, lw=1.7)
        ax.set_title("12 Zak 2003", fontsize=10)
        ax.set_ylabel("Gene A mRNA (MA)", fontsize=9)
        ax.set_xlabel("time (min)")
        report.append(f"OK 12 Zak 2003: MA, range {np.nanmin(y):.4g}..{np.nanmax(y):.4g}")
    except Exception as exc:
        ax.text(0.5, 0.5, f"failed\n{exc}", ha="center", va="center", fontsize=8)
        ax.set_title("12 Zak 2003", fontsize=10)
        report.append(f"FAIL 12 Zak 2003: {exc}")

    for ax in axes:
        ax.grid(alpha=0.25)
        if not ax.get_xlabel():
            ax.set_xlabel("time")

    out = PLOTS / "chico_table1_relevant_outputs.png"
    fig.savefig(out, dpi=220)
    (PLOTS / "chico_table1_relevant_outputs_report.txt").write_text("\n".join(report) + "\n")
    print(out)
    print("\n".join(report))


if __name__ == "__main__":
    main()

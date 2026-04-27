from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import roadrunner
from scipy.integrate import solve_ivp

from plot_chico_model_timeseries import MODELS, PLOTS, DATASET, lee_rhs


def rr_model(acc: str):
    return roadrunner.RoadRunner(str(MODELS / f"{acc}.xml"))


def sim(rr, selections, start, end, points=1000):
    rr.selections = selections
    return np.asarray(rr.simulate(start, end, points), dtype=float)


def setv(rr, key, val):
    try:
        rr[key] = val
    except Exception:
        pass


def lee_repro():
    steady = solve_ivp(lambda t, y: lee_rhs(t, y, 0.0), (0, 1e5), np.zeros(7), method="LSODA", rtol=1e-8, atol=1e-10)
    y0 = steady.y[:, -1]
    t_eval = np.linspace(0, 16 * 60, 1200)
    sol = solve_ivp(lambda t, y: lee_rhs(t, y, np.exp(-t / 20.0)), (0, 16 * 60), y0, t_eval=t_eval, method="LSODA", rtol=1e-8, atol=1e-10)
    X2, X3, X4, X9, X10, X11, X12 = sol.y
    X14 = X11 * 15.0 / (30.0 + X11)
    X15 = X11 * 100.0 / (1200.0 + X11)
    X8 = X3 * X11 / 120.0
    total_bcat = X8 + X9 + X10 + X11 + X14 + X15
    return sol.t / 60.0, X11, total_bcat


def zak_single_pulse():
    rr = roadrunner.RoadRunner(str(DATASET / "Zak_2003" / "Model-l2v1.xml"))
    setv(rr, "S_t", 0)
    rr.simulate(0, 2000 * 60, 200)
    species = ["MF", "MB", "MD", "MJ", "MA", "MC", "ME", "MH", "MG", "MK"]
    y0 = {sid: rr[sid] for sid in species}
    chunks = [(0, 10 * 60, 0), (10 * 60, 10 * 60 + 10, 1), (10 * 60 + 10, 80 * 60, 0)]
    times, vals = [], []
    for start, end, stim in chunks:
        setv(rr, "S_t", stim)
        out = sim(rr, ["time"] + species, start, end, max(20, int((end - start) / 4)))
        times.append(out[:, 0] / 60.0)
        vals.append(out[:, 1:])
    return np.concatenate(times), np.vstack(vals), y0, species


def main():
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    ax = axes.ravel()

    rr = rr_model("BIOMD0000000033")
    setv(rr, "NGF", 456000)
    out = sim(rr, ["time", "BRafActive", "ErkActive"], 0, 120)
    ax[0].plot(out[:, 0], out[:, 1], label="BRafActive")
    ax[0].plot(out[:, 0], out[:, 2], label="ErkActive")
    ax[0].set_title("1 Brown 2004: growth-factor response")
    ax[0].legend(fontsize=8)

    rr = rr_model("BIOMD0000000051")
    out = sim(rr, ["time", "cpg", "cpep", "cpyr"], 0, 40)
    for i, name in enumerate(["cpg", "cpep", "cpyr"], start=1):
        ax[1].plot(out[:, 0], out[:, i], label=name)
    ax[1].set_title("2 Chassagnole 2002: metabolic relaxation")
    ax[1].legend(fontsize=8)

    rr = rr_model("BIOMD0000000002")
    setv(rr, "B", 1)
    setv(rr, "L", 1e-5)
    out = sim(rr, ["time", "BLL", "ALL", "ILL", "DLL"], 1e-5, 1e2)
    biliganded = out[:, 1:].sum(axis=1)
    ax[2].semilogx(out[:, 0], biliganded, label="BLL+ALL+ILL+DLL")
    ax[2].semilogx(out[:, 0], out[:, 1], label="BLL", alpha=0.8)
    ax[2].set_title("4 Edelstein 1996: biliganded states")
    ax[2].legend(fontsize=8)

    t, x11, total = lee_repro()
    ax[3].plot(t, x11, label="X11 non-active beta-catenin")
    ax[3].plot(t, total, label="total beta-catenin", alpha=0.8)
    ax[3].set_title("6 Lee 2003: Wnt transient")
    ax[3].legend(fontsize=8)

    rr = rr_model("BIOMD0000000049")
    ids_ras_gtp = [sid for sid in rr.model.getFloatingSpeciesIds() if "Ras_GTP" in sid]
    ids_ras_gdp = [sid for sid in rr.model.getFloatingSpeciesIds() if sid == "Ras_GDP"]
    out = sim(rr, ["time"] + ids_ras_gdp + ids_ras_gtp, 0, 60 * 60)
    ax[4].plot(out[:, 0] / 60.0, out[:, 1], label="Ras_GDP")
    ax[4].plot(out[:, 0] / 60.0, out[:, 2:].sum(axis=1), label="total Ras_GTP")
    ax[4].set_title("9 Sasagawa 2005: Ras pathway response")
    ax[4].legend(fontsize=8)

    t, vals, y0, species = zak_single_pulse()
    for idx, name in enumerate(["MA", "MF", "MD"]):
        col = species.index(name)
        denom = max(y0[name], 1.0)
        ax[5].plot(t, np.log(np.maximum(vals[:, col], 1.0) / denom), label=name)
    ax[5].set_title("12 Zak 2003: single input pulse")
    ax[5].legend(fontsize=8)

    for a in ax:
        a.set_xlabel("time")
        a.grid(alpha=0.25)

    out = PLOTS / "scrutiny_nonoscillatory_models.png"
    fig.savefig(out, dpi=220)
    print(out)


if __name__ == "__main__":
    main()

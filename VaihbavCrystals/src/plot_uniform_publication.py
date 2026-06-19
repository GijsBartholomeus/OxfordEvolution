#!/usr/bin/env python3
"""
Publication figure: uniform d-series  —  energy + normalised heat capacity.

Output: plots/mergedheatplots/uniform_energy_heatcapacity.{pdf,png}
"""
import json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))

DSERIES_DIR     = ROOT / "runs" / "dseries"
RUNS_DIR        = ROOT / "runs"
MERGEDPLOTS_DIR = ROOT / "plots" / "mergedheatplots"
D_LIST          = [8, 10, 12, 14, 16, 18, 20, 22]

_cmap  = plt.cm.plasma
_n     = len(D_LIST)
COLORS = {d: _cmap(0.07 + 0.80 * i / (_n - 1)) for i, d in enumerate(D_LIST)}

SG_WIN, SG_POLY = 9, 3

def sg(y):
    return savgol_filter(y, SG_WIN, SG_POLY) if len(y) >= SG_WIN else y

def find_dir(d):
    p = DSERIES_DIR / f"uniform_d{d}_ordered"
    return p if p.exists() else RUNS_DIR / f"uniform_d{d}_ordered"

def load(d):
    rdir  = find_dir(d)
    meta  = json.loads((rdir / "metadata.json").read_text())
    s     = np.genfromtxt(rdir / "summary.tsv", names=True, delimiter="\t")
    if s.shape == ():
        s = np.asarray([s])
    lam1  = meta["lambda_1"]
    order = np.argsort(s["temp"])
    x    = s["temp"][order] / lam1
    E    = s["mean_E"][order]
    cs   = s["heat_capacity_S"][order]
    cs_n = cs / cs.max()
    return x, E, cs_n, meta["sweeps"]

XLIM = (0.05, 0.18)

fig = plt.figure(figsize=(6.5, 5.2))
gs  = fig.add_gridspec(2, 1,
                        hspace=0.06,
                        left=0.14, right=0.96,
                        top=0.96, bottom=0.11)
ax_E  = fig.add_subplot(gs[0])
ax_cs = fig.add_subplot(gs[1], sharex=ax_E)

legend_handles = []

for d in D_LIST:
    x, E, cs_n, sweeps = load(d)
    c  = COLORS[d]
    lw = 2.0

    h, = ax_E.plot(x, E, color=c, lw=lw, label=f"$d = {d}$")
    legend_handles.append(h)

    cs_sm = sg(cs_n)
    ax_cs.plot(x, cs_n,  color=c, lw=0.8, alpha=0.20, zorder=2)
    ax_cs.plot(x, cs_sm, color=c, lw=lw,              zorder=3)

    pk  = int(np.argmax(cs_sm))
    xpk = x[pk]
    for ax in (ax_E, ax_cs):
        ax.axvline(xpk, color=c, lw=0.7, ls="--", alpha=0.45, zorder=1)
    ax_cs.plot(xpk, cs_sm[pk], "o", color=c, ms=5.0,
               markeredgewidth=0.5, markeredgecolor="white", zorder=5)

ax_E.set_xlim(XLIM)
ax_cs.set_xlim(XLIM)
ax_E.set_ylim(-0.55, 0.01)
ax_cs.set_ylim(-0.03, 1.12)

GRID_KW = dict(color="#e4e4e4", lw=0.5, zorder=0)
for ax in (ax_E, ax_cs):
    ax.tick_params(axis="both", direction="out", which="both",
                   labelsize=9.5, top=False, right=False)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(True, which="major", **GRID_KW)
plt.setp(ax_E.get_xticklabels(), visible=False)

ax_E.set_ylabel(r"$\langle E\rangle = -2S/(Nz)$", fontsize=11)
ax_cs.set_ylabel(r"$C_S\,/\,\max(C_S)$", fontsize=11)
ax_cs.set_xlabel(r"$T^*/\lambda_1$  $(\lambda_1 = d-2)$", fontsize=11)

for ax, lbl in ((ax_E, "(a)"), (ax_cs, "(b)")):
    ax.text(0.017, 0.96, lbl, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top")

ax_E.legend(handles=legend_handles,
            loc="lower right",
            fontsize=8.5,
            frameon=True,
            framealpha=0.85,
            edgecolor="#cccccc",
            handlelength=1.4,
            borderpad=0.6,
            labelspacing=0.25,
            ncol=2)

out = MERGEDPLOTS_DIR / "uniform_energy_heatcapacity"
fig.savefig(out.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}.pdf / .png")

#!/usr/bin/env python3
"""Energy-only publication figure for the uniform d-series."""
import json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))

DSERIES_DIR     = ROOT / "runs" / "dseries"
RUNS_DIR        = ROOT / "runs"
MERGEDPLOTS_DIR = ROOT / "plots" / "mergedheatplots"
D_LIST          = [8, 10, 12, 14, 16, 18, 20, 22]

_cmap  = plt.cm.plasma
_n     = len(D_LIST)
COLORS = {d: _cmap(0.07 + 0.80 * i / (_n - 1)) for i, d in enumerate(D_LIST)}

XLIM = (0.05, 0.18)

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
    x = s["temp"][order] / lam1
    E = s["mean_E"][order]
    return x, E

fig, ax = plt.subplots(figsize=(6.5, 3.2))
fig.subplots_adjust(left=0.14, right=0.96, top=0.96, bottom=0.16)

legend_handles = []
for d in D_LIST:
    x, E = load(d)
    c = COLORS[d]
    h, = ax.plot(x, E, color=c, lw=2.0, label=f"$d = {d}$")
    legend_handles.append(h)

ax.set_xlim(XLIM)
ax.set_ylim(-0.55, 0.01)
ax.tick_params(axis="both", direction="out", which="both",
               labelsize=9.5, top=False, right=False)
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.grid(True, which="major", color="#e4e4e4", lw=0.5, zorder=0)

ax.set_ylabel(r"$\langle E\rangle = -2S/(Nz)$", fontsize=11)
ax.set_xlabel(r"$T^*/\lambda_1$  $(\lambda_1 = d-2)$", fontsize=11)

ax.legend(handles=legend_handles,
          loc="lower right",
          fontsize=8.5,
          frameon=True,
          framealpha=0.85,
          edgecolor="#cccccc",
          handlelength=1.4,
          borderpad=0.6,
          labelspacing=0.25,
          ncol=2)

out = MERGEDPLOTS_DIR / "uniform_energy_only"
fig.savefig(out.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}.pdf / .png")

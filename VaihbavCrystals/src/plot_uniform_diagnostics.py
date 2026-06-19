#!/usr/bin/env python3
"""
Publication figures for uniform d=8..20 diagnostics.

Figure 1  (uniform_diag_order_parameter.{pdf,png})
  (a) m₁w² / (d/2)   normalised Landau order parameter vs T*/λ₁
      [ground-state value = 1 for all d, collapses to universal curve]
  (b) m₁²  × Q / (d/2)   same normalisation via m₁² × Q = m₁w²

Figure 2  (uniform_diag_Gprofile.{pdf,png})
  (a) G(ℓ) vs ℓ for d=14, several temperatures
  (b) G(ℓ) vs ℓ for multiple d at T*/λ₁ ≈ 0.12 (near uniform transition)

Same plasma-colormap style as the energy/heat-capacity figures.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))

import run_hamming_sims as base

DIAG_DIR  = ROOT / "runs" / "diagnostics"
PLOTS_DIR = ROOT / "plots" / "mergedheatplots"

D_LIST_ALL = [8, 10, 12, 14, 16, 18, 20, 22]   # for colormap spacing
D_LIST     = [8, 10, 12, 14, 16, 18, 20]

_cmap  = plt.cm.plasma
_n     = len(D_LIST_ALL)
COLORS = {d: _cmap(0.07 + 0.80 * i / (_n - 1)) for i, d in enumerate(D_LIST_ALL)}

XLIM   = (0.05, 0.18)
SG_WIN, SG_POLY = 7, 3


def sg(y):
    if len(y) < SG_WIN or not np.all(np.isfinite(y)):
        return y
    return savgol_filter(y, SG_WIN, SG_POLY)


def load(d):
    path = DIAG_DIR / f"uniform_d{d}_diag" / "diag_summary.tsv"
    data = {}
    with path.open() as f:
        header = f.readline().strip().split("\t")
        rows   = [line.strip().split("\t") for line in f if line.strip()]
    for col, vals in zip(header, zip(*rows)):
        try:
            data[col] = np.array([float(v) for v in vals])
        except ValueError:
            data[col] = np.array(vals)
    order = np.argsort(data["temp_over_lam1"])
    return {k: v[order] if isinstance(v, np.ndarray) else v for k, v in data.items()}


GRID_KW = dict(color="#e4e4e4", lw=0.5, zorder=0)


def style_ax(ax):
    ax.tick_params(axis="both", direction="out", which="both",
                   labelsize=9.5, top=False, right=False)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(True, which="major", **GRID_KW)


def add_legend(ax, handles, loc="upper right"):
    ax.legend(handles=handles, loc=loc, fontsize=8.5,
              frameon=True, framealpha=0.85, edgecolor="#cccccc",
              handlelength=1.4, borderpad=0.6, labelspacing=0.25, ncol=2)


def make_figure():
    fig = plt.figure(figsize=(6.5, 5.2))
    gs  = fig.add_gridspec(2, 1, hspace=0.06,
                            left=0.14, right=0.96,
                            top=0.96, bottom=0.11)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)
    return fig, ax_top, ax_bot


# ── Figure 1: normalised order parameter ─────────────────────────────────────

fig1, ax_m1w, ax_m1 = make_figure()
handles = []

for d in D_LIST:
    Q = 1 << (d // 2)
    try:
        data = load(d)
    except FileNotFoundError:
        print(f"  Skipping d={d}: not found")
        continue

    x    = data["temp_over_lam1"]
    m1   = data["m1_sq"]
    m1w  = data["m1w_sq"]
    c    = COLORS[d]
    lw   = 2.0

    if not np.any(np.isfinite(m1w)):
        continue

    # Normalise by ground-state value so all curves start at 1
    m1w_norm  = m1w  / (d / 2)          # ground state = 1 for all d
    m1_norm   = m1   / (d / (2 * Q))    # same normalisation via m1*Q = m1w

    m1w_sm  = sg(m1w_norm)
    m1_sm   = sg(m1_norm)

    ax_m1w.plot(x, m1w_norm, color=c, lw=0.8, alpha=0.20, zorder=2)
    ax_m1w.plot(x, m1w_sm,   color=c, lw=lw,              zorder=3)

    ax_m1.plot(x, m1_norm, color=c, lw=0.8, alpha=0.20, zorder=2)
    ax_m1.plot(x, m1_sm,   color=c, lw=lw,              zorder=3)

    h, = ax_m1w.plot([], [], color=c, lw=lw, label=f"$d = {d}$")
    handles.append(h)

ax_m1w.axhline(1.0, color="gray", lw=1.0, ls=":", alpha=0.7, zorder=1)
ax_m1w.text(XLIM[1]-0.005, 1.0 + 0.015, "ground state",
            ha="right", va="bottom", fontsize=8, color="gray")

for ax in (ax_m1w, ax_m1):
    ax.set_xlim(XLIM)
    ax.set_ylim(-0.03, 1.25)
    style_ax(ax)
plt.setp(ax_m1w.get_xticklabels(), visible=False)

ax_m1w.set_ylabel(r"$m_{1,w}^2\,/\,(d/2)$",   fontsize=11)
ax_m1.set_ylabel( r"$m_1^2\,/\,[d/(2Q)]$",     fontsize=11)
ax_m1.set_xlabel(r"$T^*/\lambda_1$  $(\lambda_1 = d-2)$", fontsize=11)

for ax, lbl in ((ax_m1w, "(a)"), (ax_m1, "(b)")):
    ax.text(0.017, 0.96, lbl, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top")

add_legend(ax_m1w, handles, loc="upper right")

out1 = PLOTS_DIR / "uniform_diag_order_parameter"
fig1.savefig(out1.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
fig1.savefig(out1.with_suffix(".png"), dpi=300, bbox_inches="tight")
plt.close(fig1)
print(f"Saved: {out1}.pdf / .png")


# ── Figure 2: G(ℓ) profiles ──────────────────────────────────────────────────

D_PROFILE = 14
T_FIXED   = 0.12   # near uniform transition
T_TARGETS = [0.06, 0.09, 0.11, 0.13, 0.16]
D_PANEL_B = [8, 12, 16, 20]

fig2, (ax_pa, ax_pb) = plt.subplots(1, 2, figsize=(9.0, 3.6))
fig2.subplots_adjust(left=0.09, right=0.97, bottom=0.16, top=0.94, wspace=0.32)

# panel (a)
try:
    data_p = load(D_PROFILE)
    x_all  = data_p["temp_over_lam1"]
    G_cols = sorted([n for n in data_p if n.startswith("G_") and n[2:].isdigit()],
                    key=lambda n: int(n[2:]))
    G_mat  = np.column_stack([data_p[n] for n in G_cols])
    ell    = np.arange(1, len(G_cols) + 1)
    t_cmap = plt.cm.coolwarm
    for t_target in T_TARGETS:
        idx = int(np.argmin(np.abs(x_all - t_target)))
        t_r = (t_target - T_TARGETS[0]) / (T_TARGETS[-1] - T_TARGETS[0])
        ax_pa.plot(ell, G_mat[idx], color=t_cmap(t_r), lw=1.8,
                   label=f"$T^*/\\lambda_1={x_all[idx]:.2f}$")
except (FileNotFoundError, KeyError):
    ax_pa.text(0.5, 0.5, f"d={D_PROFILE} not found",
               ha="center", va="center", transform=ax_pa.transAxes)

ax_pa.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.6)
ax_pa.set_xlabel(r"Hamming distance $\ell$", fontsize=11)
ax_pa.set_ylabel(r"$G(\ell) = q(\ell) - q_{\rm rand}$", fontsize=11)
ax_pa.set_title(f"$d = {D_PROFILE}$  (uniform)", fontsize=10, pad=4)
ax_pa.legend(fontsize=7.5, frameon=True, framealpha=0.85, edgecolor="#cccccc",
             handlelength=1.2, labelspacing=0.22, loc="upper right")

# panel (b)
for d in D_PANEL_B:
    try:
        data_b = load(d)
        x_b    = data_b["temp_over_lam1"]
        G_cols_b = sorted([n for n in data_b if n.startswith("G_") and n[2:].isdigit()],
                          key=lambda n: int(n[2:]))
        G_mat_b  = np.column_stack([data_b[n] for n in G_cols_b])
        idx_b    = int(np.argmin(np.abs(x_b - T_FIXED)))
        ell_b    = np.arange(1, len(G_cols_b) + 1)
        ax_pb.plot(ell_b, G_mat_b[idx_b], color=COLORS[d], lw=1.8, label=f"$d = {d}$")
    except (FileNotFoundError, KeyError):
        pass

ax_pb.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.6)
ax_pb.set_xlabel(r"Hamming distance $\ell$", fontsize=11)
ax_pb.set_ylabel(r"$G(\ell)$", fontsize=11)
ax_pb.set_title(f"$T^*/\\lambda_1 \\approx {T_FIXED}$  (uniform)", fontsize=10, pad=4)
ax_pb.legend(fontsize=8.5, frameon=True, framealpha=0.85, edgecolor="#cccccc",
             handlelength=1.2, labelspacing=0.22, loc="upper right")

for ax, lbl in ((ax_pa, "(a)"), (ax_pb, "(b)")):
    style_ax(ax)
    ax.text(0.017, 0.97, lbl, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top")

out2 = PLOTS_DIR / "uniform_diag_Gprofile"
fig2.savefig(out2.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
fig2.savefig(out2.with_suffix(".png"), dpi=300, bbox_inches="tight")
plt.close(fig2)
print(f"Saved: {out2}.pdf / .png")

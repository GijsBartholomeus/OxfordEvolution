#!/usr/bin/env python3
"""
Publication figures for geometric d=8..22 diagnostics:

Figure 1  (geom_diag_order_parameter.{pdf,png})
  (a) m₁²   = Σ_{a,j} M_{aj}²           vs T*/λ₁
  (b) m₁w²  = Σ_{a,j} M_{aj}²/f_a       vs T*/λ₁

Figure 2  (geom_diag_correlation.{pdf,png})
  (a) ξ  (correlation length from G(ℓ)~exp(−ℓ/ξ))  vs T*/λ₁
  (b) G(1) = q(1) − q_rand               vs T*/λ₁

Same plasma-colormap style as geometric_energy_heatcapacity.
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

DIAG_DIR  = ROOT / "runs" / "diagnostics"
PLOTS_DIR = ROOT / "plots" / "mergedheatplots"

# d=22 excluded: N=4M genotype array causes cache-thrashing; d=8..20 shows the full trend
D_LIST_ALL  = [8, 10, 12, 14, 16, 18, 20, 22]
D_LIST      = [8, 10, 12, 14, 16, 18, 20]

_cmap  = plt.cm.plasma
_n     = len(D_LIST_ALL)   # keep same colormap spacing as main energy figure
COLORS = {d: _cmap(0.07 + 0.80 * i / (_n - 1)) for i, d in enumerate(D_LIST_ALL)}

XLIM   = (0.05, 0.40)
SG_WIN, SG_POLY = 7, 3   # Savitzky-Golay for noisy diagnostics


def sg(y):
    if len(y) < SG_WIN or not np.all(np.isfinite(y)):
        return y
    return savgol_filter(y, SG_WIN, SG_POLY)


def load(d):
    path = DIAG_DIR / f"geometric_d{d}_diag" / "diag_summary.tsv"
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


# ── helpers: axis decoration ──────────────────────────────────────────────────

GRID_KW = dict(color="#e4e4e4", lw=0.5, zorder=0)


def style_ax(ax):
    ax.tick_params(axis="both", direction="out", which="both",
                   labelsize=9.5, top=False, right=False)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(True, which="major", **GRID_KW)


def add_legend(ax, handles, loc="upper left"):
    ax.legend(handles=handles,
              loc=loc,
              fontsize=8.5,
              frameon=True,
              framealpha=0.85,
              edgecolor="#cccccc",
              handlelength=1.4,
              borderpad=0.6,
              labelspacing=0.25,
              ncol=2)


def make_figure():
    fig = plt.figure(figsize=(6.5, 5.2))
    gs  = fig.add_gridspec(2, 1,
                            hspace=0.06,
                            left=0.14, right=0.96,
                            top=0.96, bottom=0.11)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)
    return fig, ax_top, ax_bot


# ── Figure 1: order parameter m₁² and m₁w² ──────────────────────────────────

fig1, ax_m1, ax_m1w = make_figure()
handles = []

for d in D_LIST:
    try:
        data = load(d)
    except FileNotFoundError:
        print(f"  Skipping d={d}: diag_summary.tsv not found")
        continue

    x     = data["temp_over_lam1"]
    m1    = data["m1_sq"]
    m1w   = data["m1w_sq"]
    c     = COLORS[d]
    lw    = 2.0

    # skip if no valid data (all-NaN from failed d=22 run)
    if not np.any(np.isfinite(m1)):
        continue

    m1_sm  = sg(m1)
    m1w_sm = sg(m1w)

    ax_m1.plot(x, m1,    color=c, lw=0.8, alpha=0.20, zorder=2)
    ax_m1.plot(x, m1_sm, color=c, lw=lw,              zorder=3)

    ax_m1w.plot(x, m1w,    color=c, lw=0.8, alpha=0.20, zorder=2)
    ax_m1w.plot(x, m1w_sm, color=c, lw=lw,              zorder=3)

    h, = ax_m1.plot([], [], color=c, lw=lw, label=f"$d = {d}$")
    handles.append(h)

# theoretical ground-state value m₁²(T*→0) = 4/9 for all d (nested subcube)
ax_m1.axhline(4/9, color="gray", lw=1.0, ls=":", alpha=0.7, zorder=1)
ax_m1.text(XLIM[1]-0.005, 4/9 + 0.004, r"$\frac{4}{9}$",
           ha="right", va="bottom", fontsize=9, color="gray")

for ax in (ax_m1, ax_m1w):
    ax.set_xlim(XLIM)
    style_ax(ax)

ax_m1.set_ylim(-0.01, 0.52)
ax_m1w.set_ylim(bottom=0)
plt.setp(ax_m1.get_xticklabels(), visible=False)

ax_m1.set_ylabel(r"$m_1^2 = \sum_{a,j} M_{aj}^2$",              fontsize=11)
ax_m1w.set_ylabel(r"$m_{1,w}^2 = \sum_{a,j} M_{aj}^2\!/f_a$",   fontsize=11)
ax_m1w.set_xlabel(r"$T^*/\lambda_1$  $(\lambda_1 = d-2)$",        fontsize=11)

for ax, lbl in ((ax_m1, "(a)"), (ax_m1w, "(b)")):
    ax.text(0.017, 0.96, lbl, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top")

add_legend(ax_m1, handles, loc="upper right")

out1 = PLOTS_DIR / "geom_diag_order_parameter"
fig1.savefig(out1.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
fig1.savefig(out1.with_suffix(".png"), dpi=300, bbox_inches="tight")
plt.close(fig1)
print(f"Saved: {out1}.pdf / .png")


# ── Figure 2: correlation length ξ and G(1) ──────────────────────────────────

fig2, ax_xi, ax_G1 = make_figure()
handles2 = []

for d in D_LIST:
    try:
        data = load(d)
    except FileNotFoundError:
        continue

    x    = data["temp_over_lam1"]
    xi   = data["xi"]
    G1   = data.get("G_1", np.full_like(x, np.nan))
    c    = COLORS[d]
    lw   = 2.0

    if not np.any(np.isfinite(xi)):
        continue

    # ξ: only plot finite positive values
    mask_xi = np.isfinite(xi) & (xi > 0)
    if mask_xi.sum() >= 2:
        ax_xi.plot(x[mask_xi], xi[mask_xi], color=c, lw=lw,
                   marker="o", ms=3.5, markeredgewidth=0.0, zorder=3)

    # G(1): smooth
    mask_G = np.isfinite(G1)
    if mask_G.sum() >= SG_WIN:
        G1_sm = sg(G1[mask_G])
        ax_G1.plot(x[mask_G], G1[mask_G], color=c, lw=0.8, alpha=0.20, zorder=2)
        ax_G1.plot(x[mask_G], G1_sm,      color=c, lw=lw,              zorder=3)

    h, = ax_xi.plot([], [], color=c, lw=lw, label=f"$d = {d}$")
    handles2.append(h)

for ax in (ax_xi, ax_G1):
    ax.set_xlim(XLIM)
    style_ax(ax)

ax_xi.set_ylim(bottom=0)
ax_G1.set_ylim(bottom=-0.01)
plt.setp(ax_xi.get_xticklabels(), visible=False)

ax_xi.set_ylabel(r"$\xi\;$ (correlation length)",            fontsize=11)
ax_G1.set_ylabel(r"$G(\ell\!=\!1) = q(1) - q_{\rm rand}$",  fontsize=11)
ax_G1.set_xlabel(r"$T^*/\lambda_1$  $(\lambda_1 = d-2)$",    fontsize=11)

for ax, lbl in ((ax_xi, "(a)"), (ax_G1, "(b)")):
    ax.text(0.017, 0.96, lbl, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top")

add_legend(ax_xi, handles2, loc="upper left")

out2 = PLOTS_DIR / "geom_diag_correlation"
fig2.savefig(out2.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
fig2.savefig(out2.with_suffix(".png"), dpi=300, bbox_inches="tight")
plt.close(fig2)
print(f"Saved: {out2}.pdf / .png")


# ── Figure 3: G(ℓ) profiles  — two panels ────────────────────────────────────
#   (a) one representative d, curves coloured by T*/λ₁  (shows temperature evolution)
#   (b) multiple d at a fixed T*/λ₁ near the transition (shows d-scaling)

D_PROFILE  = 16          # representative d for panel (a)
T_FIXED    = 0.22        # T*/λ₁ for panel (b) cross-section (near transition)
T_TARGETS  = [0.08, 0.16, 0.20, 0.24, 0.30, 0.38]   # temperatures for panel (a)
D_PANEL_B  = [8, 12, 16, 20]

fig3, (ax_pa, ax_pb) = plt.subplots(1, 2, figsize=(9.0, 3.6))
fig3.subplots_adjust(left=0.09, right=0.97, bottom=0.16, top=0.94, wspace=0.32)

# ── panel (a): G(ℓ) vs ℓ for D_PROFILE, several temperatures ────────────────
try:
    data_p = load(D_PROFILE)
    x_all  = data_p["temp_over_lam1"]
    names  = list(data_p.keys())
    G_cols = [n for n in names if n.startswith("G_") and n[2:].isdigit()]
    G_cols = sorted(G_cols, key=lambda n: int(n[2:]))
    G_mat  = np.column_stack([data_p[n] for n in G_cols])   # (n_temps, d)
    ell    = np.arange(1, len(G_cols) + 1)

    t_cmap = plt.cm.coolwarm
    for t_target in T_TARGETS:
        idx = int(np.argmin(np.abs(x_all - t_target)))
        t_ratio = (t_target - T_TARGETS[0]) / (T_TARGETS[-1] - T_TARGETS[0])
        color = t_cmap(t_ratio)
        G_row = G_mat[idx]
        ax_pa.plot(ell, G_row, color=color, lw=1.8,
                   label=f"$T^*/\\lambda_1={x_all[idx]:.2f}$")

    ax_pa.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax_pa.set_xlabel(r"Hamming distance $\ell$", fontsize=11)
    ax_pa.set_ylabel(r"$G(\ell) = q(\ell) - q_{\rm rand}$", fontsize=11)
    ax_pa.set_title(f"$d = {D_PROFILE}$", fontsize=10, pad=4)
    ax_pa.legend(fontsize=7.5, frameon=True, framealpha=0.85,
                 edgecolor="#cccccc", handlelength=1.2, labelspacing=0.22,
                 loc="upper right")
except (FileNotFoundError, KeyError):
    ax_pa.text(0.5, 0.5, f"d={D_PROFILE} data not found",
               ha="center", va="center", transform=ax_pa.transAxes)

# ── panel (b): G(ℓ) vs ℓ for multiple d at T*/λ₁ ≈ T_FIXED ─────────────────
for d in D_PANEL_B:
    try:
        data_b = load(d)
        x_b    = data_b["temp_over_lam1"]
        names_b = list(data_b.keys())
        G_cols_b = sorted([n for n in names_b if n.startswith("G_") and n[2:].isdigit()],
                          key=lambda n: int(n[2:]))
        G_mat_b  = np.column_stack([data_b[n] for n in G_cols_b])
        idx_b    = int(np.argmin(np.abs(x_b - T_FIXED)))
        ell_b    = np.arange(1, len(G_cols_b) + 1)
        c        = COLORS[d]
        ax_pb.plot(ell_b, G_mat_b[idx_b], color=c, lw=1.8, label=f"$d = {d}$")
    except (FileNotFoundError, KeyError):
        pass

ax_pb.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.6)
ax_pb.set_xlabel(r"Hamming distance $\ell$", fontsize=11)
ax_pb.set_ylabel(r"$G(\ell)$", fontsize=11)
ax_pb.set_title(f"$T^*/\\lambda_1 \\approx {T_FIXED}$", fontsize=10, pad=4)
ax_pb.legend(fontsize=8.5, frameon=True, framealpha=0.85,
             edgecolor="#cccccc", handlelength=1.2, labelspacing=0.22,
             loc="upper right")

for ax, lbl in ((ax_pa, "(a)"), (ax_pb, "(b)")):
    style_ax(ax)
    ax.text(0.017, 0.97, lbl, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top")

out3 = PLOTS_DIR / "geom_diag_Gprofile"
fig3.savefig(out3.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
fig3.savefig(out3.with_suffix(".png"), dpi=300, bbox_inches="tight")
plt.close(fig3)
print(f"Saved: {out3}.pdf / .png")

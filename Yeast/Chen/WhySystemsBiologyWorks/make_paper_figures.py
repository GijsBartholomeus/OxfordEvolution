from __future__ import annotations

import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.path import Path as MplPath


PAPER = Path("/Users/gijsbartholomeus/Documents/STUDIE/Papers/WhySystemsBiologyWorks")
FIGURES = PAPER / "Figures"
SOURCE_FREQ = Path(
    "/Users/gijsbartholomeus/Documents/STUDIE/OxfordEvolution/code/Yeast/Chen/"
    "WhySystemsBiologyWorks/plots/figure2_complexity_frequency_chen_plus_six_N=1e3.png"
)
RANDOM_STARTS = np.array(
    [
        [0.18, 0.20],
        [0.44, 0.25],
        [0.72, 0.23],
        [0.86, 0.70],
        [0.22, 0.86],
    ]
)


def add_panel_label(ax, label: str) -> None:
    ax.text(-0.02, 1.03, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=16, fontweight="bold")


def setup_axis(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(1.3)
    ax.spines["bottom"].set_linewidth(1.3)


def draw_localized(ax) -> None:
    setup_axis(ax)
    add_panel_label(ax, "A")
    ax.set_title("Localized neutral set", fontsize=13)

    center = np.array([0.32, 0.68])
    radius = 0.19
    neutral = Circle(center, radius, facecolor="#8ecae6", edgecolor="#1f6f9f", lw=2.2, alpha=0.85)
    ax.add_patch(neutral)
    ax.text(0.32, 0.68, r"$\mathcal{N}(x_{\rm wt})$", ha="center", va="center", fontsize=11)

    for start in RANDOM_STARTS:
        direction = center - start
        norm = max(float(np.linalg.norm(direction)), 1e-12)
        target = center - radius * direction / norm
        ax.scatter([start[0]], [start[1]], marker="x", s=95, lw=2.2, color="#d62728", zorder=5)
        ax.add_patch(
            FancyArrowPatch(
                start,
                target,
                arrowstyle="-|>",
                mutation_scale=13,
                lw=1.8,
                color="#555555",
                linestyle=":",
                connectionstyle="arc3,rad=0.02",
            )
        )


def draw_extended(ax) -> None:
    setup_axis(ax)
    add_panel_label(ax, "B")
    ax.set_title("Extended neutral set", fontsize=13)

    x = np.linspace(0.06, 0.94, 320)
    y = 0.52 + 0.22 * np.sin(2.7 * np.pi * x + 0.2) + 0.08 * np.sin(7.3 * np.pi * x)
    ax.plot(x, y, color="#1f6f9f", lw=13, alpha=0.28, solid_capstyle="round")
    ax.plot(x, y, color="#1f6f9f", lw=2.4, solid_capstyle="round")

    for sx, sy in RANDOM_STARTS:
        idx = int(np.argmin((x - sx) ** 2 + (y - sy) ** 2))
        target = np.array([x[idx], y[idx]])
        ax.scatter([sx], [sy], marker="x", s=80, lw=2.0, color="#d62728", zorder=5)
        arrow = FancyArrowPatch(
            (sx, sy),
            target,
            arrowstyle="-|>",
            mutation_scale=13,
            lw=1.8,
            color="#555555",
            linestyle=":",
            connectionstyle="arc3,rad=0.15",
        )
        ax.add_patch(arrow)

    for tx in np.linspace(0.22, 0.78, 5):
        idx = int(np.argmin((x - tx) ** 2))
        dx = x[min(idx + 4, len(x) - 1)] - x[max(idx - 4, 0)]
        dy = y[min(idx + 4, len(y) - 1)] - y[max(idx - 4, 0)]
        norm = max(float(np.hypot(dx, dy)), 1e-12)
        ux, uy = dx / norm, dy / norm
        ax.add_patch(
            FancyArrowPatch(
                (x[idx] - 0.035 * ux, y[idx] - 0.035 * uy),
                (x[idx] + 0.035 * ux, y[idx] + 0.035 * uy),
                arrowstyle="-|>",
                mutation_scale=11,
                lw=1.6,
                color="#174a68",
            )
        )


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), constrained_layout=True)
    draw_localized(axes[0])
    draw_extended(axes[1])
    stable_png = FIGURES / "NeutralGeometryCartoon.png"
    stable_pdf = FIGURES / "NeutralGeometryCartoon.pdf"
    for path in (stable_png, stable_pdf):
        if path.exists():
            path.unlink()
    fig.savefig(stable_png, dpi=300)
    fig.savefig(stable_pdf)
    preview = FIGURES / f"NeutralGeometryCartoon_preview_{time.strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(preview, dpi=300)

    shutil.copy2(SOURCE_FREQ, FIGURES / "FreqComp.png")


if __name__ == "__main__":
    main()

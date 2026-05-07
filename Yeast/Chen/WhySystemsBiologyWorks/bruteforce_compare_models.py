"""Comparison figures across brute-force cloud analyses."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def plot_accessibility(ax, data: dict) -> None:
    rows = data["radius_growth"]["rows"]
    x = np.asarray([row["radius"] for row in rows], dtype=float)
    y = np.asarray([row["unique_phenotypes"]["median"] for row in rows], dtype=float)
    y05 = np.asarray([row["unique_phenotypes"]["q05"] for row in rows], dtype=float)
    y95 = np.asarray([row["unique_phenotypes"]["q95"] for row in rows], dtype=float)

    ax.fill_between(x, y05, y95, color="#9ecae1", alpha=0.35, linewidth=0)
    ax.plot(x, y, color="black", linewidth=2.2)
    ax.set_yscale("log")
    ax.set_xlabel("radius in normalized cube")
    ax.set_ylabel("unique phenotypes")
    ax.set_title(f"{data['label']} accessibility")
    ax.text(
        0.02,
        0.96,
        f"centers={data['radius_growth']['n_centers']:,}\ncloud n={data['radius_growth']['n_points_in_cloud']:,}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
    )
    ax.grid(True, axis="y", alpha=0.2)


def plot_pairwise(ax, data: dict) -> None:
    groups = data["pairwise_distance_distributions_normalized_cube"]
    stats = []
    max_x = []
    max_y = []
    for i, (name, values) in enumerate(groups.items(), start=1):
        n_points = values.get("equalized_n_points", values.get("n_points"))
        stats.append(
            {
                "label": f"{name}\n(n={n_points:,})",
                "whislo": values["q05"],
                "q1": values["q25"],
                "med": values["median"],
                "q3": values["q75"],
                "whishi": values["q95"],
                "fliers": [],
            }
        )
        max_x.append(i)
        max_y.append(values["max"])

    ax.bxp(stats, showfliers=False, patch_artist=True, widths=0.55)
    for patch in ax.patches:
        patch.set_facecolor("#d8e8f7")
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)
    ax.scatter(max_x, max_y, color="#c44e52", s=28, zorder=3, label="max")
    ax.set_ylabel("Euclidean distance in normalized cube")
    ax.set_title(f"{data['label']} pairwise distances")
    ax.tick_params(axis="x", labelrotation=0)
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(frameon=False, loc="upper right")


def main() -> None:
    chen = load_json(
        ROOT
        / "results_summaries/bruteforce_cloud/chen_bfc_1e8/locality/chen2004_locality_chen_bfc_1e8.json"
    )
    tyson = load_json(
        ROOT
        / "results_summaries/bruteforce_cloud/tyson_bfc_1e8/locality/tyson1991_locality_tyson_bfc_1e8.json"
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), constrained_layout=True)
    plot_accessibility(axes[0, 0], chen)
    plot_accessibility(axes[0, 1], tyson)
    plot_pairwise(axes[1, 0], chen)
    plot_pairwise(axes[1, 1], tyson)

    out = ROOT / "figures/bruteforce_cloud/accessibility_pairwise_chen_tyson_2x2.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()

"""Observed versus shuffled phenotype accessibility in brute-force clouds.

The shuffled null keeps the same parameter points and the same phenotype-code
frequency distribution, but randomly permutes phenotype labels over points.
"""

from __future__ import annotations

import json
import math
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parent
STATS_ROOT = ROOT / "results/bruteforce_cloud_stats"
SUMMARY_ROOT = ROOT / "results_summaries/bruteforce_cloud"


MODELS = [
    {
        "model": "chen2004",
        "label": "Chen 2004",
        "tag": "chen_bfc_1e8",
        "sample": STATS_ROOT / "chen_bfc_1e8/chen2004_bruteforce_samples_chen_bfc_1e8.npz",
        "locality_json": SUMMARY_ROOT / "chen_bfc_1e8/locality/chen2004_locality_chen_bfc_1e8.json",
    },
    {
        "model": "tyson1991",
        "label": "Tyson 1991",
        "tag": "tyson_bfc_1e8",
        "sample": STATS_ROOT / "tyson_bfc_1e8/tyson1991_bruteforce_samples_tyson_bfc_1e8.npz",
        "locality_json": SUMMARY_ROOT / "tyson_bfc_1e8/locality/tyson1991_locality_tyson_bfc_1e8.json",
    },
]


def summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    return {
        "n": int(values.size),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "mean": float(np.mean(values)),
    }


def load_cloud(path: Path, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    data = np.load(path, allow_pickle=True)
    points = np.asarray(data["all_points"], dtype=float)
    codes = np.asarray(data["all_phenotype_codes"])
    p0 = np.asarray(data["p0"], dtype=float)
    mask = np.all(np.isfinite(points), axis=1)
    points = points[mask]
    codes = codes[mask]
    points = points / np.maximum(2.0 * p0[None, :], np.finfo(float).tiny)
    if len(points) > max_points:
        idx = rng.choice(len(points), size=max_points, replace=False)
        points = points[idx]
        codes = codes[idx]
    return points, codes


def compute_growth(
    points: np.ndarray,
    codes: np.ndarray,
    radii: np.ndarray,
    n_centers: int,
    n_shuffles: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    tree = cKDTree(points)
    center_idx = rng.choice(len(points), size=min(n_centers, len(points)), replace=False)
    centers = points[center_idx]
    shuffled_codes = [rng.permutation(codes) for _ in range(n_shuffles)]
    total_unique = int(len(np.unique(codes)))

    rows = []
    for radius in radii:
        neighborhoods = tree.query_ball_point(centers, r=float(radius))
        observed = np.asarray([len(np.unique(codes[idx])) for idx in neighborhoods], dtype=float)
        shuffled = np.asarray(
            [
                [len(np.unique(shuf[idx])) for idx in neighborhoods]
                for shuf in shuffled_codes
            ],
            dtype=float,
        )
        shuffled_by_center = np.mean(shuffled, axis=0)
        rows.append(
            {
                "radius": float(radius),
                "observed_unique_phenotypes": summary(observed),
                "shuffled_unique_phenotypes": summary(shuffled_by_center),
                "observed_over_shuffled_median": float(
                    np.median(observed) / max(np.median(shuffled_by_center), np.finfo(float).tiny)
                ),
            }
        )
    return {
        "n_points": int(len(points)),
        "dimensions": int(points.shape[1]),
        "n_centers": int(len(centers)),
        "n_shuffles": int(n_shuffles),
        "total_unique_phenotypes": total_unique,
        "rows": rows,
    }


def plot_panel(ax, result: dict, label: str) -> None:
    rows = result["rows"]
    x = np.asarray([row["radius"] for row in rows])
    obs = np.asarray([row["observed_unique_phenotypes"]["median"] for row in rows])
    obs05 = np.asarray([row["observed_unique_phenotypes"]["q05"] for row in rows])
    obs95 = np.asarray([row["observed_unique_phenotypes"]["q95"] for row in rows])
    shuf = np.asarray([row["shuffled_unique_phenotypes"]["median"] for row in rows])
    shuf05 = np.asarray([row["shuffled_unique_phenotypes"]["q05"] for row in rows])
    shuf95 = np.asarray([row["shuffled_unique_phenotypes"]["q95"] for row in rows])

    ax.fill_between(x, obs05, obs95, color="black", alpha=0.10, linewidth=0)
    ax.plot(x, obs, color="black", linewidth=2.2, label="observed")
    ax.fill_between(x, shuf05, shuf95, color="#d55e00", alpha=0.16, linewidth=0)
    ax.plot(x, shuf, color="#d55e00", linestyle="--", linewidth=2.0, label="shuffled labels")
    ax.set_yscale("log")
    ax.set_xlabel("radius in normalized cube")
    ax.set_ylabel("unique phenotypes")
    ax.set_title(label)
    ax.grid(True, axis="y", alpha=0.2)
    ax.text(
        0.02,
        0.96,
        f"d={result['dimensions']}\npoints={result['n_points']:,}\ncenters={result['n_centers']:,}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )


def plot_ratio(ax, result: dict, label: str) -> None:
    rows = result["rows"]
    x = np.asarray([row["radius"] for row in rows])
    ratio = np.asarray([row["observed_over_shuffled_median"] for row in rows])
    ax.axhline(1.0, color="0.6", linewidth=1.0)
    ax.plot(x, ratio, color="#0072b2", linewidth=2.2)
    ax.set_xlabel("radius in normalized cube")
    ax.set_ylabel("observed / shuffled")
    ax.set_title(f"{label} enrichment")
    ax.grid(True, axis="y", alpha=0.2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Observed versus shuffled phenotype accessibility")
    parser.add_argument("--max-points", type=int, default=20_000)
    parser.add_argument("--centers", type=int, default=250)
    parser.add_argument("--shuffles", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    outputs = {}
    for model_index, config in enumerate(MODELS):
        local = json.loads(config["locality_json"].read_text())
        radii = np.asarray([row["radius"] for row in local["radius_growth"]["rows"]], dtype=float)
        points, codes = load_cloud(config["sample"], max_points=args.max_points, seed=args.seed + model_index)
        outputs[config["model"]] = compute_growth(
            points,
            codes,
            radii=radii,
            n_centers=args.centers,
            n_shuffles=args.shuffles,
            seed=args.seed + 100 + model_index,
        )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for col, config in enumerate(MODELS):
        result = outputs[config["model"]]
        plot_panel(axes[0, col], result, config["label"])
        plot_ratio(axes[1, col], result, config["label"])
    axes[0, 0].legend(frameon=False, loc="lower right")

    fig_dir = ROOT / "figures/bruteforce_cloud"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "accessibility_observed_vs_shuffled_chen_tyson_2x2.png"
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)

    summary_dir = ROOT / "results_summaries/bruteforce_cloud/model_comparisons"
    summary_dir.mkdir(parents=True, exist_ok=True)
    json_path = summary_dir / "accessibility_observed_vs_shuffled_chen_tyson_2x2.json"
    json_path.write_text(json.dumps(outputs, indent=2))
    print(fig_path)
    print(json_path)


if __name__ == "__main__":
    main()

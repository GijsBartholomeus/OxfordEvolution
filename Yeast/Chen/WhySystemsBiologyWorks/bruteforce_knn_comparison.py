from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
SUMMARY_ROOT = ROOT / "results_summaries" / "bruteforce_cloud"
FIGURE_ROOT = ROOT / "figures" / "bruteforce_cloud" / "model_comparisons"


MODELS = [
    (
        "chen2004",
        "chen_bfc_1e8",
        "Chen 2004",
        SUMMARY_ROOT / "chen_bfc_1e8" / "knn_locality" / "chen2004_knn_phenotype_locality_chen_bfc_1e8.json",
    ),
    (
        "tyson1991",
        "tyson_bfc_1e8",
        "Tyson 1991",
        SUMMARY_ROOT / "tyson_bfc_1e8" / "knn_locality" / "tyson1991_knn_phenotype_locality_tyson_bfc_1e8.json",
    ),
]


def stat_series(result: dict, key: str, stat: str = "mean") -> np.ndarray:
    return np.asarray([row[key].get(stat, np.nan) for row in result["rows"]], dtype=float)


def scalar_series(result: dict, key: str) -> np.ndarray:
    return np.asarray([row.get(key, np.nan) for row in result["rows"]], dtype=float)


def draw_distance_summary(ax: plt.Axes, result: dict) -> None:
    meta = result["within_between_distance"]
    labels = ["same", "different"]
    colors = ["#111111", "#d55e00"]
    for i, key in enumerate(["same_pairs", "different_pairs"], start=1):
        stats = meta[key]
        ax.vlines(i, stats["q05"], stats["q95"], color=colors[i - 1], lw=2)
        ax.vlines(i, stats["q25"], stats["q75"], color=colors[i - 1], lw=8, alpha=0.25)
        ax.scatter([i], [stats["median"]], color=colors[i - 1], s=35, zorder=3)
        ax.scatter([i], [stats["mean"]], color=colors[i - 1], s=35, marker="D", zorder=3)
    delta = meta.get("same_minus_different_mean")
    if delta is not None:
        ax.text(
            0.5,
            0.96,
            f"mean same - different = {delta:.3g}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
        )
    ax.set_xticks([1, 2], labels)
    ax.set_xlim(0.5, 2.5)
    ax.grid(True, axis="y", alpha=0.25)


def plot_comparison(results: dict[str, dict], out_path: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(12.5, 14.5), constrained_layout=True)
    fig.suptitle("k-nearest-neighbor phenotype locality", fontsize=16)

    for col, (_, _, title, _) in enumerate(MODELS):
        result = results[title]
        m = np.asarray(result["m_values"], dtype=float)
        header = (
            f"{title}\n"
            f"d={result['dimensions']}, n={result['points_used']:,}, "
            f"phenotypes={result['unique_phenotypes']:,}"
        )

        ax = axes[0, col]
        obs = stat_series(result, "observed_unique_phenotypes")
        obs_q25 = stat_series(result, "observed_unique_phenotypes", "q25")
        obs_q75 = stat_series(result, "observed_unique_phenotypes", "q75")
        shuf = stat_series(result, "shuffled_unique_phenotypes_mean_over_reps")
        theory = np.asarray([row["theoretical_random_label_expected_unique"] for row in result["rows"]], dtype=float)
        ax.plot(m, obs, color="black", lw=2.2, label="observed")
        ax.fill_between(m, obs_q25, obs_q75, color="#9ecae1", alpha=0.35, lw=0, label="observed IQR")
        ax.plot(m, shuf, color="#d55e00", lw=2, ls="--", label="shuffled")
        ax.plot(m, theory, color="#0072b2", lw=1.5, ls=":", label="frequency null")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(header)
        ax.set_ylabel("unique phenotypes")
        ax.grid(True, alpha=0.25)
        if col == 1:
            ax.legend(frameon=False, fontsize=8)

        ax = axes[1, col]
        obs_eff = stat_series(result, "observed_effective_phenotypes")
        obs_eff_q25 = stat_series(result, "observed_effective_phenotypes", "q25")
        obs_eff_q75 = stat_series(result, "observed_effective_phenotypes", "q75")
        shuf_eff = stat_series(result, "shuffled_effective_phenotypes_mean_over_reps")
        ax.plot(m, obs_eff, color="black", lw=2.2, label="observed")
        ax.fill_between(m, obs_eff_q25, obs_eff_q75, color="#9ecae1", alpha=0.35, lw=0)
        if np.any(np.isfinite(shuf_eff)):
            ax.plot(m, shuf_eff, color="#d55e00", lw=2, ls="--", label="shuffled")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel("effective phenotypes exp(H)")
        ax.grid(True, alpha=0.25)

        ax = axes[2, col]
        obs_enrich = scalar_series(result, "observed_same_phenotype_enrichment")
        shuf_enrich = scalar_series(result, "shuffled_same_phenotype_enrichment_mean")
        ax.axhline(1.0, color="0.45", lw=1.2, label="random baseline")
        ax.plot(m, obs_enrich, color="black", lw=2.2, label="observed")
        ax.plot(m, shuf_enrich, color="#d55e00", lw=2, ls="--", label="shuffled")
        ax.set_xscale("log")
        ax.set_ylabel("same-phenotype enrichment")
        ax.grid(True, alpha=0.25)

        ax = axes[3, col]
        draw_distance_summary(ax, result)
        ax.set_ylabel("pairwise distance")
        ax.set_xlabel("pair type")

    for ax in axes.flat[:6]:
        ax.set_xlabel("neighborhood size m")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    results = {}
    summary = {}
    for _model, _tag, title, path in MODELS:
        result = json.loads(path.read_text())
        results[title] = result
        summary[title] = {
            "path": str(path),
            "points_used": result["points_used"],
            "dimensions": result["dimensions"],
            "unique_phenotypes": result["unique_phenotypes"],
            "same_random_baseline_sum_p2": result["same_random_baseline_sum_p2"],
            "selected_rows": {
                str(row["m"]): {
                    "observed_unique_mean": row["observed_unique_phenotypes"]["mean"],
                    "shuffled_unique_mean": row["shuffled_unique_phenotypes_mean_over_reps"]["mean"],
                    "observed_same_phenotype_enrichment": row["observed_same_phenotype_enrichment"],
                    "shuffled_same_phenotype_enrichment": row["shuffled_same_phenotype_enrichment_mean"],
                }
                for row in result["rows"]
                if row["m"] in {10, 100, 1000, 5000}
            },
            "within_between_distance": result["within_between_distance"],
        }

    out_path = FIGURE_ROOT / "knn_phenotype_locality_chen_tyson.png"
    json_path = SUMMARY_ROOT / "model_comparisons" / "knn_phenotype_locality_chen_tyson.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    plot_comparison(results, out_path)
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved {out_path}")
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()

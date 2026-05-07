from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from wsbw_pipeline import RESULTS, SPECS


STATS_ROOT = RESULTS / "bruteforce_cloud_stats"
SUMMARY_ROOT = ROOT / "results_summaries" / "bruteforce_cloud"
FIGURE_ROOT = ROOT / "figures" / "bruteforce_cloud"


def spec_for(model: str):
    for spec in SPECS:
        if spec.key == model:
            return spec
    raise KeyError(model)


def load_sample_npz(stats_dir: Path, model: str, tag: str) -> dict[str, np.ndarray]:
    path = stats_dir / f"{model}_bruteforce_samples_{tag}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    required = ["all_points", "all_phenotype_codes", "p0"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise KeyError(f"{path} is missing required arrays: {missing}")
    return {key: np.asarray(data[key]) for key in data.files if key in required}


def normalize_points(points: np.ndarray, p0: np.ndarray) -> np.ndarray:
    scale = np.maximum(2.0 * np.asarray(p0, dtype=float), np.finfo(float).tiny)
    return np.asarray(points, dtype=float) / scale[None, :]


def finite_rows(points: np.ndarray, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.all(np.isfinite(points), axis=1)
    return points[mask], codes[mask]


def encode_labels(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique, labels, counts = np.unique(codes, return_inverse=True, return_counts=True)
    return unique, labels.astype(np.int64), counts.astype(np.int64)


def summary_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"n": 0}
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def expected_unique_from_frequencies(counts: np.ndarray, m_values: np.ndarray) -> np.ndarray:
    p = counts.astype(float) / float(np.sum(counts))
    out = []
    for m in m_values:
        out.append(float(np.sum(1.0 - np.power(1.0 - p, int(m)))))
    return np.asarray(out, dtype=float)


def effective_phenotypes(labels: np.ndarray) -> float:
    counts = np.bincount(labels).astype(float)
    counts = counts[counts > 0]
    probs = counts / np.sum(counts)
    entropy = -float(np.sum(probs * np.log(probs)))
    return float(np.exp(entropy))


def unique_counts_by_row(window: np.ndarray) -> np.ndarray:
    sorted_window = np.sort(window, axis=1)
    if sorted_window.shape[1] == 0:
        return np.zeros(sorted_window.shape[0], dtype=float)
    return (1 + np.sum(np.diff(sorted_window, axis=1) != 0, axis=1)).astype(float)


def effective_counts_by_row(window: np.ndarray) -> np.ndarray:
    return np.asarray([effective_phenotypes(row) for row in window], dtype=float)


def neighborhood_metrics(
    neighbor_labels: np.ndarray,
    center_labels: np.ndarray,
    m_values: list[int],
    compute_effective: bool = True,
) -> dict[str, np.ndarray]:
    n_centers = neighbor_labels.shape[0]
    unique_counts = np.empty((len(m_values), n_centers), dtype=float)
    eff_counts = np.full((len(m_values), n_centers), np.nan, dtype=float)
    purity = np.full((len(m_values), n_centers), np.nan, dtype=float)

    for mi, m in enumerate(m_values):
        window = neighbor_labels[:, :m]
        unique_counts[mi] = unique_counts_by_row(window)
        if compute_effective:
            eff_counts[mi] = effective_counts_by_row(window)
        if m > 1:
            non_self = neighbor_labels[:, 1:m]
            purity[mi] = np.mean(non_self == center_labels[:, None], axis=1)

    return {
        "unique": unique_counts,
        "effective": eff_counts,
        "purity": purity,
    }


def pair_distance_distributions(
    points: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    n_pairs: int,
    min_label_count: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    label_to_indices: dict[int, np.ndarray] = {}
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        if len(idx) >= min_label_count:
            label_to_indices[int(label)] = idx

    if not label_to_indices:
        return np.empty(0), np.empty(0), {"same_pairs": {"n": 0}, "different_pairs": {"n": 0}}

    label_keys = np.asarray(list(label_to_indices.keys()), dtype=int)
    label_weights = np.asarray([len(label_to_indices[int(label)]) for label in label_keys], dtype=float)
    label_weights /= np.sum(label_weights)

    same_distances = []
    attempts = 0
    while len(same_distances) < n_pairs and attempts < 20 * n_pairs:
        attempts += 1
        label = int(rng.choice(label_keys, p=label_weights))
        idx = rng.choice(label_to_indices[label], size=2, replace=False)
        same_distances.append(float(np.linalg.norm(points[idx[0]] - points[idx[1]])))

    different_distances = []
    n = len(points)
    attempts = 0
    while len(different_distances) < n_pairs and attempts < 20 * n_pairs:
        attempts += 1
        i, j = rng.choice(n, size=2, replace=False)
        if labels[i] == labels[j]:
            continue
        different_distances.append(float(np.linalg.norm(points[i] - points[j])))

    same = np.asarray(same_distances, dtype=float)
    different = np.asarray(different_distances, dtype=float)
    meta = {
        "same_pairs": summary_stats(same),
        "different_pairs": summary_stats(different),
        "same_minus_different_mean": float(np.mean(same) - np.mean(different)) if len(same) and len(different) else None,
        "labels_eligible_for_same_pairs": int(len(label_keys)),
        "min_label_count": int(min_label_count),
    }
    return same, different, meta


def analyze(args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)
    stats_dir = STATS_ROOT / args.tag
    sample = load_sample_npz(stats_dir, args.model, args.tag)
    points, codes = finite_rows(normalize_points(sample["all_points"], sample["p0"]), sample["all_phenotype_codes"])

    if len(points) > args.max_points:
        idx = rng.choice(len(points), size=args.max_points, replace=False)
        points = points[idx]
        codes = codes[idx]

    unique_codes, labels, counts = encode_labels(codes)
    n_labels = len(unique_codes)
    frequencies = counts / np.sum(counts)
    same_random_baseline = float(np.sum(frequencies**2))

    m_values = [m for m in args.m_values if 1 <= m <= len(points)]
    max_m = max(m_values)
    n_centers = min(args.centers, len(points))
    center_idx = rng.choice(len(points), size=n_centers, replace=False)
    tree = cKDTree(points)
    _, neighbor_idx = tree.query(points[center_idx], k=max_m, workers=args.workers)
    if neighbor_idx.ndim == 1:
        neighbor_idx = neighbor_idx[:, None]
    neighbor_labels = labels[neighbor_idx]
    center_labels = labels[center_idx]

    observed = neighborhood_metrics(neighbor_labels, center_labels, m_values, compute_effective=True)
    shuffled_reps = []
    for _ in range(args.shuffles):
        shuffled_labels_all = rng.permutation(labels)
        shuffled_neighbor_labels = shuffled_labels_all[neighbor_idx]
        shuffled_center_labels = shuffled_labels_all[center_idx]
        shuffled_reps.append(neighborhood_metrics(shuffled_neighbor_labels, shuffled_center_labels, m_values, compute_effective=args.shuffle_effective))

    expected_unique = expected_unique_from_frequencies(counts, np.asarray(m_values, dtype=int))

    rows = []
    for mi, m in enumerate(m_values):
        obs_unique = observed["unique"][mi]
        obs_eff = observed["effective"][mi]
        obs_purity = observed["purity"][mi]
        shuf_unique_mean = np.asarray([rep["unique"][mi].mean() for rep in shuffled_reps], dtype=float)
        if args.shuffle_effective:
            shuf_eff_mean = np.asarray([np.nanmean(rep["effective"][mi]) for rep in shuffled_reps], dtype=float)
        else:
            shuf_eff_mean = np.asarray([], dtype=float)
        if m > 1:
            shuf_purity_mean = np.asarray([np.nanmean(rep["purity"][mi]) for rep in shuffled_reps], dtype=float)
        else:
            shuf_purity_mean = np.asarray([], dtype=float)
        purity_mean = float(np.nanmean(obs_purity)) if np.any(np.isfinite(obs_purity)) else None
        rows.append(
            {
                "m": int(m),
                "observed_unique_phenotypes": summary_stats(obs_unique),
                "observed_effective_phenotypes": summary_stats(obs_eff),
                "observed_same_phenotype_purity": summary_stats(obs_purity),
                "observed_same_phenotype_enrichment": (purity_mean / same_random_baseline) if purity_mean is not None and same_random_baseline else None,
                "shuffled_unique_phenotypes_mean_over_reps": summary_stats(shuf_unique_mean),
                "shuffled_effective_phenotypes_mean_over_reps": summary_stats(shuf_eff_mean),
                "shuffled_same_phenotype_purity_mean_over_reps": summary_stats(shuf_purity_mean),
                "shuffled_same_phenotype_enrichment_mean": float(np.nanmean(shuf_purity_mean) / same_random_baseline) if len(shuf_purity_mean) and same_random_baseline else None,
                "theoretical_random_label_expected_unique": float(expected_unique[mi]),
            }
        )

    same_dist, different_dist, pair_meta = pair_distance_distributions(
        points,
        labels,
        rng,
        args.distance_pairs,
        args.min_same_label_count,
    )

    result = {
        "model": args.model,
        "label": spec_for(args.model).label,
        "tag": args.tag,
        "dimensions": int(points.shape[1]),
        "points_used": int(len(points)),
        "centers": int(n_centers),
        "unique_phenotypes": int(n_labels),
        "same_random_baseline_sum_p2": same_random_baseline,
        "m_values": [int(m) for m in m_values],
        "shuffles": int(args.shuffles),
        "rows": rows,
        "within_between_distance": pair_meta,
    }
    return result, observed, shuffled_reps, same_dist, different_dist


def row_series(result: dict, key: str, stat: str = "mean") -> np.ndarray:
    return np.asarray([row[key].get(stat, np.nan) for row in result["rows"]], dtype=float)


def plot_result(
    result: dict,
    observed: dict[str, np.ndarray],
    shuffled_reps: list[dict[str, np.ndarray]],
    same_dist: np.ndarray,
    different_dist: np.ndarray,
    out_path: Path,
) -> None:
    m = np.asarray(result["m_values"], dtype=float)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    fig.suptitle(
        f"{result['label']} kNN phenotype locality "
        f"(n={result['points_used']:,}, centers={result['centers']:,}, d={result['dimensions']})",
        fontsize=15,
    )

    ax = axes[0, 0]
    obs_mean = row_series(result, "observed_unique_phenotypes")
    obs_q25 = row_series(result, "observed_unique_phenotypes", "q25")
    obs_q75 = row_series(result, "observed_unique_phenotypes", "q75")
    shuf_mean = row_series(result, "shuffled_unique_phenotypes_mean_over_reps")
    theory = np.asarray([row["theoretical_random_label_expected_unique"] for row in result["rows"]], dtype=float)
    ax.plot(m, obs_mean, color="black", lw=2, label="observed")
    ax.fill_between(m, obs_q25, obs_q75, color="#9ecae1", alpha=0.45, label="observed IQR")
    ax.plot(m, shuf_mean, color="#d55e00", lw=2, ls="--", label="shuffled")
    ax.plot(m, theory, color="#0072b2", lw=1.5, ls=":", label="random-label expectation")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("neighborhood size m")
    ax.set_ylabel("unique phenotypes")
    ax.set_title("Richness among m nearest genotypes")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    obs_eff = row_series(result, "observed_effective_phenotypes")
    obs_eff_q25 = row_series(result, "observed_effective_phenotypes", "q25")
    obs_eff_q75 = row_series(result, "observed_effective_phenotypes", "q75")
    shuf_eff = row_series(result, "shuffled_effective_phenotypes_mean_over_reps")
    ax.plot(m, obs_eff, color="black", lw=2, label="observed")
    ax.fill_between(m, obs_eff_q25, obs_eff_q75, color="#9ecae1", alpha=0.45)
    if np.any(np.isfinite(shuf_eff)):
        ax.plot(m, shuf_eff, color="#d55e00", lw=2, ls="--", label="shuffled")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("neighborhood size m")
    ax.set_ylabel("effective phenotypes exp(H)")
    ax.set_title("Entropy-weighted diversity")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 2]
    baseline = result["same_random_baseline_sum_p2"]
    obs_enrich = np.asarray([row["observed_same_phenotype_enrichment"] for row in result["rows"]], dtype=float)
    shuf_enrich = np.asarray([row["shuffled_same_phenotype_enrichment_mean"] for row in result["rows"]], dtype=float)
    ax.axhline(1.0, color="0.5", lw=1, label="random baseline")
    ax.plot(m, obs_enrich, color="black", lw=2, label="observed")
    ax.plot(m, shuf_enrich, color="#d55e00", lw=2, ls="--", label="shuffled")
    ax.set_xscale("log")
    ax.set_xlabel("neighborhood size m")
    ax.set_ylabel("same-phenotype enrichment")
    ax.set_title(r"$P(\phi_j=\phi_i)/\sum_\phi p_\phi^2$")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    obs_purity = row_series(result, "observed_same_phenotype_purity")
    shuf_purity = row_series(result, "shuffled_same_phenotype_purity_mean_over_reps")
    ax.axhline(baseline, color="0.5", lw=1, label=r"$\sum p_\phi^2$")
    ax.plot(m, obs_purity, color="black", lw=2, label="observed")
    ax.plot(m, shuf_purity, color="#d55e00", lw=2, ls="--", label="shuffled")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("neighborhood size m")
    ax.set_ylabel("same-phenotype purity")
    ax.set_title("Neighbor purity")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    if len(same_dist) and len(different_dist):
        bins = np.linspace(min(np.min(same_dist), np.min(different_dist)), max(np.max(same_dist), np.max(different_dist)), 120)
        for values, label, color in [
            (same_dist, "same phenotype", "black"),
            (different_dist, "different phenotype", "#d55e00"),
        ]:
            sorted_values = np.sort(values)
            y = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
            ax.plot(sorted_values, y, color=color, lw=2, label=label)
        ax.set_xlabel("pairwise distance in normalized cube")
        ax.set_ylabel("CDF")
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.text(0.5, 0.5, "Not enough repeated phenotypes\nfor same-phenotype pairs", ha="center", va="center")
    ax.set_title("Within vs between phenotype distances")

    ax = axes[1, 2]
    text = [
        f"unique phenotypes: {result['unique_phenotypes']:,}",
        f"random same baseline: {baseline:.3g}",
        f"same-pair mean minus different-pair mean:",
        str(result["within_between_distance"].get("same_minus_different_mean")),
    ]
    for row in result["rows"]:
        if row["m"] in [10, 100, 1000, 5000, 10000]:
            text.append(
                f"m={row['m']:,}: Uobs={row['observed_unique_phenotypes']['mean']:.1f}, "
                f"Ushuf={row['shuffled_unique_phenotypes_mean_over_reps']['mean']:.1f}, "
                f"E={row['observed_same_phenotype_enrichment']}"
            )
    ax.axis("off")
    ax.text(0.02, 0.98, "\n".join(text), ha="left", va="top", fontsize=10)
    ax.set_title("Summary")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def write_outputs(args: argparse.Namespace) -> tuple[Path, Path]:
    result, observed, shuffled_reps, same_dist, different_dist = analyze(args)
    summary_dir = SUMMARY_ROOT / args.tag / "knn_locality"
    figure_dir = FIGURE_ROOT / args.tag / "knn_locality"
    summary_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    json_path = summary_dir / f"{args.model}_knn_phenotype_locality_{args.tag}.json"
    png_path = figure_dir / f"{args.model}_knn_phenotype_locality_{args.tag}.png"
    json_path.write_text(json.dumps(result, indent=2))
    plot_result(result, observed, shuffled_reps, same_dist, different_dist, png_path)
    return json_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser(description="kNN phenotype locality tests for brute-force genotype clouds")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--model", required=True, choices=[spec.key for spec in SPECS])
    parser.add_argument("--max-points", type=int, default=50000)
    parser.add_argument("--centers", type=int, default=5000)
    parser.add_argument("--m-values", type=int, nargs="+", default=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000])
    parser.add_argument("--shuffles", type=int, default=10)
    parser.add_argument("--shuffle-effective", action="store_true")
    parser.add_argument("--distance-pairs", type=int, default=50000)
    parser.add_argument("--min-same-label-count", type=int, default=2)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    json_path, png_path = write_outputs(args)
    print(f"Saved {json_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()

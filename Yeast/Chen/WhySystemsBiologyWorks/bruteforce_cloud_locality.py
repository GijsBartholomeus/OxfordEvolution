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
from scipy.spatial.distance import pdist

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from wsbw_pipeline import RESULTS, SPECS


STATS_ROOT = RESULTS / "bruteforce_cloud_stats"
PALETTE = np.asarray(["#3b4cc0", "#78b7ff", "#f6c85f", "#c44e52"])


def spec_for(model: str):
    for spec in SPECS:
        if spec.key == model:
            return spec
    raise KeyError(model)


def load_summary(stats_dir: Path, model: str, tag: str) -> dict:
    path = stats_dir / f"{model}_bruteforce_summary_{tag}.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def load_sample_npz(stats_dir: Path, model: str, tag: str) -> dict[str, np.ndarray]:
    path = stats_dir / f"{model}_bruteforce_samples_{tag}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    out: dict[str, np.ndarray] = {}
    for key in [
        "all_points",
        "all_complexities",
        "all_objectives",
        "neutral_points",
        "neutral_complexities",
        "neutral_objectives",
        "all_phenotype_codes",
        "neutral_phenotype_codes",
        "wt_phenotype_points",
        "wt_phenotype_complexities",
        "wt_phenotype_objectives",
        "wt_phenotype_codes",
        "p0",
        "parameter_names",
    ]:
        if key in data.files:
            out[key] = np.asarray(data[key])
    return out


def normalize_points(points: np.ndarray, p0: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=float) / np.maximum(2.0 * np.asarray(p0, dtype=float)[None, :], np.finfo(float).tiny)


def finite_rows(points: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    mask = np.all(np.isfinite(points), axis=1)
    for arr in arrays:
        mask &= np.isfinite(arr)
    return (points[mask], *(arr[mask] for arr in arrays), mask)


def quartile_labels(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    edges = np.quantile(values[np.isfinite(values)], [0.25, 0.5, 0.75])
    labels = np.digitize(values, edges, right=True)
    return labels.astype(int), edges


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


def pairwise_distribution(points: np.ndarray, rng: np.random.Generator, max_points: int) -> tuple[np.ndarray, dict]:
    if len(points) < 2:
        return np.empty(0), {"n_points": int(len(points)), "n_distances": 0}
    if len(points) > max_points:
        idx = rng.choice(len(points), size=max_points, replace=False)
        points = points[idx]
    distances = pdist(points, metric="euclidean")
    stats = summary_stats(distances)
    stats["n_points"] = int(len(points))
    stats["n_distances"] = int(len(distances))
    return distances, stats


def finite_point_rows(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        return np.empty((0, 0), dtype=float)
    return points[np.all(np.isfinite(points), axis=1)]


def equalized_pairwise_groups(
    candidates: dict[str, np.ndarray],
    rng: np.random.Generator,
    max_points: int,
    min_points: int,
    equal_n: int | None,
) -> tuple[dict[str, tuple[np.ndarray, dict]], dict]:
    filtered = {
        name: finite_point_rows(points)
        for name, points in candidates.items()
        if len(finite_point_rows(points)) >= min_points
    }
    available = {name: int(len(finite_point_rows(points))) for name, points in candidates.items()}
    if not filtered:
        return {}, {
            "available_points_before_filter": available,
            "min_points_required": int(min_points),
            "equal_n_points": 0,
        }
    inferred_n = min(len(points) for points in filtered.values())
    if equal_n is not None and equal_n > 0:
        inferred_n = min(inferred_n, int(equal_n))
    inferred_n = min(inferred_n, int(max_points))
    out: dict[str, tuple[np.ndarray, dict]] = {}
    for name, points in filtered.items():
        distances, stats = pairwise_distribution(points, rng, inferred_n)
        stats["n_points_available"] = int(len(points))
        stats["equalized_n_points"] = int(inferred_n)
        out[name] = (distances, stats)
    meta = {
        "available_points_before_filter": available,
        "included_groups": list(out),
        "min_points_required": int(min_points),
        "equal_n_points": int(inferred_n),
        "requested_equal_n_points": int(equal_n) if equal_n is not None else None,
        "max_pairwise_points": int(max_points),
    }
    return out, meta


def add_random_pairwise_controls(
    candidates: dict[str, np.ndarray],
    points: np.ndarray,
    n_controls: int,
) -> None:
    if n_controls <= 1:
        candidates["all random"] = points
        return
    for idx in range(n_controls):
        candidates[f"random cube control {idx + 1}"] = points


def local_quartile_stats(
    points: np.ndarray,
    complexities: np.ndarray,
    rng: np.random.Generator,
    max_points: int,
    k_values: list[int],
    n_shuffles: int,
) -> tuple[dict, np.ndarray, np.ndarray]:
    if len(points) > max_points:
        idx = rng.choice(len(points), size=max_points, replace=False)
        points = points[idx]
        complexities = complexities[idx]

    labels, edges = quartile_labels(complexities)
    max_k = min(max(k_values), len(points) - 1)
    tree = cKDTree(points)
    _, nn = tree.query(points, k=max_k + 1)
    nn = nn[:, 1:]

    stats: dict[str, dict] = {}
    for k in k_values:
        if k > max_k:
            continue
        neighbors = nn[:, :k]
        neighbor_labels = labels[neighbors]
        same_fraction = np.mean(neighbor_labels == labels[:, None], axis=1)
        abs_delta_k = np.mean(np.abs(complexities[neighbors] - complexities[:, None]), axis=1)

        shuffled_same = []
        shuffled_abs_delta = []
        for _ in range(n_shuffles):
            shuffled = rng.permutation(labels)
            shuffled_complexities = rng.permutation(complexities)
            shuffled_same.append(float(np.mean(np.mean(shuffled[neighbors] == shuffled[:, None], axis=1))))
            shuffled_abs_delta.append(float(np.mean(np.mean(np.abs(shuffled_complexities[neighbors] - shuffled_complexities[:, None]), axis=1))))

        per_quartile = {}
        for label in range(4):
            mask = labels == label
            per_quartile[f"quartile_{label + 1}"] = {
                "n": int(np.sum(mask)),
                "same_quartile_fraction_mean": float(np.mean(same_fraction[mask])) if np.any(mask) else None,
                "mean_abs_complexity_delta": float(np.mean(abs_delta_k[mask])) if np.any(mask) else None,
            }

        stats[f"k={k}"] = {
            "n_points": int(len(points)),
            "quartile_edges": [float(x) for x in edges],
            "same_quartile_fraction": summary_stats(same_fraction),
            "same_quartile_fraction_shuffle_mean": float(np.mean(shuffled_same)) if shuffled_same else None,
            "same_quartile_enrichment_over_shuffle": float(np.mean(same_fraction) / np.mean(shuffled_same)) if shuffled_same and np.mean(shuffled_same) else None,
            "mean_abs_complexity_delta_to_neighbors": summary_stats(abs_delta_k),
            "mean_abs_complexity_delta_shuffle_mean": float(np.mean(shuffled_abs_delta)) if shuffled_abs_delta else None,
            "per_quartile": per_quartile,
        }
    return stats, points, labels


def radius_growth(
    points: np.ndarray,
    complexities: np.ndarray,
    codes: np.ndarray | None,
    rng: np.random.Generator,
    n_centers: int,
    radii: np.ndarray,
) -> dict:
    if len(points) == 0:
        return {}
    center_n = min(n_centers, len(points))
    center_idx = rng.choice(len(points), size=center_n, replace=False)
    centers = points[center_idx]
    tree = cKDTree(points)
    total_complexities = max(1, len(np.unique(complexities)))
    total_codes = max(1, len(np.unique(codes))) if codes is not None and len(codes) == len(points) else None

    rows = []
    for radius in radii:
        neighborhoods = tree.query_ball_point(centers, r=float(radius))
        n_points = np.asarray([len(idx) for idx in neighborhoods], dtype=float)
        n_complexities = np.asarray([len(np.unique(complexities[idx])) for idx in neighborhoods], dtype=float)
        row = {
            "radius": float(radius),
            "points": summary_stats(n_points),
            "unique_complexities": summary_stats(n_complexities),
            "unique_complexity_fraction_of_sample": summary_stats(n_complexities / total_complexities),
        }
        if total_codes is not None:
            n_codes = np.asarray([len(np.unique(codes[idx])) for idx in neighborhoods], dtype=float)
            row["unique_phenotypes"] = summary_stats(n_codes)
            row["unique_phenotype_fraction_of_sample"] = summary_stats(n_codes / total_codes)
        rows.append(row)
    return {
        "n_centers": int(center_n),
        "n_points_in_cloud": int(len(points)),
        "total_unique_complexities_in_cloud": int(total_complexities),
        "total_unique_phenotypes_in_cloud": int(total_codes) if total_codes is not None else None,
        "phenotype_codes_available": total_codes is not None,
        "rows": rows,
    }


def pca_scores(points: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    centered = points - np.mean(points, axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    n_components = min(n_components, vt.shape[0])
    scores = centered @ vt[:n_components].T
    variance = singular**2
    explained = variance / np.sum(variance) if np.sum(variance) else np.full_like(variance, np.nan)
    return scores, explained[:n_components]


def plot_pair_grid(
    out_dir: Path,
    model: str,
    tag: str,
    points: np.ndarray,
    labels: np.ndarray,
    parameter_names: list[str] | None,
    max_points: int,
    rng: np.random.Generator,
) -> Path:
    if len(points) > max_points:
        idx = rng.choice(len(points), size=max_points, replace=False)
        points = points[idx]
        labels = labels[idx]
    dim = points.shape[1]
    fig, axes = plt.subplots(dim - 1, dim - 1, figsize=(2.2 * (dim - 1), 2.2 * (dim - 1)), constrained_layout=True)
    axes = np.asarray(axes)
    axis_labels = []
    for idx in range(dim):
        if parameter_names is not None and idx < len(parameter_names):
            axis_labels.append(f"u{idx + 1} ({parameter_names[idx]})")
        else:
            axis_labels.append(f"u{idx + 1}")
    for i in range(dim - 1):
        for j in range(dim - 1):
            ax = axes[i, j]
            x_dim = j
            y_dim = i + 1
            if x_dim >= y_dim:
                ax.axis("off")
                continue
            ax.scatter(points[:, x_dim], points[:, y_dim], c=PALETTE[labels], s=2, alpha=0.25, linewidths=0)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            if i == dim - 2:
                ax.set_xlabel(axis_labels[x_dim], fontsize=7)
            else:
                ax.set_xticks([])
            if j == 0:
                ax.set_ylabel(axis_labels[y_dim], fontsize=7)
            else:
                ax.set_yticks([])
    fig.suptitle(f"{model} {tag}: all normalized parameter pairs colored by K quartile (n={len(points):,})")
    out = out_dir / f"{model}_locality_pairgrid_{tag}.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_analysis(
    out_dir: Path,
    model: str,
    tag: str,
    points: np.ndarray,
    complexities: np.ndarray,
    objectives: np.ndarray,
    labels: np.ndarray,
    pairwise_groups: dict[str, tuple[np.ndarray, dict]],
    growth: dict,
    max_plot_points: int,
    rng: np.random.Generator,
) -> Path:
    if len(points) > max_plot_points:
        idx = rng.choice(len(points), size=max_plot_points, replace=False)
        plot_points = points[idx]
        plot_labels = labels[idx]
    else:
        plot_points = points
        plot_labels = labels

    n_dim = points.shape[1]
    n_axis_pairs = min(math.ceil(n_dim / 2), 6)
    fig = plt.figure(figsize=(16, 5 + 3.6 * max(2, n_axis_pairs)), constrained_layout=True)
    gs = fig.add_gridspec(max(3, n_axis_pairs), 3)

    ax_nn = fig.add_subplot(gs[0, 0])
    ax_dist = fig.add_subplot(gs[0, 1])
    ax_growth = fig.add_subplot(gs[0, 2])

    # Lightweight PCA for context.
    scores, explained = pca_scores(plot_points, min(4, n_dim))
    ax_pca12 = fig.add_subplot(gs[1, 0])
    ax_pca34 = fig.add_subplot(gs[1, 1])
    if scores.shape[1] >= 2:
        ax_pca12.scatter(scores[:, 0], scores[:, 1], c=PALETTE[plot_labels], s=4, alpha=0.45, linewidths=0)
        ax_pca12.set_xlabel(f"PC1 ({explained[0]:.1%})")
        ax_pca12.set_ylabel(f"PC2 ({explained[1]:.1%})")
        ax_pca12.set_title("PCA 1/2")
    if scores.shape[1] >= 4:
        ax_pca34.scatter(scores[:, 2], scores[:, 3], c=PALETTE[plot_labels], s=4, alpha=0.45, linewidths=0)
        ax_pca34.set_xlabel(f"PC3 ({explained[2]:.1%})")
        ax_pca34.set_ylabel(f"PC4 ({explained[3]:.1%})")
        ax_pca34.set_title("PCA 3/4")

    # Same-quartile fraction is drawn from JSON-ish stats later by caller; leave a concise textual panel.
    ax_nn.axis("off")
    ax_nn.text(
        0.02,
        0.98,
        f"{model} / {tag}\n"
        f"points in locality cloud: {len(points):,}\n"
        f"dimensions: {n_dim}\n"
        f"plot sample: {len(plot_points):,}\n"
        f"quartile colors: blue -> red\n"
        f"WT-distance min/median/max:\n"
        f"{np.nanmin(objectives):.3g} / {np.nanmedian(objectives):.3g} / {np.nanmax(objectives):.3g}",
        ha="left",
        va="top",
        fontsize=10,
    )

    names = list(pairwise_groups)
    distributions = [pairwise_groups[name][0] for name in names]
    if distributions:
        ax_dist.boxplot(
            distributions,
            tick_labels=[
                f"{name}\n(n={pairwise_groups[name][1].get('n_points', 0):,}"
                f"; avail={pairwise_groups[name][1].get('n_points_available', pairwise_groups[name][1].get('n_points', 0)):,})"
                for name in names
            ],
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "#d7e6f5", "edgecolor": "black"},
            medianprops={"color": "black"},
        )
        for idx, values in enumerate(distributions, start=1):
            if len(values):
                ax_dist.scatter([idx], [np.max(values)], color="#c44e52", s=24, zorder=3, label="max" if idx == 1 else None)
        if any(len(values) for values in distributions):
            ax_dist.legend(frameon=False, fontsize=8)
    else:
        ax_dist.text(0.5, 0.5, "No pairwise groups passed filters", ha="center", va="center", transform=ax_dist.transAxes)
    ax_dist.set_ylabel("Euclidean distance in normalized cube")
    ax_dist.set_title("Pairwise distance distributions")

    if growth:
        radii = np.asarray([row["radius"] for row in growth["rows"]])
        y_key = "unique_phenotypes" if growth.get("phenotype_codes_available") else "unique_complexities"
        med = np.asarray([row[y_key]["median"] for row in growth["rows"]], dtype=float)
        q25 = np.asarray([row[y_key]["q25"] for row in growth["rows"]], dtype=float)
        q75 = np.asarray([row[y_key]["q75"] for row in growth["rows"]], dtype=float)
        ax_growth.plot(radii, med, color="black", lw=2)
        ax_growth.fill_between(radii, q25, q75, color="#9ecae1", alpha=0.45)
        ax_growth.set_yscale("log")
        ax_growth.set_xlabel("radius in normalized cube")
        ax_growth.set_ylabel(y_key.replace("_", " "))
        ax_growth.set_title(f"Accessible diversity around random centers\n(n centers={growth['n_centers']:,})")

    # Full coordinate pair views. Tyson can show every axis pair; Chen will be capped by --max-axis-pairs.
    start_row = 2
    for pair_idx in range(n_axis_pairs):
        row = start_row + pair_idx // 3
        col = pair_idx % 3
        ax = fig.add_subplot(gs[row, col])
        x_dim = 2 * pair_idx
        y_dim = x_dim + 1
        if y_dim >= n_dim:
            ax.hist(plot_points[:, x_dim], bins=60, color="#4c78a8", alpha=0.7)
            ax.set_xlabel(f"u{x_dim + 1}")
            ax.set_title(f"Axis {x_dim + 1}")
        else:
            ax.scatter(plot_points[:, x_dim], plot_points[:, y_dim], c=PALETTE[plot_labels], s=4, alpha=0.35, linewidths=0)
            ax.set_xlabel(f"u{x_dim + 1}")
            ax.set_ylabel(f"u{y_dim + 1}")
            ax.set_title(f"Normalized axes {x_dim + 1}/{y_dim + 1}")
        ax.set_xlim(0, 1)
        if y_dim < n_dim:
            ax.set_ylim(0, 1)

    out = out_dir / f"{model}_locality_{tag}.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def main(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    stats_dir = STATS_ROOT / args.tag
    sample = load_sample_npz(stats_dir, args.model, args.tag)
    summary = load_summary(stats_dir, args.model, args.tag)
    if "p0" not in sample:
        raise KeyError("sample npz is missing p0")

    points = np.asarray(sample["all_points"], dtype=float)
    complexities = np.asarray(sample["all_complexities"], dtype=float)
    objectives = np.asarray(sample["all_objectives"], dtype=float)
    points, complexities, objectives, mask = finite_rows(points, complexities, objectives)
    p0 = np.asarray(sample["p0"], dtype=float)
    parameter_names = None
    if "parameter_names" in sample:
        parameter_names = [str(x) for x in np.asarray(sample["parameter_names"], dtype=object)]
    norm_points = normalize_points(points, p0)
    codes = None
    if "all_phenotype_codes" in sample:
        raw_codes = np.asarray(sample["all_phenotype_codes"])
        if len(raw_codes) == len(mask):
            codes = raw_codes[mask]
    strict_points = np.empty((0, norm_points.shape[1]), dtype=float)
    if "neutral_points" in sample and len(sample["neutral_points"]):
        strict_points = normalize_points(np.asarray(sample["neutral_points"], dtype=float), p0)
        strict_points = strict_points[np.all(np.isfinite(strict_points), axis=1)]
    wt_phenotype_points = np.empty((0, norm_points.shape[1]), dtype=float)
    if "wt_phenotype_points" in sample and len(sample["wt_phenotype_points"]):
        wt_phenotype_points = normalize_points(np.asarray(sample["wt_phenotype_points"], dtype=float), p0)
        wt_phenotype_points = wt_phenotype_points[np.all(np.isfinite(wt_phenotype_points), axis=1)]

    if len(norm_points) > args.max_locality_points:
        idx = rng.choice(len(norm_points), size=args.max_locality_points, replace=False)
        locality_points = norm_points[idx]
        locality_complexities = complexities[idx]
        locality_objectives = objectives[idx]
        locality_codes = codes[idx] if codes is not None else None
    else:
        locality_points = norm_points
        locality_complexities = complexities
        locality_objectives = objectives
        locality_codes = codes

    locality_stats, locality_points, locality_labels = local_quartile_stats(
        locality_points,
        locality_complexities,
        rng,
        max_points=args.max_locality_points,
        k_values=args.k_values,
        n_shuffles=args.n_shuffles,
    )

    radii = np.geomspace(args.min_radius, args.max_radius or math.sqrt(norm_points.shape[1]), args.n_radii)
    growth = radius_growth(
        locality_points,
        locality_complexities,
        locality_codes,
        rng,
        args.radius_centers,
        radii,
    )

    neutral_cutoff = args.neutral_cutoff
    pairwise_candidates: dict[str, np.ndarray] = {}
    add_random_pairwise_controls(pairwise_candidates, norm_points, args.random_pairwise_controls)
    neutral_mask = objectives <= neutral_cutoff
    if len(strict_points):
        pairwise_candidates[f"WT neutral\nf<={neutral_cutoff:g}"] = strict_points
    else:
        pairwise_candidates[f"WT neutral\nf<={neutral_cutoff:g}"] = norm_points[neutral_mask]
    wt_code = summary.get("wildtype_code")
    wt_mask = None
    if not args.exclude_wt_phenotype_pairwise:
        if len(wt_phenotype_points):
            pairwise_candidates["WT phenotype\nsame bitstring"] = wt_phenotype_points
        elif codes is not None and wt_code is not None:
            wt_mask = codes == np.asarray(wt_code, dtype=codes.dtype)
            pairwise_candidates["WT phenotype\nsame bitstring"] = norm_points[wt_mask]
    if args.fallback_loose_cutoff is not None:
        loose_mask = objectives <= args.fallback_loose_cutoff
        loose_name = f"WT loose neutral\nf<={args.fallback_loose_cutoff:g}"
        if (
            "WT phenotype\nsame bitstring" not in pairwise_candidates
            or len(pairwise_candidates["WT phenotype\nsame bitstring"]) < args.min_pairwise_group_points
            or args.always_include_loose_cutoff
        ):
            pairwise_candidates[loose_name] = norm_points[loose_mask]

    pairwise_groups, pairwise_meta = equalized_pairwise_groups(
        pairwise_candidates,
        rng,
        args.max_pairwise_points,
        args.min_pairwise_group_points,
        args.pairwise_equal_n,
    )

    out_dir = stats_dir / "locality"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_analysis(
        out_dir,
        args.model,
        args.tag,
        locality_points,
        locality_complexities,
        locality_objectives,
        locality_labels,
        pairwise_groups,
        growth,
        args.max_plot_points,
        rng,
    )
    pair_grid_path = None
    if args.pairgrid and locality_points.shape[1] <= args.max_pairgrid_dims:
        pair_grid_path = plot_pair_grid(
            out_dir,
            args.model,
            args.tag,
            locality_points,
            locality_labels,
            parameter_names,
            args.max_pairgrid_points,
            rng,
        )

    result = {
        "model": args.model,
        "label": summary.get("label", spec_for(args.model).label),
        "tag": args.tag,
        "sample_npz": str(stats_dir / f"{args.model}_bruteforce_samples_{args.tag}.npz"),
        "stats_source_summary": str(stats_dir / f"{args.model}_bruteforce_summary_{args.tag}.json"),
        "dimensions": int(norm_points.shape[1]),
        "points_loaded": int(len(norm_points)),
        "points_used_for_locality": int(len(locality_points)),
        "neutral_cutoff": neutral_cutoff,
        "wt_phenotype_source": "saved wt_phenotype_points" if len(wt_phenotype_points) else "phenotype_code == wildtype_code" if wt_mask is not None else "unavailable; used fallback cutoff" if args.fallback_loose_cutoff is not None else "unavailable",
        "wt_phenotype_points_in_sample": int(len(wt_phenotype_points)) if len(wt_phenotype_points) else int(np.sum(wt_mask)) if wt_mask is not None else None,
        "fallback_loose_cutoff": args.fallback_loose_cutoff,
        "pairwise_equalization": pairwise_meta,
        "phenotype_codes_available_for_radius_growth": bool(locality_codes is not None),
        "local_quartile_neighbor_stats": locality_stats,
        "radius_growth": growth,
        "pairwise_distance_distributions_normalized_cube": {name: stats for name, (_, stats) in pairwise_groups.items()},
        "plot": str(plot_path),
        "pairgrid_plot": str(pair_grid_path) if pair_grid_path is not None else None,
    }
    json_path = out_dir / f"{args.model}_locality_{args.tag}.json"
    json_path.write_text(json.dumps(result, indent=2))
    print(f"Saved {json_path}")
    print(f"Saved {plot_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze brute-force cloud locality in the original normalized parameter cube")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--model", required=True, choices=[spec.key for spec in SPECS])
    parser.add_argument("--neutral-cutoff", type=float, required=True)
    parser.add_argument("--fallback-loose-cutoff", type=float, default=None)
    parser.add_argument("--always-include-loose-cutoff", action="store_true")
    parser.add_argument("--min-pairwise-group-points", type=int, default=2)
    parser.add_argument("--pairwise-equal-n", type=int, default=None)
    parser.add_argument("--random-pairwise-controls", type=int, default=1)
    parser.add_argument("--exclude-wt-phenotype-pairwise", action="store_true")
    parser.add_argument("--pairgrid", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-pairgrid-dims", type=int, default=12)
    parser.add_argument("--max-pairgrid-points", type=int, default=30000)
    parser.add_argument("--max-locality-points", type=int, default=50000)
    parser.add_argument("--max-pairwise-points", type=int, default=3000)
    parser.add_argument("--max-plot-points", type=int, default=50000)
    parser.add_argument("--max-axis-pairs", type=int, default=4)
    parser.add_argument("--radius-centers", type=int, default=500)
    parser.add_argument("--min-radius", type=float, default=0.03)
    parser.add_argument("--max-radius", type=float, default=None)
    parser.add_argument("--n-radii", type=int, default=24)
    parser.add_argument("--k-values", type=int, nargs="+", default=[5, 10, 25, 50])
    parser.add_argument("--n-shuffles", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())

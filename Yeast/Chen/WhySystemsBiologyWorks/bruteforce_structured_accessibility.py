"""Accessibility curves with synthetic structured phenotype controls.

This is a power check for the radius-accessibility statistic.  It keeps the
same genotype point cloud and the same phenotype-code frequency distribution,
then compares:

* observed labels from the simulations;
* randomly shuffled labels;
* compact synthetic labels, made by assigning each phenotype to contiguous
  blocks in a spatial KD ordering of the same points.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
STATS_ROOT = ROOT / "results/bruteforce_cloud_stats"
SUMMARY_ROOT = ROOT / "results_summaries/bruteforce_cloud"
FIGURE_ROOT = ROOT / "figures/bruteforce_cloud"


MODEL_CONFIGS = {
    "chen2004": {
        "label": "Chen 2004",
        "tag": "chen_bfc_1e8",
        "sample": STATS_ROOT / "chen_bfc_1e8/chen2004_bruteforce_samples_chen_bfc_1e8.npz",
        "locality": SUMMARY_ROOT / "chen_bfc_1e8/locality/chen2004_locality_chen_bfc_1e8.json",
    },
    "tyson1991": {
        "label": "Tyson 1991",
        "tag": "tyson_bfc_1e8",
        "sample": STATS_ROOT / "tyson_bfc_1e8/tyson1991_bruteforce_samples_tyson_bfc_1e8.npz",
        "locality": SUMMARY_ROOT / "tyson_bfc_1e8/locality/tyson1991_locality_tyson_bfc_1e8.json",
    },
}


def summarize(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    return {
        "n": int(values.size),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def load_cloud(path: Path, max_points: int | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    data = np.load(path, allow_pickle=True)
    points = np.asarray(data["all_points"], dtype=float)
    codes = np.asarray(data["all_phenotype_codes"])
    p0 = np.asarray(data["p0"], dtype=float)

    mask = np.all(np.isfinite(points), axis=1)
    points = points[mask]
    codes = codes[mask]

    points = points / np.maximum(2.0 * p0[None, :], np.finfo(float).tiny)
    if max_points is not None and len(points) > max_points:
        idx = rng.choice(len(points), size=max_points, replace=False)
        points = points[idx]
        codes = codes[idx]
    return points, codes


def kd_spatial_order(points: np.ndarray, leaf_size: int = 256) -> np.ndarray:
    """Return a deterministic ordering where nearby rows are often spatially close.

    This is not a formal space-filling curve, but recursive median splitting
    gives compact blocks without needing expensive clustering in 136D.
    """
    order: list[np.ndarray] = []

    def recurse(indices: np.ndarray) -> None:
        if len(indices) <= leaf_size:
            block = points[indices]
            axis = int(np.argmax(np.ptp(block, axis=0)))
            order.append(indices[np.argsort(block[:, axis], kind="mergesort")])
            return
        block = points[indices]
        axis = int(np.argmax(np.ptp(block, axis=0)))
        sorted_idx = indices[np.argsort(block[:, axis], kind="mergesort")]
        mid = len(sorted_idx) // 2
        recurse(sorted_idx[:mid])
        recurse(sorted_idx[mid:])

    recurse(np.arange(len(points)))
    return np.concatenate(order)


def compact_block_labels(
    points: np.ndarray,
    codes: np.ndarray,
    seed: int,
    leaf_size: int,
    pieces_per_phenotype: int,
) -> np.ndarray:
    """Assign the same phenotype counts to compact spatial blocks.

    Each phenotype is split into one or a few chunks.  Chunks are placed as
    contiguous intervals along the KD spatial ordering.  This preserves the
    exact multiset of phenotype labels while creating strong spatial structure.
    """
    rng = np.random.default_rng(seed)
    labels, counts = np.unique(codes, return_counts=True)
    order = kd_spatial_order(points, leaf_size=leaf_size)

    chunks: list[tuple[object, int, float]] = []
    for label, count in zip(labels, counts):
        n_pieces = max(1, min(int(pieces_per_phenotype), int(count)))
        splits = np.full(n_pieces, int(count) // n_pieces, dtype=int)
        splits[: int(count) % n_pieces] += 1
        for size in splits:
            if size > 0:
                chunks.append((label, int(size), float(rng.random())))

    # Large chunks first makes the synthetic control visibly compact; the random
    # tie-break prevents a deterministic relation between label value and space.
    chunks.sort(key=lambda item: (-item[1], item[2]))

    structured = np.empty_like(codes)
    cursor = 0
    for label, size, _ in chunks:
        idx = order[cursor : cursor + size]
        structured[idx] = label
        cursor += size
    if cursor != len(codes):
        raise RuntimeError(f"Structured assignment filled {cursor} of {len(codes)} points")
    return structured


def random_projection_slab_labels(points: np.ndarray, codes: np.ndarray, seed: int) -> np.ndarray:
    """A simpler structured control: labels occupy slabs along one random axis."""
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=points.shape[1])
    direction /= np.linalg.norm(direction)
    order = np.argsort(points @ direction, kind="mergesort")
    labels, counts = np.unique(codes, return_counts=True)
    perm = rng.permutation(len(labels))
    structured = np.empty_like(codes)
    cursor = 0
    for j in perm[np.argsort(-counts[perm], kind="mergesort")]:
        size = int(counts[j])
        structured[order[cursor : cursor + size]] = labels[j]
        cursor += size
    return structured


def unique_counts(neighborhoods: list[list[int]], codes: np.ndarray) -> np.ndarray:
    return np.asarray([len(np.unique(codes[idx])) for idx in neighborhoods], dtype=float)


def radius_curves(
    points: np.ndarray,
    observed_codes: np.ndarray,
    radii: np.ndarray,
    n_centers: int,
    n_shuffles: int,
    seed: int,
    leaf_size: int,
    pieces_per_phenotype: int,
) -> dict:
    rng = np.random.default_rng(seed)
    center_idx = rng.choice(len(points), size=min(n_centers, len(points)), replace=False)

    compact_codes = compact_block_labels(
        points,
        observed_codes,
        seed=seed + 101,
        leaf_size=leaf_size,
        pieces_per_phenotype=pieces_per_phenotype,
    )
    slab_codes = random_projection_slab_labels(points, observed_codes, seed=seed + 202)
    shuffled_codes = [rng.permutation(observed_codes) for _ in range(n_shuffles)]

    code_sets = {
        "observed": observed_codes,
        "compact_blocks": compact_codes,
        "random_projection_slabs": slab_codes,
    }
    for i, shuf in enumerate(shuffled_codes):
        code_sets[f"shuffle_{i}"] = shuf

    radii = np.asarray(radii, dtype=float)
    radii2 = radii * radii
    n_radii = len(radii)
    n_used_centers = len(center_idx)
    unique_by_kind = {
        key: np.zeros((n_radii, n_used_centers), dtype=float)
        for key in code_sets
    }
    points_in_ball = np.zeros((n_radii, n_used_centers), dtype=float)

    # In high dimensions cKDTree radius queries can be slower than brute-force
    # vectorized distances.  Sorting once per center lets every radius reuse the
    # same ordering.
    points = np.asarray(points, dtype=np.float32)
    point_norm2 = np.einsum("ij,ij->i", points, points)
    for center_col, idx in enumerate(center_idx):
        center = points[idx]
        d2 = point_norm2 + float(np.dot(center, center)) - 2.0 * (points @ center)
        d2 = np.maximum(d2, 0.0)
        order = np.argsort(d2, kind="mergesort")
        sorted_d2 = d2[order]
        counts_at_radii = np.searchsorted(sorted_d2, radii2, side="right")
        points_in_ball[:, center_col] = counts_at_radii
        for row, count in enumerate(counts_at_radii):
            prefix = order[: int(count)]
            for key, labels in code_sets.items():
                unique_by_kind[key][row, center_col] = len(np.unique(labels[prefix]))

    rows = []
    for row, radius in enumerate(radii):
        observed = unique_by_kind["observed"][row]
        compact = unique_by_kind["compact_blocks"][row]
        slab = unique_by_kind["random_projection_slabs"][row]
        shuffled = np.asarray([unique_by_kind[f"shuffle_{i}"][row] for i in range(n_shuffles)], dtype=float)
        shuffled_center_mean = np.mean(shuffled, axis=0)
        rows.append(
            {
                "radius": float(radius),
                "points_in_ball": summarize(points_in_ball[row]),
                "observed": summarize(observed),
                "shuffled": summarize(shuffled_center_mean),
                "compact_blocks": summarize(compact),
                "random_projection_slabs": summarize(slab),
                "observed_over_shuffled_median": float(
                    np.median(observed) / max(np.median(shuffled_center_mean), np.finfo(float).tiny)
                ),
                "compact_over_shuffled_median": float(
                    np.median(compact) / max(np.median(shuffled_center_mean), np.finfo(float).tiny)
                ),
                "slab_over_shuffled_median": float(
                    np.median(slab) / max(np.median(shuffled_center_mean), np.finfo(float).tiny)
                ),
            }
        )

    labels, counts = np.unique(observed_codes, return_counts=True)
    return {
        "n_points": int(len(points)),
        "dimensions": int(points.shape[1]),
        "n_centers": int(n_used_centers),
        "n_shuffles": int(n_shuffles),
        "total_unique_phenotypes": int(len(labels)),
        "phenotype_count_distribution": {
            "min": int(np.min(counts)),
            "q25": float(np.quantile(counts, 0.25)),
            "median": float(np.median(counts)),
            "q75": float(np.quantile(counts, 0.75)),
            "q95": float(np.quantile(counts, 0.95)),
            "max": int(np.max(counts)),
        },
        "structured_control": {
            "compact_blocks": "same phenotype counts assigned to contiguous blocks in recursive KD spatial order",
            "random_projection_slabs": "same phenotype counts assigned to contiguous slabs along one random projection",
            "pieces_per_phenotype": int(pieces_per_phenotype),
            "leaf_size": int(leaf_size),
        },
        "rows": rows,
    }


def plot_model(ax: plt.Axes, result: dict, title: str) -> None:
    rows = result["rows"]
    radii = np.asarray([row["radius"] for row in rows])

    series = [
        ("observed", "black", "-", "observed"),
        ("shuffled", "#d55e00", "--", "shuffled"),
        ("compact_blocks", "#0072b2", "-", "synthetic compact blobs"),
        ("random_projection_slabs", "#009e73", "-.", "synthetic slabs"),
    ]
    for key, color, style, label in series:
        med = np.asarray([row[key]["median"] for row in rows])
        q05 = np.asarray([row[key]["q05"] for row in rows])
        q95 = np.asarray([row[key]["q95"] for row in rows])
        ax.fill_between(radii, q05, q95, color=color, alpha=0.10, linewidth=0)
        ax.plot(radii, med, color=color, linestyle=style, linewidth=2.0, label=label)

    ax.set_yscale("log")
    ax.set_xlabel("radius in normalized cube")
    ax.set_ylabel("unique phenotypes")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.2)
    ax.text(
        0.02,
        0.96,
        f"d={result['dimensions']}\npoints={result['n_points']:,}\nphenotypes={result['total_unique_phenotypes']:,}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
    )


def run_model(model: str, args: argparse.Namespace) -> tuple[dict, Path, Path]:
    config = MODEL_CONFIGS[model]
    points, codes = load_cloud(config["sample"], max_points=args.max_points, seed=args.seed)
    locality = json.loads(config["locality"].read_text())
    radii = np.asarray([row["radius"] for row in locality["radius_growth"]["rows"]], dtype=float)

    result = radius_curves(
        points,
        codes,
        radii=radii,
        n_centers=args.centers,
        n_shuffles=args.shuffles,
        seed=args.seed,
        leaf_size=args.leaf_size,
        pieces_per_phenotype=args.pieces_per_phenotype,
    )
    result.update({"model": model, "label": config["label"], "tag": config["tag"], "sample_npz": str(config["sample"])})

    fig_dir = FIGURE_ROOT / config["tag"] / "structured_accessibility"
    sum_dir = SUMMARY_ROOT / config["tag"] / "structured_accessibility"
    fig_dir.mkdir(parents=True, exist_ok=True)
    sum_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{model}_structured_accessibility_{config['tag']}"
    json_path = sum_dir / f"{stem}.json"
    json_path.write_text(json.dumps(result, indent=2))

    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    plot_model(ax, result, config["label"])
    ax.legend(fontsize=8)
    fig_path = fig_dir / f"{stem}.png"
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)
    return result, fig_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Structured accessibility controls")
    parser.add_argument("--models", nargs="+", default=["chen2004", "tyson1991"], choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--max-points", type=int, default=100_000)
    parser.add_argument("--centers", type=int, default=500)
    parser.add_argument("--shuffles", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--leaf-size", type=int, default=256)
    parser.add_argument("--pieces-per-phenotype", type=int, default=1)
    args = parser.parse_args()

    combined: dict[str, dict] = {}
    figure_paths: list[Path] = []
    json_paths: list[Path] = []
    for model in args.models:
        result, fig_path, json_path = run_model(model, args)
        combined[model] = result
        figure_paths.append(fig_path)
        json_paths.append(json_path)
        print(f"Saved {fig_path}")
        print(f"Saved {json_path}")

    if len(args.models) > 1:
        fig, axes = plt.subplots(1, len(args.models), figsize=(7.2 * len(args.models), 5.4), constrained_layout=True)
        if len(args.models) == 1:
            axes = [axes]
        for ax, model in zip(axes, args.models):
            plot_model(ax, combined[model], combined[model]["label"])
        axes[-1].legend(fontsize=8, loc="lower right")
        out_dir = FIGURE_ROOT / "model_comparisons"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig_path = out_dir / "structured_accessibility_chen_tyson.png"
        fig.savefig(fig_path, dpi=220)
        plt.close(fig)

        out_json_dir = SUMMARY_ROOT / "model_comparisons"
        out_json_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_json_dir / "structured_accessibility_chen_tyson.json"
        json_path.write_text(json.dumps(combined, indent=2))
        print(f"Saved {fig_path}")
        print(f"Saved {json_path}")


if __name__ == "__main__":
    main()

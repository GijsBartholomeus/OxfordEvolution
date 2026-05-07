#!/usr/bin/env python3
"""Check whether a saved brute-force cloud looks uniform in the normalized cube."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from wsbw_pipeline import RESULTS, SPECS, prepare_models


RAW_ROOT = RESULTS / "bruteforce_cloud"
STATS_ROOT = RESULTS / "bruteforce_cloud_stats"
SUMMARY_ROOT = ROOT / "results_summaries" / "bruteforce_cloud"
FIGURE_ROOT = ROOT / "figures" / "bruteforce_cloud"


def spec_for(model: str):
    for spec in SPECS:
        if spec.key == model:
            return spec
    raise KeyError(model)


def parameter_names_for(model: str, dim: int) -> list[str]:
    try:
        audit = prepare_models()
        names = list(audit[spec_for(model).key]["free_parameters"])
        if len(names) == dim:
            return names
    except Exception:
        pass
    return [f"u{idx + 1}" for idx in range(dim)]


def stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def empirical_ks_uniform(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return float("nan")
    x = np.clip(x, 0.0, 1.0)
    upper = np.arange(1, n + 1, dtype=float) / n - x
    lower = x - np.arange(0, n, dtype=float) / n
    return float(max(np.max(upper), np.max(lower)))


def boundary_cdf(t: np.ndarray, dim: int) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    out = np.ones_like(t)
    mask = (t >= 0.0) & (t <= 0.5)
    out[t < 0.0] = 0.0
    out[mask] = 1.0 - np.power(1.0 - 2.0 * t[mask], dim)
    return out


def empirical_ks_boundary(b: np.ndarray, dim: int) -> float:
    b = np.sort(np.asarray(b, dtype=float))
    b = b[np.isfinite(b)]
    n = b.size
    if n == 0:
        return float("nan")
    f = boundary_cdf(b, dim)
    upper = np.arange(1, n + 1, dtype=float) / n - f
    lower = f - np.arange(0, n, dtype=float) / n
    return float(max(np.max(upper), np.max(lower)))


def chunk_files(tag: str, model: str) -> list[Path]:
    directory = RAW_ROOT / tag
    return sorted(directory.glob(f"{model}_bruteforce_cloud_N=*chunk-*.npz"))


def load_from_sample_npz(tag: str, model: str) -> tuple[np.ndarray, np.ndarray, dict]:
    path = STATS_ROOT / tag / f"{model}_bruteforce_samples_{tag}.npz"
    with np.load(path, allow_pickle=True) as data:
        points = np.asarray(data["all_points"], dtype=np.float32)
        p0 = np.asarray(data["p0"], dtype=np.float32)
    return points, p0, {"source": "sample_npz", "path": str(path)}


def raw_chunk_counts(files: list[Path]) -> np.ndarray:
    counts = []
    for path in files:
        with np.load(path, allow_pickle=True) as data:
            counts.append(int(data["points"].shape[0]))
    return np.asarray(counts, dtype=np.int64)


def choose_indices(total: int, n: int, rng: np.random.Generator) -> np.ndarray:
    n = min(int(n), int(total))
    if n <= 0:
        raise ValueError("No points available")
    return np.sort(rng.choice(total, size=n, replace=False).astype(np.int64))


def load_from_raw_chunks(tag: str, model: str, max_points: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict]:
    files = chunk_files(tag, model)
    if not files:
        raise FileNotFoundError(f"No raw chunk files for {model} in {RAW_ROOT / tag}")
    counts = raw_chunk_counts(files)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    selected = choose_indices(int(offsets[-1]), max_points, rng)
    with np.load(files[0], allow_pickle=True) as data:
        p0 = np.asarray(data["p0"], dtype=np.float32)
    points = np.empty((len(selected), len(p0)), dtype=np.float32)
    write_at = 0
    nonempty = 0
    for chunk_idx, path in enumerate(files):
        start = offsets[chunk_idx]
        stop = offsets[chunk_idx + 1]
        left = np.searchsorted(selected, start, side="left")
        right = np.searchsorted(selected, stop, side="left")
        if right <= left:
            continue
        local_idx = selected[left:right] - start
        with np.load(path, allow_pickle=True) as data:
            chunk_points = np.asarray(data["points"], dtype=np.float32)
            n = len(local_idx)
            points[write_at : write_at + n] = chunk_points[local_idx]
            write_at += n
        nonempty += 1
        if nonempty % 25 == 0:
            print(f"  loaded from {nonempty} chunks; rows={write_at:,}/{len(selected):,}", flush=True)
    return points[:write_at], p0, {
        "source": "raw_chunks",
        "chunk_files": len(files),
        "source_successful_points": int(offsets[-1]),
        "sampled_points": int(write_at),
    }


def normalize_points(points: np.ndarray, p0: np.ndarray) -> np.ndarray:
    scale = np.maximum(2.0 * p0[None, :], np.finfo(np.float32).tiny)
    x = np.asarray(points, dtype=np.float32) / scale
    return x


def load_points(args: argparse.Namespace, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    sample_path = STATS_ROOT / args.tag / f"{args.model}_bruteforce_samples_{args.tag}.npz"
    if args.source == "sample" or (args.source == "auto" and sample_path.exists()):
        points, p0, meta = load_from_sample_npz(args.tag, args.model)
    else:
        points, p0, meta = load_from_raw_chunks(args.tag, args.model, args.max_points, rng)

    if len(points) > args.max_points:
        idx = rng.choice(len(points), size=args.max_points, replace=False)
        points = points[idx]
        meta["subsampled_from_sample_npz"] = int(len(idx))

    x = normalize_points(points, p0)
    meta.update({
        "sampled_points": int(len(x)),
        "dimensions": int(x.shape[1]),
        "outside_unit_cube_fraction": float(np.mean((x < 0.0) | (x > 1.0))),
    })
    return x, meta


def sample_pair_distances(x: np.ndarray, pair_count: int, rng: np.random.Generator) -> np.ndarray:
    n = len(x)
    m = int(min(pair_count, max(1, n * 10)))
    a = rng.integers(0, n, size=m)
    b = rng.integers(0, n, size=m)
    same = a == b
    while np.any(same):
        b[same] = rng.integers(0, n, size=int(np.sum(same)))
        same = a == b
    diffs = x[a].astype(np.float32) - x[b].astype(np.float32)
    return np.sqrt(np.einsum("ij,ij->i", diffs, diffs)).astype(np.float32)


def random_cube_distances(dim: int, pair_count: int, rng: np.random.Generator, block: int = 200_000) -> np.ndarray:
    out = np.empty(pair_count, dtype=np.float32)
    write_at = 0
    while write_at < pair_count:
        n = min(block, pair_count - write_at)
        a = rng.random((n, dim), dtype=np.float32)
        b = rng.random((n, dim), dtype=np.float32)
        diff = a - b
        out[write_at : write_at + n] = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        write_at += n
    return out


def covariance_eigenvalues(x: np.ndarray) -> np.ndarray:
    xc = x.astype(np.float64) - np.mean(x, axis=0, keepdims=True)
    cov = (xc.T @ xc) / max(1, len(xc) - 1)
    return np.linalg.eigvalsh(cov)[::-1]


def participation_ratio(evals: np.ndarray) -> float:
    evals = np.asarray(evals, dtype=float)
    denom = float(np.sum(evals * evals))
    if denom <= 0:
        return 0.0
    return float(np.sum(evals) ** 2 / denom)


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    x, load_meta = load_points(args, rng)
    n, dim = x.shape
    parameter_names = parameter_names_for(args.model, dim)

    coord_means = np.mean(x, axis=0)
    coord_stds = np.std(x, axis=0, ddof=1)
    coord_ks = np.asarray([empirical_ks_uniform(x[:, j]) for j in range(dim)], dtype=float)
    mean_z = (coord_means - 0.5) / math.sqrt(1.0 / (12.0 * n))
    boundary = np.min(np.minimum(x, 1.0 - x), axis=1)
    boundary_ks = empirical_ks_boundary(boundary, dim)

    evals = covariance_eigenvalues(x)
    pair_dist = sample_pair_distances(x, args.pair_count, rng)
    cube_dist = random_cube_distances(dim, len(pair_dist), rng)
    pair_sq = pair_dist * pair_dist
    cube_sq_mean = dim / 6.0
    cube_sq_var = 7.0 * dim / 180.0

    result = {
        "model": args.model,
        "label": spec_for(args.model).label,
        "tag": args.tag,
        "seed": int(args.seed),
        "load": load_meta,
        "n": int(n),
        "dimensions": int(dim),
        "top_coordinate_mean_deviations": [
            {
                "axis": int(idx),
                "parameter": parameter_names[int(idx)],
                "mean": float(coord_means[int(idx)]),
                "std": float(coord_stds[int(idx)]),
                "ks_uniform": float(coord_ks[int(idx)]),
                "mean_z": float(mean_z[int(idx)]),
                "min": float(np.min(x[:, int(idx)])),
                "max": float(np.max(x[:, int(idx)])),
            }
            for idx in np.argsort(np.abs(coord_means - 0.5))[::-1][: min(12, dim)]
        ],
        "coordinate_means": stats(coord_means),
        "coordinate_stds": stats(coord_stds),
        "coordinate_ks_uniform": stats(coord_ks),
        "coordinate_mean_z_scores": stats(mean_z),
        "max_abs_coordinate_mean_z": float(np.max(np.abs(mean_z))),
        "boundary_distance_to_nearest_face": stats(boundary),
        "boundary_ks_against_uniform_cube": float(boundary_ks),
        "covariance_eigenvalues": stats(evals),
        "covariance_participation_ratio": participation_ratio(evals),
        "uniform_cube_expected_coordinate_std": math.sqrt(1.0 / 12.0),
        "uniform_cube_expected_covariance_eigenvalue": 1.0 / 12.0,
        "pairwise_distances": stats(pair_dist),
        "random_cube_pairwise_distances_mc": stats(cube_dist),
        "pairwise_squared_distance": stats(pair_sq),
        "uniform_cube_expected_squared_distance_mean": float(cube_sq_mean),
        "uniform_cube_expected_squared_distance_sd": float(math.sqrt(cube_sq_var)),
        "elapsed_seconds": float(time.time() - t0),
    }

    summary_dir = SUMMARY_ROOT / args.output_tag / "uniformity"
    figure_dir = FIGURE_ROOT / args.output_tag / "uniformity"
    summary_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    json_path = summary_dir / f"{args.model}_cloud_uniformity_{args.output_tag}.json"
    png_path = figure_dir / f"{args.model}_cloud_uniformity_{args.output_tag}.png"
    json_path.write_text(json.dumps(result, indent=2))
    plot_uniformity(result, x, coord_means, coord_stds, coord_ks, boundary, evals, pair_dist, cube_dist, png_path)
    return json_path, png_path


def plot_uniformity(
    result: dict,
    x: np.ndarray,
    coord_means: np.ndarray,
    coord_stds: np.ndarray,
    coord_ks: np.ndarray,
    boundary: np.ndarray,
    evals: np.ndarray,
    pair_dist: np.ndarray,
    cube_dist: np.ndarray,
    out_path: Path,
) -> None:
    dim = x.shape[1]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), constrained_layout=True)

    axes[0, 0].hist(coord_means, bins=30, color="#4c78a8", alpha=0.8)
    axes[0, 0].axvline(0.5, color="black", lw=1.5)
    axes[0, 0].set_title("Coordinate means")
    axes[0, 0].set_xlabel("mean normalized coordinate")
    axes[0, 0].set_ylabel("parameters")

    axes[0, 1].hist(coord_stds, bins=30, color="#59a14f", alpha=0.8)
    axes[0, 1].axvline(math.sqrt(1.0 / 12.0), color="black", lw=1.5)
    axes[0, 1].set_title("Coordinate standard deviations")
    axes[0, 1].set_xlabel("std")

    axes[0, 2].hist(coord_ks, bins=30, color="#f28e2b", alpha=0.8)
    axes[0, 2].set_title("Per-coordinate KS to U(0,1)")
    axes[0, 2].set_xlabel("KS statistic")

    sorted_boundary = np.sort(boundary)
    y = np.arange(1, len(sorted_boundary) + 1) / len(sorted_boundary)
    axes[1, 0].plot(sorted_boundary, y, color="black", lw=2, label="observed")
    grid = np.linspace(0, 0.5, 300)
    axes[1, 0].plot(grid, boundary_cdf(grid, dim), color="#e15759", lw=2, label="uniform cube")
    axes[1, 0].set_title("Distance to nearest cube face")
    axes[1, 0].set_xlabel("min_j min(x_j, 1-x_j)")
    axes[1, 0].set_ylabel("CDF")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(np.arange(1, len(evals) + 1), evals, marker="o", ms=3, lw=1.5)
    axes[1, 1].axhline(1.0 / 12.0, color="black", lw=1.5, label="uniform expectation")
    axes[1, 1].set_title("Covariance spectrum")
    axes[1, 1].set_xlabel("axis")
    axes[1, 1].set_ylabel("eigenvalue")
    axes[1, 1].legend(frameon=False)

    for values, label, color in [
        (pair_dist, "observed", "black"),
        (cube_dist, "uniform cube MC", "#e15759"),
    ]:
        vals = np.sort(values)
        yy = np.arange(1, len(vals) + 1) / len(vals)
        axes[1, 2].plot(vals, yy, lw=2, label=label, color=color)
    axes[1, 2].set_title("Pairwise distance CDF")
    axes[1, 2].set_xlabel("distance")
    axes[1, 2].set_ylabel("CDF")
    axes[1, 2].legend(frameon=False)

    fig.suptitle(
        f"{result['label']} saved-cloud uniformity check\n"
        f"{result['tag']}, n={result['n']:,}, d={result['dimensions']}",
        fontsize=14,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=[spec.key for spec in SPECS])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output-tag", default=None)
    parser.add_argument("--source", choices=["auto", "sample", "raw"], default="auto")
    parser.add_argument("--max-points", type=int, default=200_000)
    parser.add_argument("--pair-count", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_tag is None:
        args.output_tag = args.tag
    json_path, png_path = run(args)
    print(f"Saved {json_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()

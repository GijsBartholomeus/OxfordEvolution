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

from wsbw_pipeline import RESULTS, SPECS


RAW_ROOT = RESULTS / "bruteforce_cloud"
SUMMARY_ROOT = ROOT / "results_summaries" / "bruteforce_cloud"
FIGURE_ROOT = ROOT / "figures" / "bruteforce_cloud"


def spec_for(model: str):
    for spec in SPECS:
        if spec.key == model:
            return spec
    raise KeyError(model)


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


def chunk_files(tag: str, model: str) -> list[Path]:
    directory = RAW_ROOT / tag
    files = sorted(directory.glob(f"{model}_bruteforce_cloud_N=*chunk-*.npz"))
    if not files:
        raise FileNotFoundError(f"No raw chunk npz files found in {directory} for {model}")
    return files


def chunk_count(path: Path) -> int:
    with np.load(path, allow_pickle=True) as data:
        return int(data["points"].shape[0])


def choose_global_indices(counts: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    total = int(np.sum(counts))
    n = min(int(max_points), total)
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    return np.sort(rng.choice(total, size=n, replace=False).astype(np.int64))


def load_uniform_sample(
    files: list[Path],
    model: str,
    tag: str,
    max_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    t0 = time.time()
    counts = np.asarray([chunk_count(path) for path in files], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    selected = choose_global_indices(counts, max_points, rng)
    n_sample = len(selected)
    if n_sample == 0:
        raise ValueError("No points available to sample")

    first = np.load(files[0], allow_pickle=True)
    p0 = np.asarray(first["p0"], dtype=np.float32)
    dim = len(p0)
    first.close()

    points = np.empty((n_sample, dim), dtype=np.float32)
    codes = np.empty(n_sample, dtype=np.uint64)

    write_at = 0
    cursor = 0
    for chunk_idx, path in enumerate(files):
        start = offsets[chunk_idx]
        stop = offsets[chunk_idx + 1]
        left = np.searchsorted(selected, start, side="left", sorter=None)
        right = np.searchsorted(selected, stop, side="left", sorter=None)
        if right <= left:
            continue
        local_idx = selected[left:right] - start
        with np.load(path, allow_pickle=True) as data:
            chunk_points = np.asarray(data["points"], dtype=np.float32)
            chunk_codes = np.asarray(data["phenotype_codes"], dtype=np.uint64)
            n = len(local_idx)
            points[write_at : write_at + n] = chunk_points[local_idx]
            codes[write_at : write_at + n] = chunk_codes[local_idx]
            write_at += n
        cursor += 1
        if cursor % 10 == 0:
            print(f"  sampled from {cursor} nonempty chunks; rows={write_at:,}/{n_sample:,}", flush=True)

    points = points[:write_at]
    codes = codes[:write_at]
    scale = np.maximum(2.0 * p0[None, :], np.finfo(np.float32).tiny)
    points = points / scale
    meta = {
        "source_tag": tag,
        "model": model,
        "chunk_files": len(files),
        "source_successful_points": int(np.sum(counts)),
        "sampled_points": int(len(points)),
        "sample_seconds": float(time.time() - t0),
    }
    return points, codes, p0, meta


def make_radii(dim: int, min_radius: float, max_radius: float | None, n_radii: int) -> np.ndarray:
    if max_radius is None:
        max_radius = math.sqrt(dim)
    return np.geomspace(float(min_radius), float(max_radius), int(n_radii))


def accessibility_curve(
    points: np.ndarray,
    codes: np.ndarray,
    centers: int,
    radii: np.ndarray,
    rng: np.random.Generator,
    center_block: int,
) -> dict:
    t0 = time.time()
    n = len(points)
    center_n = min(int(centers), n)
    center_idx = rng.choice(n, size=center_n, replace=False)
    center_points = np.asarray(points[center_idx], dtype=np.float32)
    point_norm2 = np.einsum("ij,ij->i", points, points).astype(np.float32)

    counts_by_radius = np.empty((len(radii), center_n), dtype=np.float64)
    unique_by_radius = np.empty((len(radii), center_n), dtype=np.float64)

    center_offset = 0
    for block_start in range(0, center_n, center_block):
        block_stop = min(block_start + center_block, center_n)
        centers_block = center_points[block_start:block_stop]
        center_norm2 = np.einsum("ij,ij->i", centers_block, centers_block).astype(np.float32)
        dist2 = point_norm2[:, None] + center_norm2[None, :] - 2.0 * (points @ centers_block.T)
        np.maximum(dist2, 0.0, out=dist2)
        for local_col, global_col in enumerate(range(block_start, block_stop)):
            d2 = dist2[:, local_col]
            for ri, radius in enumerate(radii):
                mask = d2 <= float(radius * radius)
                counts_by_radius[ri, global_col] = float(np.sum(mask))
                unique_by_radius[ri, global_col] = float(len(np.unique(codes[mask])))
        center_offset += block_stop - block_start
        print(f"  processed centers {center_offset:,}/{center_n:,}", flush=True)

    rows = []
    total_unique = len(np.unique(codes))
    for ri, radius in enumerate(radii):
        rows.append(
            {
                "radius": float(radius),
                "points": summary_stats(counts_by_radius[ri]),
                "unique_phenotypes": summary_stats(unique_by_radius[ri]),
                "unique_phenotype_fraction_of_sample": summary_stats(unique_by_radius[ri] / max(1, total_unique)),
            }
        )
    return {
        "n_centers": int(center_n),
        "n_points_in_cloud": int(n),
        "total_unique_phenotypes_in_cloud": int(total_unique),
        "rows": rows,
        "accessibility_seconds": float(time.time() - t0),
    }


def plot_curve(result: dict, out_path: Path, x_log: bool) -> None:
    rows = result["radius_growth"]["rows"]
    radii = np.asarray([row["radius"] for row in rows], dtype=float)
    med = np.asarray([row["unique_phenotypes"]["median"] for row in rows], dtype=float)
    q25 = np.asarray([row["unique_phenotypes"]["q25"] for row in rows], dtype=float)
    q75 = np.asarray([row["unique_phenotypes"]["q75"] for row in rows], dtype=float)
    q05 = np.asarray([row["unique_phenotypes"]["q05"] for row in rows], dtype=float)
    q95 = np.asarray([row["unique_phenotypes"]["q95"] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 6), constrained_layout=True)
    ax.plot(radii, med, color="black", lw=2.5, label="median")
    ax.fill_between(radii, q25, q75, color="#9ecae1", alpha=0.45, label="IQR")
    ax.fill_between(radii, q05, q95, color="#9ecae1", alpha=0.18, label="5-95%")
    ax.set_yscale("log")
    if x_log:
        ax.set_xscale("log")
    ax.set_xlabel("radius in normalized cube")
    ax.set_ylabel("unique phenotypes")
    ax.set_title(
        f"{result['label']} radius accessibility\n"
        f"n={result['sample']['sampled_points']:,}, centers={result['radius_growth']['n_centers']:,}, "
        f"d={result['dimensions']}"
    )
    ax.legend(frameon=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    rng = np.random.default_rng(args.seed)
    files = chunk_files(args.source_tag, args.model)
    points, codes, _p0, sample_meta = load_uniform_sample(files, args.model, args.source_tag, args.max_points, rng)
    radii = make_radii(points.shape[1], args.min_radius, args.max_radius, args.n_radii)
    growth = accessibility_curve(points, codes, args.centers, radii, rng, args.center_block)
    result = {
        "model": args.model,
        "label": spec_for(args.model).label,
        "source_tag": args.source_tag,
        "tag": args.output_tag,
        "dimensions": int(points.shape[1]),
        "seed": int(args.seed),
        "sample": sample_meta,
        "radius_growth": growth,
    }

    summary_dir = SUMMARY_ROOT / args.output_tag / "radius_accessibility_large"
    figure_dir = FIGURE_ROOT / args.output_tag / "radius_accessibility_large"
    summary_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    json_path = summary_dir / f"{args.model}_radius_accessibility_{args.output_tag}.json"
    png_path = figure_dir / f"{args.model}_radius_accessibility_{args.output_tag}.png"
    json_path.write_text(json.dumps(result, indent=2))
    plot_curve(result, png_path, args.x_log)
    return json_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Large-sample radius accessibility from brute-force chunk files")
    parser.add_argument("--model", required=True, choices=[spec.key for spec in SPECS])
    parser.add_argument("--source-tag", required=True, help="Raw brute-force cloud tag to sample from")
    parser.add_argument("--output-tag", required=True)
    parser.add_argument("--max-points", type=int, default=1_000_000)
    parser.add_argument("--centers", type=int, default=100)
    parser.add_argument("--center-block", type=int, default=4)
    parser.add_argument("--min-radius", type=float, default=0.03)
    parser.add_argument("--max-radius", type=float, default=None)
    parser.add_argument("--n-radii", type=int, default=24)
    parser.add_argument("--x-log", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    json_path, png_path = run(args)
    print(f"Saved {json_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()

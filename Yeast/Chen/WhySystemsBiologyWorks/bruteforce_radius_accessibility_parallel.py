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
WORK_ROOT = RESULTS / "bruteforce_radius_accessibility_parallel"
SUMMARY_ROOT = ROOT / "results_summaries" / "bruteforce_cloud"
FIGURE_ROOT = ROOT / "figures" / "bruteforce_cloud"


def spec_for(model: str):
    for spec in SPECS:
        if spec.key == model:
            return spec
    raise KeyError(model)


def chunk_files(tag: str, model: str) -> list[Path]:
    files = sorted((RAW_ROOT / tag).glob(f"{model}_bruteforce_cloud_N=*chunk-*.npz"))
    if not files:
        raise FileNotFoundError(f"No raw chunk files for {model} in {RAW_ROOT / tag}")
    return files


def chunk_count(path: Path) -> int:
    with np.load(path, allow_pickle=True) as data:
        return int(data["points"].shape[0])


def make_radii(dim: int, min_radius: float, max_radius: float | None, n_radii: int) -> np.ndarray:
    if max_radius is None:
        max_radius = math.sqrt(dim)
    return np.geomspace(float(min_radius), float(max_radius), int(n_radii)).astype(np.float32)


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


def prepare(args: argparse.Namespace) -> None:
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    files = chunk_files(args.source_tag, args.model)
    counts = np.asarray([chunk_count(path) for path in files], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    total = int(offsets[-1])
    n_sample = min(int(args.max_points), total)
    selected = np.sort(rng.choice(total, size=n_sample, replace=False).astype(np.int64))

    with np.load(files[0], allow_pickle=True) as first:
        p0 = np.asarray(first["p0"], dtype=np.float32)
        parameter_names = np.asarray(first["parameter_names"], dtype=object)
    dim = len(p0)
    points = np.empty((n_sample, dim), dtype=np.float32)
    codes = np.empty(n_sample, dtype=np.uint64)

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
            n = len(local_idx)
            points[write_at : write_at + n] = np.asarray(data["points"], dtype=np.float32)[local_idx]
            codes[write_at : write_at + n] = np.asarray(data["phenotype_codes"], dtype=np.uint64)[local_idx]
            write_at += n
        nonempty += 1
        if nonempty % 20 == 0:
            print(f"sampled from {nonempty} nonempty chunks; rows={write_at:,}/{n_sample:,}", flush=True)

    points = points[:write_at]
    codes = codes[:write_at]
    scale = np.maximum(2.0 * p0[None, :], np.finfo(np.float32).tiny)
    points = points / scale
    unique_codes, label_ids = np.unique(codes, return_inverse=True)
    label_ids = label_ids.astype(np.int32)
    center_count = min(int(args.centers), len(points))
    center_indices = rng.choice(len(points), size=center_count, replace=False).astype(np.int64)
    radii = make_radii(dim, args.min_radius, args.max_radius, args.n_radii)

    out_dir = WORK_ROOT / args.output_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    cloud_path = out_dir / f"{args.model}_radius_cloud_{args.output_tag}.npz"
    np.savez(
        cloud_path,
        points=points.astype(np.float32),
        label_ids=label_ids,
        unique_codes=unique_codes.astype(np.uint64),
        center_indices=center_indices,
        radii=radii,
        p0=p0,
        parameter_names=parameter_names,
    )
    meta = {
        "model": args.model,
        "label": spec_for(args.model).label,
        "source_tag": args.source_tag,
        "output_tag": args.output_tag,
        "source_successful_points": total,
        "sampled_points": int(len(points)),
        "unique_phenotypes_in_sample": int(len(unique_codes)),
        "centers": int(center_count),
        "dimensions": int(dim),
        "radii": radii.tolist(),
        "cloud_npz": str(cloud_path),
        "elapsed_seconds": float(time.time() - t0),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


def shard(args: argparse.Namespace) -> None:
    t0 = time.time()
    work_dir = WORK_ROOT / args.output_tag
    cloud_path = work_dir / f"{args.model}_radius_cloud_{args.output_tag}.npz"
    with np.load(cloud_path, allow_pickle=True) as data:
        points = np.asarray(data["points"], dtype=np.float32)
        label_ids = np.asarray(data["label_ids"], dtype=np.int32)
        center_indices = np.asarray(data["center_indices"], dtype=np.int64)
        radii = np.asarray(data["radii"], dtype=np.float32)
        unique_codes = np.asarray(data["unique_codes"], dtype=np.uint64)

    start = int(args.center_start)
    stop = min(start + int(args.center_count), len(center_indices))
    point_norm2 = np.einsum("ij,ij->i", points, points).astype(np.float32)
    rows = []
    radius2 = (radii * radii).astype(np.float32)
    for center_pos in range(start, stop):
        idx = int(center_indices[center_pos])
        center = points[idx]
        d2 = point_norm2 + float(np.dot(center, center)) - 2.0 * (points @ center)
        np.maximum(d2, 0.0, out=d2)
        order = np.argsort(d2, kind="stable")
        sorted_d2 = d2[order]
        sorted_labels = label_ids[order]
        prefix_counts = np.searchsorted(sorted_d2, radius2, side="right")
        seen = np.zeros(len(unique_codes), dtype=bool)
        prev = 0
        unique_counts = []
        n_unique = 0
        for count in prefix_counts:
            if count > prev:
                labels = np.unique(sorted_labels[prev:count])
                new = ~seen[labels]
                if np.any(new):
                    n_unique += int(np.count_nonzero(new))
                    seen[labels[new]] = True
                prev = int(count)
            unique_counts.append(n_unique)
        rows.append(
            {
                "center_position": int(center_pos),
                "center_index": idx,
                "points_by_radius": prefix_counts.astype(int).tolist(),
                "unique_phenotypes_by_radius": unique_counts,
            }
        )
        print(f"processed center {center_pos + 1}/{len(center_indices)}", flush=True)

    shard_dir = work_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    out = shard_dir / f"shard_{start:05d}_{stop:05d}.json"
    out.write_text(
        json.dumps(
            {
                "model": args.model,
                "output_tag": args.output_tag,
                "center_start": start,
                "center_stop": stop,
                "radii": radii.tolist(),
                "rows": rows,
                "elapsed_seconds": float(time.time() - t0),
            },
            indent=2,
        )
    )
    print(f"Saved {out}", flush=True)


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
        f"n={result['sampled_points']:,}, centers={result['radius_growth']['n_centers']:,}, "
        f"d={result['dimensions']}"
    )
    ax.legend(frameon=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def merge(args: argparse.Namespace) -> None:
    work_dir = WORK_ROOT / args.output_tag
    meta = json.loads((work_dir / "metadata.json").read_text())
    shard_paths = sorted((work_dir / "shards").glob("shard_*.json"))
    rows = []
    for path in shard_paths:
        rows.extend(json.loads(path.read_text())["rows"])
    radii = np.asarray(meta["radii"], dtype=float)
    point_counts = np.asarray([row["points_by_radius"] for row in rows], dtype=float).T
    unique_counts = np.asarray([row["unique_phenotypes_by_radius"] for row in rows], dtype=float).T
    growth_rows = []
    for i, radius in enumerate(radii):
        growth_rows.append(
            {
                "radius": float(radius),
                "points": summary_stats(point_counts[i]),
                "unique_phenotypes": summary_stats(unique_counts[i]),
                "unique_phenotype_fraction_of_sample": summary_stats(
                    unique_counts[i] / max(1, meta["unique_phenotypes_in_sample"])
                ),
            }
        )
    result = {
        **meta,
        "radius_growth": {
            "n_centers": int(len(rows)),
            "n_points_in_cloud": int(meta["sampled_points"]),
            "total_unique_phenotypes_in_cloud": int(meta["unique_phenotypes_in_sample"]),
            "rows": growth_rows,
            "shards_merged": len(shard_paths),
        },
    }
    summary_dir = SUMMARY_ROOT / args.output_tag / "radius_accessibility_large"
    figure_dir = FIGURE_ROOT / args.output_tag / "radius_accessibility_large"
    summary_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    json_path = summary_dir / f"{args.model}_radius_accessibility_{args.output_tag}.json"
    png_path = figure_dir / f"{args.model}_radius_accessibility_{args.output_tag}.png"
    json_path.write_text(json.dumps(result, indent=2))
    plot_curve(result, png_path, args.x_log)
    print(f"Saved {json_path}", flush=True)
    print(f"Saved {png_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel radius accessibility from a shared sampled cloud")
    sub = parser.add_subparsers(dest="command", required=True)

    common_models = {"choices": [spec.key for spec in SPECS]}
    p = sub.add_parser("prepare")
    p.add_argument("--model", required=True, **common_models)
    p.add_argument("--source-tag", required=True)
    p.add_argument("--output-tag", required=True)
    p.add_argument("--max-points", type=int, default=10_000_000)
    p.add_argument("--centers", type=int, default=50)
    p.add_argument("--min-radius", type=float, default=0.03)
    p.add_argument("--max-radius", type=float, default=None)
    p.add_argument("--n-radii", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("shard")
    p.add_argument("--model", required=True, **common_models)
    p.add_argument("--output-tag", required=True)
    p.add_argument("--center-start", type=int, required=True)
    p.add_argument("--center-count", type=int, default=1)

    p = sub.add_parser("merge")
    p.add_argument("--model", required=True, **common_models)
    p.add_argument("--output-tag", required=True)
    p.add_argument("--x-log", action="store_true")

    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "shard":
        shard(args)
    elif args.command == "merge":
        merge(args)


if __name__ == "__main__":
    main()

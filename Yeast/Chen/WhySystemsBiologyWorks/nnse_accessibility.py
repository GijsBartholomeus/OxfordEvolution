from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "nnse_accessibility"
RESULTS.mkdir(parents=True, exist_ok=True)


@dataclass
class AccessibilityConfig:
    npz: str
    n_random: int = 10000
    max_cloud: int = 50000
    seed: int = 42
    space: str = "unit"
    k_nearest: int = 1


def find_candidate_npzs(root: Path = ROOT, model: str = "chen2004") -> list[Path]:
    base = root / "results" / "nnse_parallel"
    paths = [
        *base.glob(f"**/{model}_nnse_parallel_merged_*chains.npz"),
        *base.glob(f"**/{model}_nnse_parallel_chain-*_seed-*.npz"),
    ]
    return sorted(set(paths))


def count_neutral_points(path: Path) -> int:
    try:
        data = np.load(path, allow_pickle=True)
        return int(len(data["neutral_points"])) if "neutral_points" in data.files else 0
    except Exception:
        return 0


def choose_biggest_neutral_npz(root: Path = ROOT, model: str = "chen2004") -> Path:
    paths = find_candidate_npzs(root, model)
    if not paths:
        raise FileNotFoundError(f"No NNSE neutral-set npz files found under {root / 'results' / 'nnse_parallel'}")
    return max(paths, key=lambda path: (count_neutral_points(path), path.stat().st_size, path.stat().st_mtime))


def load_neutral_npz(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    required = {"neutral_points", "neutral_objective_values", "p0", "parameter_names", "bin_thresholds"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
    return {
        "path": path,
        "neutral_points": np.asarray(data["neutral_points"], dtype=float),
        "neutral_objective_values": np.asarray(data["neutral_objective_values"], dtype=float),
        "p0": np.asarray(data["p0"], dtype=float),
        "parameter_names": np.asarray(data["parameter_names"], dtype=object),
        "bin_thresholds": np.asarray(data["bin_thresholds"], dtype=float),
    }


def as_unit_coordinates(points: np.ndarray, p0: np.ndarray) -> np.ndarray:
    unit = points / (2.0 * p0)
    return np.clip(unit, 0.0, 1.0)


def subsample_rows(points: np.ndarray, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    if len(points) <= max_rows:
        return points
    idx = rng.choice(len(points), size=max_rows, replace=False)
    return points[idx]


def nearest_distances(query: np.ndarray, cloud: np.ndarray, k: int = 1) -> np.ndarray:
    tree = cKDTree(cloud)
    distances, _ = tree.query(query, k=k, workers=-1)
    if k == 1:
        return np.asarray(distances, dtype=float)
    return np.asarray(distances, dtype=float)[:, 0]


def random_starts(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0.0, 1.0, size=(n, dim))


def cloud_rms_radius(cloud: np.ndarray) -> tuple[np.ndarray, float]:
    center = np.mean(cloud, axis=0)
    diffs = cloud - center
    rms = float(np.sqrt(np.mean(np.sum(diffs * diffs, axis=1))))
    return center, rms


def null_ball(cloud: np.ndarray, rng: np.random.Generator, n: int | None = None) -> np.ndarray:
    n = len(cloud) if n is None else n
    dim = cloud.shape[1]
    center, rms = cloud_rms_radius(cloud)
    radius = rms * math.sqrt((dim + 2.0) / dim)
    directions = rng.normal(size=(n, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = rng.uniform(0.0, 1.0, size=n) ** (1.0 / dim) * radius
    return np.clip(center + directions * radii[:, None], 0.0, 1.0)


def null_box(cloud: np.ndarray, rng: np.random.Generator, n: int | None = None) -> np.ndarray:
    n = len(cloud) if n is None else n
    dim = cloud.shape[1]
    center, rms = cloud_rms_radius(cloud)
    side = math.sqrt(12.0 * rms * rms / dim)
    return np.clip(center + rng.uniform(-0.5 * side, 0.5 * side, size=(n, dim)), 0.0, 1.0)


def null_shuffled_marginals(cloud: np.ndarray, rng: np.random.Generator, n: int | None = None) -> np.ndarray:
    n = len(cloud) if n is None else n
    out = np.empty((n, cloud.shape[1]), dtype=float)
    for j in range(cloud.shape[1]):
        out[:, j] = rng.choice(cloud[:, j], size=n, replace=True)
    return out


def null_covariance_ellipsoid(cloud: np.ndarray, rng: np.random.Generator, n: int | None = None) -> np.ndarray:
    n = len(cloud) if n is None else n
    dim = cloud.shape[1]
    center = np.mean(cloud, axis=0)
    cov = np.cov(cloud, rowvar=False)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    jitter = max(1e-12, float(np.trace(cov)) / max(dim, 1) * 1e-9)
    cov = cov + np.eye(dim) * jitter
    samples = rng.multivariate_normal(np.zeros(dim), cov, size=n)
    return np.clip(center + samples, 0.0, 1.0)


def null_tube(cloud: np.ndarray, rng: np.random.Generator, n: int | None = None) -> np.ndarray:
    n = len(cloud) if n is None else n
    dim = cloud.shape[1]
    center = np.mean(cloud, axis=0)
    _, rms = cloud_rms_radius(cloud)
    direction = rng.normal(size=dim)
    direction /= np.linalg.norm(direction)
    length = min(1.0, 2.0 * rms)
    radius = max(rms / math.sqrt(max(dim - 1, 1)), 1e-6)
    t = rng.uniform(-0.5 * length, 0.5 * length, size=n)
    noise = rng.normal(size=(n, dim))
    noise -= (noise @ direction)[:, None] * direction[None, :]
    noise_norm = np.linalg.norm(noise, axis=1, keepdims=True)
    noise = noise / np.maximum(noise_norm, 1e-12)
    radii = rng.uniform(0.0, 1.0, size=n) ** (1.0 / max(dim - 1, 1)) * radius
    return np.clip(center + t[:, None] * direction[None, :] + radii[:, None] * noise, 0.0, 1.0)


def summarize_distances(distances: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "q05": float(np.quantile(distances, 0.05)),
        "q25": float(np.quantile(distances, 0.25)),
        "q75": float(np.quantile(distances, 0.75)),
        "q95": float(np.quantile(distances, 0.95)),
        "min": float(np.min(distances)),
        "max": float(np.max(distances)),
    }


def run_accessibility(config: AccessibilityConfig) -> dict:
    rng = np.random.default_rng(config.seed)
    loaded = load_neutral_npz(Path(config.npz))
    neutral = loaded["neutral_points"]
    if len(neutral) < 2:
        raise ValueError("Need at least two neutral points for accessibility/null analysis")
    cloud = as_unit_coordinates(neutral, loaded["p0"])
    cloud = subsample_rows(cloud, config.max_cloud, rng)
    starts = random_starts(config.n_random, cloud.shape[1], rng)

    nulls = {
        "nnse_cloud": cloud,
        "compact_ball": null_ball(cloud, rng),
        "compact_box": null_box(cloud, rng),
        "covariance_ellipsoid": null_covariance_ellipsoid(cloud, rng),
        "shuffled_marginals": null_shuffled_marginals(cloud, rng),
        "synthetic_tube": null_tube(cloud, rng),
    }
    distances = {name: nearest_distances(starts, target, config.k_nearest) for name, target in nulls.items()}
    summary = {
        "config": asdict(config),
        "neutral_points_total": int(len(neutral)),
        "neutral_points_used": int(len(cloud)),
        "dimension": int(cloud.shape[1]),
        "distance_summary": {name: summarize_distances(vals) for name, vals in distances.items()},
    }
    return {
        "summary": summary,
        "distances": distances,
        "cloud": cloud,
        "starts": starts,
        "nulls": nulls,
        "loaded": loaded,
    }


def save_accessibility(result: dict, tag: str) -> tuple[Path, Path]:
    out_npz = RESULTS / f"{tag}.npz"
    np.savez_compressed(
        out_npz,
        starts=result["starts"],
        cloud=result["cloud"],
        **{f"dist_{name}": values for name, values in result["distances"].items()},
    )
    out_json = RESULTS / f"{tag}.json"
    out_json.write_text(json.dumps(result["summary"], indent=2))
    return out_npz, out_json


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compare NNSE neutral-set accessibility against null geometries")
    parser.add_argument("--npz", default=None, help="Merged NNSE neutral-set npz. Defaults to the largest available file.")
    parser.add_argument("--model", default="chen2004")
    parser.add_argument("--n-random", type=int, default=10000)
    parser.add_argument("--max-cloud", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    npz = Path(args.npz) if args.npz else choose_biggest_neutral_npz(ROOT, args.model)
    tag = args.tag or f"{args.model}_accessibility_{npz.parent.name}_Nrand{args.n_random:g}"
    result = run_accessibility(
        AccessibilityConfig(
            npz=str(npz),
            n_random=args.n_random,
            max_cloud=args.max_cloud,
            seed=args.seed,
        )
    )
    out_npz, out_json = save_accessibility(result, tag=tag)
    print(f"Using {npz}")
    print(f"Saved {out_npz}")
    print(f"Saved {out_json}")
    for name, stats in result["summary"]["distance_summary"].items():
        print(f"{name:22s} median={stats['median']:.4g}  q05={stats['q05']:.4g}  q95={stats['q95']:.4g}")


if __name__ == "__main__":
    main()

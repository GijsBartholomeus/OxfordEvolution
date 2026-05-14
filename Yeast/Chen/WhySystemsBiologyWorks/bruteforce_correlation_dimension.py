from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parent
RAW_ROOT = ROOT / "results" / "bruteforce_cloud"
FIGURE_ROOT = ROOT / "figures" / "bruteforce_cloud"
SUMMARY_ROOT = ROOT / "results_summaries" / "bruteforce_cloud"


MODEL_LABELS = {
    "chen2004": "Chen 2004",
    "tyson1991": "Tyson 1991",
}


def sample_label(value: float) -> str:
    text = f"{value:g}"
    return text.replace(".", "p").replace("-", "m")


def summary_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"n": 0}
    return {
        "n": int(len(values)),
        "min": float(np.min(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def load_threshold_points(model: str, tag: str, cutoff: float, max_points: int | None, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    paths = sorted((RAW_ROOT / tag).glob(f"{model}_bruteforce_cloud_N=*chunk-*.npz"))
    if not paths:
        raise FileNotFoundError(f"No raw chunks found under {RAW_ROOT / tag} for {model}")

    kept_points = None
    p0 = None
    parameter_names = None
    seen = 0
    saved = 0
    successes = 0
    attempted = 0

    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            points = np.asarray(data["points"], dtype=np.float32)
            values = np.asarray(data["objectives"], dtype=np.float32)
            if p0 is None:
                p0 = np.asarray(data["p0"], dtype=np.float32)
            if parameter_names is None and "parameter_names" in data.files:
                parameter_names = [str(x) for x in data["parameter_names"]]
            if "samples_attempted" in data.files:
                attempted += int(data["samples_attempted"][0])
            successes += len(values)

        mask = np.isfinite(values) & (values <= cutoff) & np.all(np.isfinite(points), axis=1)
        selected = points[mask]
        seen += len(selected)
        if len(selected) == 0:
            continue

        if max_points is None:
            kept_points = selected if kept_points is None else np.vstack([kept_points, selected]).astype(np.float32)
            saved += len(selected)
            continue

        if kept_points is None:
            kept_points = np.empty((0, selected.shape[1]), dtype=np.float32)
        for row in selected:
            saved += 1
            if len(kept_points) < max_points:
                kept_points = np.vstack([kept_points, row[None, :]]).astype(np.float32)
            else:
                j = int(rng.integers(0, saved))
                if j < max_points:
                    kept_points[j] = row

    if p0 is None:
        raise RuntimeError("No p0 found")
    if kept_points is None:
        kept_points = np.empty((0, len(p0)), dtype=np.float32)

    meta = {
        "raw_chunks": len(paths),
        "samples_attempted": attempted,
        "successes_saved_in_chunks": successes,
        "threshold_points_seen": seen,
        "threshold_points_used": int(len(kept_points)),
        "reservoir_max_points": max_points,
        "parameter_names": parameter_names,
    }
    return kept_points, p0, meta


def normalize(points: np.ndarray, p0: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float32) / np.maximum(2.0 * np.asarray(p0, dtype=np.float32)[None, :], np.finfo(np.float32).tiny)


def make_random_cube(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    return rng.random((n, dim), dtype=np.float32)


def make_covariance_ellipsoid_like(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n, dim = points.shape
    center = np.mean(points, axis=0)
    cov = np.cov(points, rowvar=False)
    cov = np.atleast_2d(cov) + np.eye(dim) * 1e-10
    samples = rng.multivariate_normal(np.zeros(dim), cov, size=n).astype(np.float32)
    return np.clip(center[None, :] + samples, 0.0, 1.0).astype(np.float32)


def automatic_radii(points: np.ndarray, n_radii: int, seed: int, pair_sample: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(points)
    if n < 2:
        raise ValueError("Need at least two points")
    i = rng.integers(0, n, size=pair_sample)
    j = rng.integers(0, n, size=pair_sample)
    mask = i != j
    distances = np.linalg.norm(points[i[mask]] - points[j[mask]], axis=1)
    distances = distances[np.isfinite(distances) & (distances > 0)]
    if len(distances) == 0:
        raise RuntimeError("Could not sample positive pair distances")
    lo = float(np.quantile(distances, 0.001))
    hi = float(np.quantile(distances, 0.995))
    lo = max(lo, float(np.min(distances)))
    hi = max(hi, lo * 1.01)
    return np.geomspace(lo, hi, n_radii)


def correlation_curve(points: np.ndarray, radii: np.ndarray) -> dict:
    n = len(points)
    tree = cKDTree(points)
    counts = []
    for radius in radii:
        # count_neighbors includes self-pairs and both directions for self-counts;
        # subtract self-pairs and divide by ordered non-self pairs.
        total = int(tree.count_neighbors(tree, float(radius)))
        nonself = max(0, total - n)
        counts.append(nonself / max(1, n * (n - 1)))
    c = np.asarray(counts, dtype=float)
    log_r = np.log(np.asarray(radii, dtype=float))
    log_c = np.full_like(c, np.nan, dtype=float)
    mask = c > 0
    log_c[mask] = np.log(c[mask])
    slopes = np.full_like(c, np.nan, dtype=float)
    if np.count_nonzero(mask) >= 3:
        slopes[mask] = np.gradient(log_c[mask], log_r[mask])
    return {
        "radii": [float(x) for x in radii],
        "correlation_sum": [float(x) for x in c],
        "local_slope": [float(x) if np.isfinite(x) else None for x in slopes],
    }


def fit_slope(radii: np.ndarray, corr: np.ndarray, c_low: float, c_high: float) -> dict:
    corr = np.asarray(corr, dtype=float)
    mask = np.isfinite(corr) & (corr >= c_low) & (corr <= c_high) & (corr > 0)
    if np.count_nonzero(mask) < 3:
        return {"n": int(np.count_nonzero(mask)), "slope": None, "intercept": None, "r2": None}
    x = np.log(radii[mask])
    y = np.log(corr[mask])
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "n": int(np.count_nonzero(mask)),
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
        "c_low": c_low,
        "c_high": c_high,
        "radius_min": float(np.min(radii[mask])),
        "radius_max": float(np.max(radii[mask])),
    }


def plot_curves(result: dict, figure_path: Path) -> None:
    curves = result["curves"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for name, curve in curves.items():
        radii = np.asarray(curve["radii"], dtype=float)
        corr = np.asarray(curve["correlation_sum"], dtype=float)
        slopes = np.asarray([np.nan if x is None else x for x in curve["local_slope"]], dtype=float)
        axes[0].plot(radii, corr, lw=2.2, label=name)
        axes[1].plot(radii, slopes, lw=2.0, label=name)

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("radius in normalized cube")
    axes[0].set_ylabel("correlation sum C(r)")
    axes[0].set_title("Correlation integral")

    axes[1].set_xscale("log")
    axes[1].set_xlabel("radius in normalized cube")
    axes[1].set_ylabel("local slope d log C / d log r")
    axes[1].set_title("Finite-scale dimension")
    axes[1].axhline(result["ambient_dimension"], color="0.55", ls=":", lw=1.8, label=f"ambient d={result['ambient_dimension']}")

    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=9)

    title = (
        f"{MODEL_LABELS.get(result['model'], result['model'])} correlation dimension diagnostic\n"
        f"{result['tag']}, f <= {result['neutral_cutoff']:g}, n={result['points_used']:,}"
    )
    fig.suptitle(title)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    start = time.time()
    rng = np.random.default_rng(args.seed)
    points_raw, p0, meta = load_threshold_points(args.model, args.tag, args.neutral_cutoff, args.max_points, args.seed)
    points = normalize(points_raw, p0)
    if len(points) < 10:
        raise RuntimeError(f"Only {len(points)} threshold points found; too few for correlation dimension")

    radii = automatic_radii(points, args.n_radii, args.seed + 1, args.radius_pair_sample)
    nulls = {
        "neutral": points,
        "random cube": make_random_cube(len(points), points.shape[1], rng),
        "covariance ellipsoid": make_covariance_ellipsoid_like(points, rng),
    }

    curves = {}
    fits = {}
    for name, cloud in nulls.items():
        curve = correlation_curve(cloud, radii)
        curves[name] = curve
        fits[name] = fit_slope(radii, np.asarray(curve["correlation_sum"], dtype=float), args.fit_c_low, args.fit_c_high)

    result = {
        "model": args.model,
        "tag": args.tag,
        "neutral_cutoff": args.neutral_cutoff,
        "ambient_dimension": int(points.shape[1]),
        "points_seen": int(meta["threshold_points_seen"]),
        "points_used": int(len(points)),
        "load_metadata": meta,
        "fit_window": {"c_low": args.fit_c_low, "c_high": args.fit_c_high},
        "fits": fits,
        "point_coordinate_summary": {
            "mean": [float(x) for x in np.mean(points, axis=0)],
            "std": [float(x) for x in np.std(points, axis=0)],
        },
        "curves": curves,
        "elapsed_seconds": float(time.time() - start),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate finite-sample correlation dimension for brute-force neutral clouds")
    parser.add_argument("--model", required=True, choices=sorted(MODEL_LABELS))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--neutral-cutoff", type=float, required=True)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--n-radii", type=int, default=60)
    parser.add_argument("--radius-pair-sample", type=int, default=500000)
    parser.add_argument("--fit-c-low", type=float, default=1e-4)
    parser.add_argument("--fit-c-high", type=float, default=1e-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-tag", default=None)
    args = parser.parse_args()

    result = run(args)
    out_tag = args.out_tag or f"{args.tag}_f{sample_label(args.neutral_cutoff)}"
    fig_dir = FIGURE_ROOT / args.tag / "correlation_dimension"
    json_dir = SUMMARY_ROOT / args.tag / "correlation_dimension"
    fig_path = fig_dir / f"{args.model}_correlation_dimension_{out_tag}.png"
    json_path = json_dir / f"{args.model}_correlation_dimension_{out_tag}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w") as handle:
        json.dump(result, handle, indent=2)
    plot_curves(result, fig_path)
    print(f"Saved {json_path}")
    print(f"Saved {fig_path}")
    print(json.dumps({k: result[k] for k in ["model", "tag", "neutral_cutoff", "points_seen", "points_used", "fits", "elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()

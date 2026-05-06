from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
STATS_ROOT = RESULTS / "bruteforce_cloud_stats"
OUT_ROOT = RESULTS / "neutral_geometry"


MODEL_LABELS = {
    "chen2004": "Chen 2004",
    "tyson1991": "Tyson 1991",
}


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


def infer_sample_path(model: str, tag: str) -> Path:
    return STATS_ROOT / tag / f"{model}_bruteforce_samples_{tag}.npz"


def output_dir_for(args: argparse.Namespace, npz_path: Path) -> Path:
    label = args.out_tag or args.tag or npz_path.stem
    return OUT_ROOT / label


def finite_points(points: np.ndarray, values: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    points = np.asarray(points, dtype=float)
    mask = np.all(np.isfinite(points), axis=1)
    if values is not None:
        values = np.asarray(values, dtype=float)
        mask &= np.isfinite(values)
        return points[mask], values[mask]
    return points[mask], None


def load_points(args: argparse.Namespace) -> tuple[Path, np.ndarray, np.ndarray | None, np.ndarray, str]:
    npz_path = Path(args.npz) if args.npz else infer_sample_path(args.model, args.tag)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    data = np.load(npz_path, allow_pickle=True)
    if "p0" not in data.files:
        raise KeyError(f"{npz_path} is missing p0, needed to normalize parameter coordinates")
    p0 = np.asarray(data["p0"], dtype=float)

    source = args.source
    values = None
    if source == "neutral":
        if "neutral_points" in data.files and len(data["neutral_points"]):
            points = np.asarray(data["neutral_points"], dtype=float)
            values = np.asarray(data["neutral_objective_values"], dtype=float) if "neutral_objective_values" in data.files else (
                np.asarray(data["neutral_objectives"], dtype=float) if "neutral_objectives" in data.files else None
            )
            source_desc = "neutral_points"
        elif "candidate_vectors" in data.files and "candidate_objective_values" in data.files:
            values_all = np.asarray(data["candidate_objective_values"], dtype=float)
            mask = values_all <= args.neutral_cutoff
            points = np.asarray(data["candidate_vectors"], dtype=float)[mask]
            values = values_all[mask]
            source_desc = f"candidate_vectors objective<= {args.neutral_cutoff:g}"
        elif "all_points" in data.files and "all_objectives" in data.files:
            values_all = np.asarray(data["all_objectives"], dtype=float)
            mask = values_all <= args.neutral_cutoff
            points = np.asarray(data["all_points"], dtype=float)[mask]
            values = values_all[mask]
            source_desc = f"all_points objective<= {args.neutral_cutoff:g}"
        else:
            raise KeyError(f"{npz_path} has no neutral point arrays")
    elif source == "wt-phenotype":
        if "wt_phenotype_points" not in data.files:
            raise KeyError(f"{npz_path} has no wt_phenotype_points; re-run the brute-force merger from the latest code")
        points = np.asarray(data["wt_phenotype_points"], dtype=float)
        values = np.asarray(data["wt_phenotype_objectives"], dtype=float) if "wt_phenotype_objectives" in data.files else None
        source_desc = "wt_phenotype_points"
    elif source == "all":
        if "all_points" not in data.files:
            raise KeyError(f"{npz_path} has no all_points")
        points = np.asarray(data["all_points"], dtype=float)
        values = np.asarray(data["all_objectives"], dtype=float) if "all_objectives" in data.files else None
        source_desc = "all_points"
    else:
        raise ValueError(source)

    points, values = finite_points(points, values)
    if len(points) == 0:
        raise RuntimeError(f"No points loaded from {source_desc}")
    return npz_path, points, values, p0, source_desc


def transform_points(points: np.ndarray, p0: np.ndarray, coordinate_space: str) -> np.ndarray:
    p0 = np.asarray(p0, dtype=float)
    if coordinate_space == "normalized":
        return points / np.maximum(2.0 * p0[None, :], np.finfo(float).tiny)
    if coordinate_space == "log-ratio":
        floor = np.maximum(p0[None, :] * 1e-12, np.finfo(float).tiny)
        return np.log(np.maximum(points, floor) / np.maximum(p0[None, :], np.finfo(float).tiny))
    raise ValueError(coordinate_space)


def covariance_spectrum(points: np.ndarray) -> dict:
    centered = points - np.mean(points, axis=0, keepdims=True)
    if len(points) < 2:
        eigenvalues = np.zeros(points.shape[1], dtype=float)
    else:
        _, singular, _ = np.linalg.svd(centered, full_matrices=False)
        eigenvalues = (singular**2) / max(1, len(points) - 1)
    if len(eigenvalues) < points.shape[1]:
        eigenvalues = np.pad(eigenvalues, (0, points.shape[1] - len(eigenvalues)))
    eigenvalues = np.sort(eigenvalues)[::-1]
    total = float(np.sum(eigenvalues))
    if total <= 0:
        explained = np.zeros_like(eigenvalues)
        d_eff = 0.0
    else:
        explained = eigenvalues / total
        d_eff = float(total**2 / np.sum(eigenvalues**2)) if np.sum(eigenvalues**2) > 0 else 0.0
    cumulative = np.cumsum(explained)

    def dims_for(frac: float) -> int:
        if len(cumulative) == 0 or cumulative[-1] <= 0:
            return 0
        return int(np.searchsorted(cumulative, frac, side="left") + 1)

    return {
        "eigenvalues": [float(x) for x in eigenvalues],
        "explained_fraction": [float(x) for x in explained],
        "cumulative_explained_fraction": [float(x) for x in cumulative],
        "participation_ratio_effective_dimension": d_eff,
        "dims_for_50pct": dims_for(0.50),
        "dims_for_80pct": dims_for(0.80),
        "dims_for_90pct": dims_for(0.90),
        "dims_for_95pct": dims_for(0.95),
        "axis_variance_stats": summary_stats(eigenvalues),
    }


def auto_radii(points: np.ndarray, n_radii: int) -> np.ndarray:
    if len(points) < 2:
        return np.asarray([0.0])
    distances = pdist(points, metric="euclidean")
    distances = distances[np.isfinite(distances) & (distances > 0)]
    if len(distances) == 0:
        return np.asarray([0.0])
    lo = max(float(np.quantile(distances, 0.001)), float(np.min(distances)))
    hi = float(np.max(distances))
    mid_hi = float(np.quantile(distances, 0.995))
    if lo <= 0 or not np.isfinite(lo):
        lo = float(np.min(distances))
    geom = np.geomspace(lo, max(lo * 1.01, mid_hi), max(2, n_radii - 2))
    return np.unique(np.concatenate([[0.0], geom, [hi]]))


def component_curve(points: np.ndarray, radii: np.ndarray) -> dict:
    n = len(points)
    if n == 0:
        return {"radii": [], "components": [], "largest_component_fraction": [], "singleton_fraction": []}
    tree = cKDTree(points)
    components = []
    largest = []
    singletons = []
    for radius in radii:
        if radius <= 0:
            labels = np.arange(n)
            n_components = n
        else:
            pairs = tree.query_pairs(float(radius), output_type="ndarray")
            if len(pairs) == 0:
                labels = np.arange(n)
                n_components = n
            else:
                row = np.concatenate([pairs[:, 0], pairs[:, 1]])
                col = np.concatenate([pairs[:, 1], pairs[:, 0]])
                graph = coo_matrix((np.ones(len(row), dtype=np.int8), (row, col)), shape=(n, n)).tocsr()
                n_components, labels = connected_components(graph, directed=False, return_labels=True)
        counts = np.bincount(labels, minlength=n_components)
        components.append(int(n_components))
        largest.append(float(np.max(counts) / n))
        singletons.append(float(np.sum(counts == 1) / n))

    radii_arr = np.asarray(radii, dtype=float)
    components_arr = np.asarray(components, dtype=int)
    largest_arr = np.asarray(largest, dtype=float)

    def first_radius_for(mask: np.ndarray) -> float | None:
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return None
        return float(radii_arr[idx[0]])

    return {
        "radii": [float(x) for x in radii_arr],
        "components": components,
        "largest_component_fraction": largest,
        "singleton_fraction": singletons,
        "radius_largest_component_ge_50pct": first_radius_for(largest_arr >= 0.50),
        "radius_largest_component_ge_90pct": first_radius_for(largest_arr >= 0.90),
        "radius_all_connected": first_radius_for(components_arr == 1),
    }


def parse_sample_sizes(text: str | None, n: int) -> list[int]:
    if text:
        sizes = [int(x) for x in text.replace(",", " ").split() if x.strip()]
    else:
        base = [25, 50, 100, 200, 500, 1000, 2000, 3000]
        sizes = [x for x in base if x <= n]
        if n not in sizes:
            sizes.append(n)
    return sorted(set(x for x in sizes if 2 <= x <= n))


def stability_analysis(
    points: np.ndarray,
    radii: np.ndarray,
    sample_sizes: list[int],
    reps: int,
    rng: np.random.Generator,
) -> list[dict]:
    rows = []
    for size in sample_sizes:
        d_eff = []
        d80 = []
        r50 = []
        r90 = []
        rall = []
        for _ in range(reps):
            idx = rng.choice(len(points), size=size, replace=False) if len(points) > size else np.arange(len(points))
            sample = points[idx]
            spec = covariance_spectrum(sample)
            curve = component_curve(sample, radii)
            d_eff.append(spec["participation_ratio_effective_dimension"])
            d80.append(spec["dims_for_80pct"])
            for store, key in [(r50, "radius_largest_component_ge_50pct"), (r90, "radius_largest_component_ge_90pct"), (rall, "radius_all_connected")]:
                value = curve[key]
                store.append(np.nan if value is None else value)
        rows.append(
            {
                "sample_size": int(size),
                "replicates": int(reps),
                "effective_dimension": summary_stats(np.asarray(d_eff)),
                "dims_for_80pct": summary_stats(np.asarray(d80)),
                "radius_largest_component_ge_50pct": summary_stats(np.asarray(r50)),
                "radius_largest_component_ge_90pct": summary_stats(np.asarray(r90)),
                "radius_all_connected": summary_stats(np.asarray(rall)),
            }
        )
    return rows


def plot_results(out_dir: Path, model: str, tag: str, points: np.ndarray, spectrum: dict, curve: dict, stability: list[dict]) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    ax_eig, ax_cum, ax_comp, ax_lcc, ax_deff, ax_radii = axes.ravel()

    eig = np.asarray(spectrum["eigenvalues"], dtype=float)
    explained = np.asarray(spectrum["explained_fraction"], dtype=float)
    cumulative = np.asarray(spectrum["cumulative_explained_fraction"], dtype=float)
    ranks = np.arange(1, len(eig) + 1)
    ax_eig.plot(ranks, np.maximum(explained, np.finfo(float).tiny), marker="o", ms=3, lw=1.5)
    ax_eig.set_yscale("log")
    ax_eig.set_xlabel("PCA axis")
    ax_eig.set_ylabel("variance fraction")
    ax_eig.set_title(f"Covariance spectrum\n$d_{{eff}}$={spectrum['participation_ratio_effective_dimension']:.2f}")

    ax_cum.plot(ranks, cumulative, marker="o", ms=3, lw=1.5)
    ax_cum.axhline(0.8, color="0.7", ls="--", lw=1)
    ax_cum.axhline(0.95, color="0.7", ls=":", lw=1)
    ax_cum.set_ylim(0, 1.02)
    ax_cum.set_xlabel("PCA axes retained")
    ax_cum.set_ylabel("cumulative variance")
    ax_cum.set_title("Variance captured")

    radii = np.asarray(curve["radii"], dtype=float)
    components = np.asarray(curve["components"], dtype=float)
    largest = np.asarray(curve["largest_component_fraction"], dtype=float)
    singletons = np.asarray(curve["singleton_fraction"], dtype=float)
    positive = radii > 0
    ax_comp.plot(radii[positive], components[positive], color="black", lw=2)
    ax_comp.set_xscale("log")
    ax_comp.set_yscale("log")
    ax_comp.set_xlabel("epsilon radius")
    ax_comp.set_ylabel("# connected components")
    ax_comp.set_title(f"H0 component curve\nn={len(points):,}")

    ax_lcc.plot(radii[positive], largest[positive], color="#1f77b4", lw=2, label="largest component")
    ax_lcc.plot(radii[positive], singletons[positive], color="#c44e52", lw=2, label="singletons")
    ax_lcc.set_xscale("log")
    ax_lcc.set_ylim(-0.02, 1.02)
    ax_lcc.set_xlabel("epsilon radius")
    ax_lcc.set_ylabel("fraction")
    ax_lcc.set_title("Graph connectivity")
    ax_lcc.legend(frameon=False, fontsize=8)

    sizes = np.asarray([row["sample_size"] for row in stability], dtype=float)
    med = np.asarray([row["effective_dimension"]["median"] for row in stability], dtype=float)
    q25 = np.asarray([row["effective_dimension"].get("q25", np.nan) for row in stability], dtype=float)
    q75 = np.asarray([row["effective_dimension"].get("q75", np.nan) for row in stability], dtype=float)
    ax_deff.plot(sizes, med, marker="o", color="black")
    ax_deff.fill_between(sizes, q25, q75, color="0.8")
    ax_deff.set_xscale("log")
    ax_deff.set_xlabel("subsample size")
    ax_deff.set_ylabel("participation-ratio dimension")
    ax_deff.set_title("Sample-size stability")

    for key, label, color in [
        ("radius_largest_component_ge_50pct", "LCC >= 50%", "#4c78a8"),
        ("radius_largest_component_ge_90pct", "LCC >= 90%", "#f58518"),
        ("radius_all_connected", "all connected", "#54a24b"),
    ]:
        vals = np.asarray([row[key].get("median", np.nan) for row in stability], dtype=float)
        ax_radii.plot(sizes, vals, marker="o", lw=1.5, label=label, color=color)
    ax_radii.set_xscale("log")
    ax_radii.set_yscale("log")
    ax_radii.set_xlabel("subsample size")
    ax_radii.set_ylabel("epsilon radius")
    ax_radii.set_title("Connectivity scale stability")
    ax_radii.legend(frameon=False, fontsize=8)

    fig.suptitle(f"{MODEL_LABELS.get(model, model)} neutral geometry: {tag}", fontsize=14)
    out = out_dir / f"{model}_neutral_geometry_{tag}.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def main(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    npz_path, raw_points, objective_values, p0, source_desc = load_points(args)
    points = transform_points(raw_points, p0, args.coordinate_space)

    if len(points) > args.max_points:
        idx = rng.choice(len(points), size=args.max_points, replace=False)
        points = points[idx]
        objective_values = objective_values[idx] if objective_values is not None and len(objective_values) == len(raw_points) else objective_values

    if len(points) < 2:
        raise RuntimeError(f"Need at least two points for geometry analysis, got {len(points)}")

    connected_n = min(args.max_connected_points, len(points))
    connected_idx = rng.choice(len(points), size=connected_n, replace=False) if len(points) > connected_n else np.arange(len(points))
    connected_points = points[connected_idx]
    radii = auto_radii(connected_points, args.n_radii)

    spectrum = covariance_spectrum(points)
    curve = component_curve(connected_points, radii)
    sample_sizes = parse_sample_sizes(args.sample_sizes, connected_n)
    stability = stability_analysis(connected_points, radii, sample_sizes, args.stability_reps, rng)

    tag = args.out_tag or args.tag or npz_path.stem
    out_dir = output_dir_for(args, npz_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_results(out_dir, args.model, tag, connected_points, spectrum, curve, stability)

    result = {
        "model": args.model,
        "label": MODEL_LABELS.get(args.model, args.model),
        "tag": tag,
        "source_npz": str(npz_path),
        "source": source_desc,
        "coordinate_space": args.coordinate_space,
        "neutral_cutoff": args.neutral_cutoff if args.source == "neutral" else None,
        "points_available": int(len(raw_points)),
        "points_analyzed_covariance": int(len(points)),
        "points_analyzed_connectedness": int(len(connected_points)),
        "ambient_dimension": int(points.shape[1]),
        "objective_value_stats": summary_stats(objective_values) if objective_values is not None else None,
        "covariance_spectrum": spectrum,
        "connectedness_h0_curve": curve,
        "sample_size_stability": stability,
        "plot": str(plot_path),
    }
    json_path = out_dir / f"{args.model}_neutral_geometry_{tag}.json"
    json_path.write_text(json.dumps(result, indent=2))
    print(f"Saved {json_path}")
    print(f"Saved {plot_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neutral-set connectedness and covariance-spectrum geometry analysis")
    parser.add_argument("--model", required=True, choices=sorted(MODEL_LABELS))
    parser.add_argument("--tag", default=None, help="bruteforce_cloud_stats tag, e.g. tyson_bfc_1e8")
    parser.add_argument("--npz", default=None, help="explicit NPZ path; overrides --tag inference")
    parser.add_argument("--out-tag", default=None)
    parser.add_argument("--source", choices=["neutral", "wt-phenotype", "all"], default="neutral")
    parser.add_argument("--neutral-cutoff", type=float, default=0.05)
    parser.add_argument("--coordinate-space", choices=["normalized", "log-ratio"], default="normalized")
    parser.add_argument("--max-points", type=int, default=100000)
    parser.add_argument("--max-connected-points", type=int, default=3000)
    parser.add_argument("--n-radii", type=int, default=28)
    parser.add_argument("--sample-sizes", default=None, help="comma/space separated sizes for stability curves")
    parser.add_argument("--stability-reps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())

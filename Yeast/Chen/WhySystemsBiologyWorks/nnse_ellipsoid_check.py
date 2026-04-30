from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from nnse_accessibility import choose_biggest_neutral_npz
from nnse_sloppy_projection import summarize_vector
from nnse_sloppy_subspace import ROOT, compute_tyson_hessian, load_neutral_points, sloppy_basis


RESULTS = ROOT / "results" / "ellipsoid_check"
RESULTS.mkdir(parents=True, exist_ok=True)


@dataclass
class EllipsoidCheckConfig:
    model: str = "tyson1991"
    npz: str | None = None
    neutral_threshold: float = 0.05
    k_sloppy: int = 3
    seed: int = 42
    n_points: int | None = None
    t_end: float = 100.0
    n_time: int = 501
    tag: str | None = None


def finite_positive_rows(points: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(points), axis=1) & np.all(points > 0, axis=1)


def select_points(points: np.ndarray, values: np.ndarray, n_points: int | None, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.where(finite_positive_rows(points))[0]
    if len(idx) == 0:
        raise ValueError("No finite positive points available after thresholding")
    if n_points is not None and n_points > 0 and len(idx) > n_points:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=n_points, replace=False))
    return idx, points[idx], values[idx]


def pca(centered: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    total = float(np.sum(eigvals))
    frac = eigvals / total if total > 0 else np.full_like(eigvals, np.nan)
    return eigvals, eigvecs, frac


def cumulative_dimension(frac: np.ndarray, target: float) -> int | None:
    finite = frac[np.isfinite(frac)]
    if len(finite) == 0:
        return None
    return int(np.searchsorted(np.cumsum(finite), target, side="left") + 1)


def axis_stats(coords: np.ndarray, names: list[str]) -> list[dict]:
    rows = []
    for idx, name in enumerate(names):
        x = coords[:, idx]
        rows.append(
            {
                "axis": name,
                "rms": float(np.sqrt(np.mean(x * x))),
                "median_abs": float(np.median(np.abs(x))),
                "q95_abs": float(np.quantile(np.abs(x), 0.95)),
                "max_abs": float(np.max(np.abs(x))),
            }
        )
    return rows


def ellipsoid_radii(coords: np.ndarray, axis_lengths: np.ndarray) -> np.ndarray:
    safe = np.where(axis_lengths > 0, axis_lengths, np.inf)
    return np.sqrt(np.sum((coords / safe[None, :]) ** 2, axis=1))


def analyze_ellipsoid(config: EllipsoidCheckConfig) -> dict:
    if config.model != "tyson1991":
        raise NotImplementedError("Ellipsoid check currently uses the Tyson Hessian implementation")

    npz_path = Path(config.npz) if config.npz else choose_biggest_neutral_npz(ROOT, config.model)
    loaded = load_neutral_points(npz_path, config.neutral_threshold)
    selected_idx, selected, values = select_points(
        loaded["points"], loaded["values"], config.n_points, config.seed
    )

    wt_res = compute_tyson_hessian(
        loaded["p0"],
        loaded["parameter_names"],
        t_end=config.t_end,
        n_time=config.n_time,
    )
    supported_idx = wt_res["supported_indices"]
    supported_names = wt_res["mapped_names"]
    unsupported_names = wt_res["unsupported_names"]
    eigvecs = wt_res["eigvecs"]
    eigvals = wt_res["eigvals"]
    wt_basis = sloppy_basis(eigvecs, config.k_sloppy)

    supported = selected[:, supported_idx]
    wt = loaded["p0"][supported_idx]
    log_delta = np.log(supported) - np.log(wt)[None, :]

    # Coordinates in the full WT Hessian eigenbasis. Columns are stiff-to-sloppy.
    hessian_coords = log_delta @ eigvecs
    axis_labels = [f"hessian_axis_{idx + 1}" for idx in range(hessian_coords.shape[1])]
    sloppy_coords = hessian_coords[:, -config.k_sloppy :]
    stiff_coords = hessian_coords[:, : -config.k_sloppy] if config.k_sloppy < hessian_coords.shape[1] else np.empty((len(hessian_coords), 0))

    parallel = log_delta @ wt_basis @ wt_basis.T
    orthogonal = log_delta - parallel
    total_norm = np.linalg.norm(log_delta, axis=1)
    orthogonal_norm = np.linalg.norm(orthogonal, axis=1)
    parallel_norm = np.linalg.norm(parallel, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        residual_fraction = np.where(total_norm > 0, orthogonal_norm / total_norm, 0.0)
        explained_fraction = np.where(total_norm > 0, parallel_norm**2 / total_norm**2, 1.0)

    pca_eigvals, pca_eigvecs, pca_frac = pca(log_delta - np.mean(log_delta, axis=0))

    # WT-centered, WT-Hessian-axis-aligned ellipsoids. These are descriptive:
    # q95 axes show a robust ellipsoid; max axes plus max radius encloses all points.
    q95_axes = np.quantile(np.abs(hessian_coords), 0.95, axis=0)
    max_axes = np.max(np.abs(hessian_coords), axis=0)
    q95_radii = ellipsoid_radii(hessian_coords, q95_axes)
    max_axis_radii = ellipsoid_radii(hessian_coords, max_axes)

    sloppy_q95_axes = np.quantile(np.abs(sloppy_coords), 0.95, axis=0)
    sloppy_max_axes = np.max(np.abs(sloppy_coords), axis=0)
    sloppy_q95_radii = ellipsoid_radii(sloppy_coords, sloppy_q95_axes)
    sloppy_max_axis_radii = ellipsoid_radii(sloppy_coords, sloppy_max_axes)

    summary = {
        "config": asdict(config),
        "npz": str(npz_path),
        "point_source": loaded["source"],
        "points_in_file": int(len(loaded["points"])),
        "points_sampled": int(len(selected)),
        "supported_parameter_names": supported_names,
        "unsupported_parameter_names": unsupported_names,
        "wt_hessian_eigenvalues_descending": eigvals.tolist(),
        "residual_fraction_outside_wt_sloppy": summarize_vector("orthogonal_norm / total_norm", residual_fraction),
        "explained_fraction_by_wt_sloppy": summarize_vector("parallel_norm^2 / total_norm^2", explained_fraction),
        "orthogonal_log_distance": summarize_vector("||orthogonal residual||", orthogonal_norm),
        "parallel_log_distance": summarize_vector("||projection onto WT sloppy subspace||", parallel_norm),
        "total_log_distance_to_wt": summarize_vector("||log(theta)-log(theta_wt)||", total_norm),
        "stiff_coordinate_norm": summarize_vector("||WT-Hessian stiff coordinates||", np.linalg.norm(stiff_coords, axis=1) if stiff_coords.size else np.zeros(len(selected))),
        "sloppy_coordinate_norm": summarize_vector("||WT-Hessian sloppy coordinates||", np.linalg.norm(sloppy_coords, axis=1)),
        "pca_explained_fraction": pca_frac.tolist(),
        "pca_dimensions_for_80pct": cumulative_dimension(pca_frac, 0.80),
        "pca_dimensions_for_90pct": cumulative_dimension(pca_frac, 0.90),
        "pca_dimensions_for_95pct": cumulative_dimension(pca_frac, 0.95),
        "full_hessian_basis_axis_stats": axis_stats(hessian_coords, axis_labels),
        "full_wt_centered_ellipsoid_radii_using_q95_abs_axes": summarize_vector("sqrt(sum((coord/q95_abs_axis)^2))", q95_radii),
        "full_wt_centered_ellipsoid_radii_using_max_abs_axes": summarize_vector("sqrt(sum((coord/max_abs_axis)^2))", max_axis_radii),
        "full_ellipsoid_scale_needed_with_max_abs_axes_to_contain_all": float(np.max(max_axis_radii)) if len(max_axis_radii) else None,
        "sloppy_only_ellipsoid_radii_using_q95_abs_axes": summarize_vector("sqrt(sum((sloppy_coord/q95_abs_axis)^2))", sloppy_q95_radii),
        "sloppy_only_ellipsoid_radii_using_max_abs_axes": summarize_vector("sqrt(sum((sloppy_coord/max_abs_axis)^2))", sloppy_max_axis_radii),
        "sloppy_only_ellipsoid_scale_needed_with_max_abs_axes_to_contain_all": float(np.max(sloppy_max_axis_radii)) if len(sloppy_max_axis_radii) else None,
    }

    return {
        "summary": summary,
        "selected_indices": selected_idx,
        "objective_values": values,
        "selected_points": selected,
        "supported_points": supported,
        "supported_parameter_names": np.asarray(supported_names, dtype=object),
        "unsupported_parameter_names": np.asarray(unsupported_names, dtype=object),
        "wt_supported": wt,
        "wt_hessian_eigvals": eigvals,
        "wt_hessian_eigvecs": eigvecs,
        "log_delta": log_delta,
        "hessian_coords": hessian_coords,
        "sloppy_coords": sloppy_coords,
        "stiff_coords": stiff_coords,
        "residual_fraction": residual_fraction,
        "explained_fraction": explained_fraction,
        "orthogonal_norm": orthogonal_norm,
        "parallel_norm": parallel_norm,
        "total_norm": total_norm,
        "pca_eigvals": pca_eigvals,
        "pca_eigvecs": pca_eigvecs,
        "pca_explained_fraction": pca_frac,
        "q95_axes": q95_axes,
        "max_axes": max_axes,
        "q95_radii": q95_radii,
        "max_axis_radii": max_axis_radii,
        "sloppy_q95_axes": sloppy_q95_axes,
        "sloppy_max_axes": sloppy_max_axes,
        "sloppy_q95_radii": sloppy_q95_radii,
        "sloppy_max_axis_radii": sloppy_max_axis_radii,
    }


def plot_result(result: dict, fig_path: Path) -> None:
    residual = np.asarray(result["residual_fraction"], dtype=float)
    explained = np.asarray(result["explained_fraction"], dtype=float)
    sloppy = np.asarray(result["sloppy_coords"], dtype=float)
    objectives = np.asarray(result["objective_values"], dtype=float)
    pca_frac = np.asarray(result["pca_explained_fraction"], dtype=float)
    axis_rms = np.sqrt(np.mean(np.asarray(result["hessian_coords"], dtype=float) ** 2, axis=0))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes[0, 0].hist(residual[np.isfinite(residual)], bins=35, color="#4c78a8", alpha=0.85)
    axes[0, 0].set_xlabel("orthogonal residual fraction")
    axes[0, 0].set_ylabel("count")
    axes[0, 0].set_title("Outside WT sloppy subspace")

    axes[0, 1].hist(explained[np.isfinite(explained)], bins=35, color="#59a14f", alpha=0.85)
    axes[0, 1].set_xlabel("explained fraction")
    axes[0, 1].set_ylabel("count")
    axes[0, 1].set_title("Explained by WT sloppy subspace")

    k = min(8, len(axis_rms))
    axes[0, 2].bar(np.arange(1, k + 1), axis_rms[:k], color="#b07aa1", alpha=0.85)
    axes[0, 2].set_xlabel("WT Hessian axis, stiff to sloppy")
    axes[0, 2].set_ylabel("RMS coordinate")
    axes[0, 2].set_title("Spread in WT Hessian basis")

    if sloppy.shape[1] >= 2:
        sc = axes[1, 0].scatter(sloppy[:, 0], sloppy[:, 1], c=objectives, cmap="viridis", s=18, alpha=0.85)
        axes[1, 0].set_xlabel("sloppy coord 1")
        axes[1, 0].set_ylabel("sloppy coord 2")
        fig.colorbar(sc, ax=axes[1, 0], label="objective")
    else:
        axes[1, 0].hist(sloppy[:, 0], bins=35)
        axes[1, 0].set_xlabel("sloppy coord 1")
    axes[1, 0].set_title("WT sloppy projection")

    k_pca = min(8, len(pca_frac))
    axes[1, 1].bar(np.arange(1, k_pca + 1), pca_frac[:k_pca], color="#f28e2b", alpha=0.85)
    axes[1, 1].set_xlabel("PCA axis")
    axes[1, 1].set_ylabel("variance fraction")
    axes[1, 1].set_title("PCA of WT-centered log cloud")

    axes[1, 2].plot(np.arange(1, k_pca + 1), np.cumsum(pca_frac[:k_pca]), marker="o", color="#e15759")
    axes[1, 2].set_ylim(0, 1.03)
    axes[1, 2].set_xlabel("PCA dimensions")
    axes[1, 2].set_ylabel("cumulative variance")
    axes[1, 2].set_title("Cloud dimensionality")

    fig.savefig(fig_path, dpi=220)
    plt.close(fig)


def save_result(result: dict, tag: str) -> tuple[Path, Path, Path]:
    npz_path = RESULTS / f"{tag}.npz"
    json_path = RESULTS / f"{tag}.json"
    fig_path = RESULTS / f"{tag}.png"
    arrays = {key: value for key, value in result.items() if key != "summary"}
    np.savez_compressed(npz_path, **arrays)
    json_path.write_text(json.dumps(result["summary"], indent=2))
    plot_result(result, fig_path)
    return npz_path, json_path, fig_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether neutral points form a WT-centered sloppy ellipsoid")
    parser.add_argument("--model", default="tyson1991")
    parser.add_argument("--npz", default=None)
    parser.add_argument("--neutral-threshold", type=float, default=0.05)
    parser.add_argument("--k-sloppy", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-points", type=int, default=None)
    parser.add_argument("--t-end", type=float, default=100.0)
    parser.add_argument("--n-time", type=int, default=501)
    parser.add_argument("--tag", default=None)
    config = EllipsoidCheckConfig(**vars(parser.parse_args()))
    result = analyze_ellipsoid(config)
    tag = config.tag or f"{config.model}_ellipsoid_thr{config.neutral_threshold:g}_k{config.k_sloppy}"
    out_npz, out_json, out_png = save_result(result, tag)
    print(f"Saved {out_npz}")
    print(f"Saved {out_json}")
    print(f"Saved {out_png}")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()

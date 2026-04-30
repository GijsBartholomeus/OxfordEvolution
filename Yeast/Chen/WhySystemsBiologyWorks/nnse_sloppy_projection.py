from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from nnse_accessibility import choose_biggest_neutral_npz
from nnse_sloppy_subspace import ROOT, compute_tyson_hessian, load_neutral_points, sloppy_basis


RESULTS = ROOT / "results" / "sloppy_projection"
RESULTS.mkdir(parents=True, exist_ok=True)


@dataclass
class SloppyProjectionConfig:
    model: str = "tyson1991"
    npz: str | None = None
    neutral_threshold: float = 1.0
    k_sloppy: int = 3
    n_points: int | None = None
    seed: int = 42
    t_end: float = 100.0
    n_time: int = 501
    tag: str | None = None


def finite_positive_rows(points: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(points), axis=1) & np.all(points > 0, axis=1)


def select_points(points: np.ndarray, values: np.ndarray, n_points: int | None, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.where(finite_positive_rows(points))[0]
    if len(idx) == 0:
        raise ValueError("No finite positive points available")
    if n_points is not None and n_points > 0 and len(idx) > n_points:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=n_points, replace=False))
    return idx, points[idx], values[idx]


def summarize_vector(name: str, x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"name": name, "n": 0}
    return {
        "name": name,
        "n": int(len(x)),
        "min": float(np.min(x)),
        "q05": float(np.quantile(x, 0.05)),
        "q25": float(np.quantile(x, 0.25)),
        "median": float(np.median(x)),
        "q75": float(np.quantile(x, 0.75)),
        "q95": float(np.quantile(x, 0.95)),
        "max": float(np.max(x)),
    }


def pca_basis(centered: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    return eigvals, eigvecs[:, :k], eigvecs


def subspace_chordal(u: np.ndarray, v: np.ndarray) -> float:
    singular = np.linalg.svd(u.T @ v, compute_uv=False)
    return float(np.sqrt(max(0.0, min(u.shape[1], v.shape[1]) - np.sum(singular * singular))))


def principal_angles_degrees(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    singular = np.linalg.svd(u.T @ v, compute_uv=False)
    return np.degrees(np.arccos(np.clip(singular, -1.0, 1.0)))


def analyze_projection(config: SloppyProjectionConfig) -> dict:
    if config.model != "tyson1991":
        raise NotImplementedError("Projection analysis currently uses the Tyson Hessian implementation")

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
    wt_basis = sloppy_basis(wt_res["eigvecs"], config.k_sloppy)

    supported_points = selected[:, supported_idx]
    wt = loaded["p0"][supported_idx]
    log_delta = np.log(supported_points) - np.log(wt)[None, :]

    parallel = log_delta @ wt_basis @ wt_basis.T
    orthogonal = log_delta - parallel
    total_norm = np.linalg.norm(log_delta, axis=1)
    parallel_norm = np.linalg.norm(parallel, axis=1)
    orthogonal_norm = np.linalg.norm(orthogonal, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        residual_fraction = np.where(total_norm > 0, orthogonal_norm / total_norm, 0.0)
        explained_fraction = np.where(total_norm > 0, (parallel_norm**2) / (total_norm**2), 1.0)

    pca_eigvals, pca_top, pca_all = pca_basis(log_delta - np.mean(log_delta, axis=0), config.k_sloppy)
    pca_chordal = subspace_chordal(pca_top, wt_basis)
    pca_angles = principal_angles_degrees(pca_top, wt_basis)
    total_variance = float(np.sum(pca_eigvals))
    pca_explained = pca_eigvals / total_variance if total_variance > 0 else np.full_like(pca_eigvals, np.nan)

    sloppy_coords = log_delta @ wt_basis
    if sloppy_coords.shape[1] >= 2:
        scatter_xy = sloppy_coords[:, :2]
    else:
        scatter_xy = np.column_stack([sloppy_coords[:, 0], np.zeros(len(sloppy_coords))])

    summary = {
        "config": asdict(config),
        "npz": str(npz_path),
        "point_source": loaded["source"],
        "points_in_file": int(len(loaded["points"])),
        "points_sampled": int(len(selected)),
        "supported_parameter_names": supported_names,
        "unsupported_parameter_names": unsupported_names,
        "wt_hessian_eigenvalues_descending": wt_res["eigvals"].tolist(),
        "residual_fraction": summarize_vector("orthogonal_norm / total_norm", residual_fraction),
        "explained_fraction": summarize_vector("parallel_norm^2 / total_norm^2", explained_fraction),
        "total_log_distance_to_wt": summarize_vector("||log(theta)-log(theta_wt)||", total_norm),
        "parallel_log_distance": summarize_vector("||projection onto WT sloppy subspace||", parallel_norm),
        "orthogonal_log_distance": summarize_vector("||orthogonal residual||", orthogonal_norm),
        "pca_top_vs_wt_sloppy_chordal": pca_chordal,
        "pca_top_vs_wt_sloppy_angles_degrees": pca_angles.tolist(),
        "pca_explained_fraction": pca_explained.tolist(),
    }

    return {
        "summary": summary,
        "selected_indices": selected_idx,
        "objective_values": values,
        "selected_points": selected,
        "supported_points": supported_points,
        "supported_parameter_names": np.asarray(supported_names, dtype=object),
        "unsupported_parameter_names": np.asarray(unsupported_names, dtype=object),
        "wt_supported": wt,
        "wt_sloppy_basis": wt_basis,
        "wt_hessian_eigvals": wt_res["eigvals"],
        "log_delta": log_delta,
        "parallel": parallel,
        "orthogonal": orthogonal,
        "total_norm": total_norm,
        "parallel_norm": parallel_norm,
        "orthogonal_norm": orthogonal_norm,
        "residual_fraction": residual_fraction,
        "explained_fraction": explained_fraction,
        "sloppy_coords": sloppy_coords,
        "scatter_xy": scatter_xy,
        "pca_eigvals": pca_eigvals,
        "pca_eigvecs": pca_all,
        "pca_explained_fraction": pca_explained,
    }


def save_result(result: dict, tag: str) -> tuple[Path, Path, Path]:
    npz_path = RESULTS / f"{tag}.npz"
    json_path = RESULTS / f"{tag}.json"
    fig_path = RESULTS / f"{tag}.png"
    arrays = {k: v for k, v in result.items() if k != "summary"}
    np.savez_compressed(npz_path, **arrays)
    json_path.write_text(json.dumps(result["summary"], indent=2))
    plot_projection(result, fig_path)
    return npz_path, json_path, fig_path


def plot_projection(result: dict, fig_path: Path) -> None:
    residual = np.asarray(result["residual_fraction"], dtype=float)
    explained = np.asarray(result["explained_fraction"], dtype=float)
    xy = np.asarray(result["scatter_xy"], dtype=float)
    values = np.asarray(result["objective_values"], dtype=float)
    pca_explained = np.asarray(result["pca_explained_fraction"], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    axes[0, 0].hist(residual[np.isfinite(residual)], bins=40, color="#4c78a8", alpha=0.85)
    axes[0, 0].set_xlabel("orthogonal residual fraction")
    axes[0, 0].set_ylabel("count")
    axes[0, 0].set_title("Distance outside WT sloppy subspace")

    axes[0, 1].hist(explained[np.isfinite(explained)], bins=40, color="#59a14f", alpha=0.85)
    axes[0, 1].set_xlabel("explained fraction")
    axes[0, 1].set_ylabel("count")
    axes[0, 1].set_title("WT sloppy-subspace explained displacement")

    sc = axes[1, 0].scatter(xy[:, 0], xy[:, 1], c=values, s=16, cmap="viridis", alpha=0.8)
    axes[1, 0].axhline(0, color="0.8", lw=0.8)
    axes[1, 0].axvline(0, color="0.8", lw=0.8)
    axes[1, 0].set_xlabel("WT sloppy coordinate 1")
    axes[1, 0].set_ylabel("WT sloppy coordinate 2")
    axes[1, 0].set_title("Projection into WT sloppy plane")
    fig.colorbar(sc, ax=axes[1, 0], label="objective")

    k = min(10, len(pca_explained))
    axes[1, 1].bar(np.arange(1, k + 1), pca_explained[:k], color="#f28e2b", alpha=0.85)
    axes[1, 1].set_xlabel("PCA axis")
    axes[1, 1].set_ylabel("variance fraction")
    axes[1, 1].set_title("PCA of log-displacements from WT")

    fig.savefig(fig_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test whether Tyson neutral points lie near the WT sloppy subspace")
    parser.add_argument("--model", default="tyson1991")
    parser.add_argument("--npz", default=None)
    parser.add_argument("--neutral-threshold", type=float, default=1.0)
    parser.add_argument("--k-sloppy", type=int, default=3)
    parser.add_argument("--n-points", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t-end", type=float, default=100.0)
    parser.add_argument("--n-time", type=int, default=501)
    parser.add_argument("--tag", default=None)
    config = SloppyProjectionConfig(**vars(parser.parse_args()))
    result = analyze_projection(config)
    tag = config.tag or f"{config.model}_sloppy_projection_thr{config.neutral_threshold:g}_k{config.k_sloppy}"
    out_npz, out_json, out_png = save_result(result, tag)
    print(f"Saved {out_npz}")
    print(f"Saved {out_json}")
    print(f"Saved {out_png}")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()

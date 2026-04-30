from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from nnse_sloppy_subspace import (
    ROOT,
    chordal_distance,
    compute_tyson_hessian,
    principal_angles,
    sloppy_basis,
)
from wsbw_nnse import get_spec, setup_rr
from wsbw_pipeline import prepare_models


RESULTS = ROOT / "results" / "sloppy_random_cube"
RESULTS.mkdir(parents=True, exist_ok=True)


@dataclass
class RandomCubeConfig:
    model: str = "tyson1991"
    n_points: int = 50
    k_sloppy: int = 3
    seed: int = 42
    t_end: float = 100.0
    n_time: int = 501
    tag: str | None = None


def summarize_vector(name: str, values: np.ndarray, skip_zero: bool = False) -> dict:
    x = np.asarray(values, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if skip_zero:
        x = x[x > 0]
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


def load_p0_and_names(model: str) -> tuple[np.ndarray, list[str]]:
    audit = prepare_models()
    spec = get_spec(model)
    params = audit[spec.key]["free_parameters"]
    _, defaults, _ = setup_rr(spec, audit[spec.key]["promoted_sbml"], params)
    p0 = np.asarray([defaults[pid] for pid in params], dtype=float)
    return p0, params


def analyze_random_cube(config: RandomCubeConfig) -> dict:
    if config.model != "tyson1991":
        raise NotImplementedError("Random-cube sloppy analysis currently uses the Tyson Hessian implementation")

    rng = np.random.default_rng(config.seed)
    p0, parameter_names = load_p0_and_names(config.model)
    points = 2.0 * p0[None, :] * rng.uniform(0.0, 1.0, size=(config.n_points, len(p0)))

    wt_res = compute_tyson_hessian(p0, parameter_names, t_end=config.t_end, n_time=config.n_time)
    supported_idx = wt_res["supported_indices"]
    mapped_names = wt_res["mapped_names"]
    unsupported_names = wt_res["unsupported_names"]
    wt_basis = sloppy_basis(wt_res["eigvecs"], config.k_sloppy)

    eigvals = []
    bases = []
    failed = []
    start = time.time()
    for i, vector in enumerate(points):
        try:
            res = compute_tyson_hessian(vector, parameter_names, t_end=config.t_end, n_time=config.n_time)
            eigvals.append(res["eigvals"])
            bases.append(sloppy_basis(res["eigvecs"], config.k_sloppy))
        except Exception as exc:
            failed.append({"index": int(i), "error": str(exc)})
            eigvals.append(np.full(len(mapped_names), np.nan))
            bases.append(np.full((len(mapped_names), min(config.k_sloppy, len(mapped_names))), np.nan))

    bases_arr = np.asarray(bases, dtype=float)
    eigvals_arr = np.asarray(eigvals, dtype=float)
    valid = np.array([np.all(np.isfinite(basis)) for basis in bases_arr], dtype=bool)
    valid_indices = np.where(valid)[0]
    n_valid = len(valid_indices)

    pairwise_chordal = np.full((n_valid, n_valid), np.nan, dtype=float)
    pairwise_max_angle = np.full((n_valid, n_valid), np.nan, dtype=float)
    for a_pos, a in enumerate(valid_indices):
        for b_pos, b in enumerate(valid_indices):
            if b_pos < a_pos:
                pairwise_chordal[a_pos, b_pos] = pairwise_chordal[b_pos, a_pos]
                pairwise_max_angle[a_pos, b_pos] = pairwise_max_angle[b_pos, a_pos]
                continue
            angles = principal_angles(bases_arr[a], bases_arr[b])
            pairwise_chordal[a_pos, b_pos] = chordal_distance(bases_arr[a], bases_arr[b])
            pairwise_max_angle[a_pos, b_pos] = float(np.max(angles))

    wt_chordal = np.full(config.n_points, np.nan, dtype=float)
    wt_max_angle = np.full(config.n_points, np.nan, dtype=float)
    for i in valid_indices:
        angles = principal_angles(bases_arr[i], wt_basis)
        wt_chordal[i] = chordal_distance(bases_arr[i], wt_basis)
        wt_max_angle[i] = float(np.max(angles))

    supported_points = points[:, supported_idx]
    p0_supported = p0[supported_idx]
    eps = np.finfo(float).tiny
    log_delta = np.log(np.maximum(supported_points, eps)) - np.log(p0_supported)[None, :]
    parallel = log_delta @ wt_basis @ wt_basis.T
    orthogonal = log_delta - parallel
    total_norm = np.linalg.norm(log_delta, axis=1)
    parallel_norm = np.linalg.norm(parallel, axis=1)
    orthogonal_norm = np.linalg.norm(orthogonal, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        residual_fraction = np.where(total_norm > 0, orthogonal_norm / total_norm, 0.0)
        explained_fraction = np.where(total_norm > 0, parallel_norm**2 / total_norm**2, 1.0)

    elapsed = time.time() - start
    summary = {
        "config": asdict(config),
        "sampling": "independent linear uniform parameters theta_i ~ Uniform(0, 2*theta_i_WT)",
        "parameter_names": parameter_names,
        "p0": p0.tolist(),
        "supported_parameter_names": mapped_names,
        "unsupported_parameter_names": unsupported_names,
        "valid_hessians": int(n_valid),
        "failed_hessians": failed,
        "elapsed_seconds": elapsed,
        "pairwise_chordal": summarize_vector("pairwise_chordal", pairwise_chordal, skip_zero=True),
        "pairwise_max_angle_degrees": summarize_vector(
            "pairwise_max_angle_degrees", np.degrees(pairwise_max_angle), skip_zero=True
        ),
        "wt_chordal": summarize_vector("wt_chordal", wt_chordal),
        "wt_max_angle_degrees": summarize_vector("wt_max_angle_degrees", np.degrees(wt_max_angle)),
        "residual_fraction": summarize_vector("orthogonal_norm / total_norm", residual_fraction),
        "explained_fraction": summarize_vector("parallel_norm^2 / total_norm^2", explained_fraction),
        "total_log_distance_to_wt": summarize_vector("||log(theta)-log(theta_wt)||", total_norm),
    }

    return {
        "summary": summary,
        "points": points,
        "p0": p0,
        "parameter_names": np.asarray(parameter_names, dtype=object),
        "supported_points": supported_points,
        "supported_parameter_names": np.asarray(mapped_names, dtype=object),
        "unsupported_parameter_names": np.asarray(unsupported_names, dtype=object),
        "eigvals": eigvals_arr,
        "bases": bases_arr,
        "valid": valid,
        "pairwise_chordal": pairwise_chordal,
        "pairwise_max_angle": pairwise_max_angle,
        "wt_chordal": wt_chordal,
        "wt_max_angle": wt_max_angle,
        "log_delta": log_delta,
        "residual_fraction": residual_fraction,
        "explained_fraction": explained_fraction,
        "total_norm": total_norm,
    }


def save_result(result: dict, tag: str) -> tuple[Path, Path, Path]:
    npz_path = RESULTS / f"{tag}.npz"
    json_path = RESULTS / f"{tag}.json"
    fig_path = RESULTS / f"{tag}.png"
    arrays = {k: v for k, v in result.items() if k != "summary"}
    np.savez_compressed(npz_path, **arrays)
    json_path.write_text(json.dumps(result["summary"], indent=2))
    plot_result(result, fig_path)
    return npz_path, json_path, fig_path


def plot_result(result: dict, fig_path: Path) -> None:
    pairwise = np.asarray(result["pairwise_chordal"], dtype=float).ravel()
    pairwise = pairwise[np.isfinite(pairwise) & (pairwise > 0)]
    wt = np.asarray(result["wt_chordal"], dtype=float)
    wt = wt[np.isfinite(wt)]
    residual = np.asarray(result["residual_fraction"], dtype=float)
    residual = residual[np.isfinite(residual)]
    explained = np.asarray(result["explained_fraction"], dtype=float)
    explained = explained[np.isfinite(explained)]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    axes[0, 0].hist(pairwise, bins=30, color="#4c78a8", alpha=0.85)
    axes[0, 0].set_title("Pairwise sloppy-space distances")
    axes[0, 0].set_xlabel("chordal distance")
    axes[0, 0].set_ylabel("count")

    axes[0, 1].hist(wt, bins=30, color="#f28e2b", alpha=0.85)
    axes[0, 1].set_title("Distance to WT sloppy space")
    axes[0, 1].set_xlabel("chordal distance")
    axes[0, 1].set_ylabel("count")

    axes[1, 0].hist(residual, bins=30, color="#e15759", alpha=0.85)
    axes[1, 0].set_title("Random displacement outside WT sloppy space")
    axes[1, 0].set_xlabel("orthogonal residual fraction")
    axes[1, 0].set_ylabel("count")

    axes[1, 1].hist(explained, bins=30, color="#59a14f", alpha=0.85)
    axes[1, 1].set_title("Random displacement explained by WT sloppy space")
    axes[1, 1].set_xlabel("explained fraction")
    axes[1, 1].set_ylabel("count")

    fig.savefig(fig_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tyson random-cube control for sloppy-subspace analyses")
    parser.add_argument("--model", default="tyson1991")
    parser.add_argument("--n-points", type=int, default=50)
    parser.add_argument("--k-sloppy", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t-end", type=float, default=100.0)
    parser.add_argument("--n-time", type=int, default=501)
    parser.add_argument("--tag", default=None)
    config = RandomCubeConfig(**vars(parser.parse_args()))
    result = analyze_random_cube(config)
    tag = config.tag or f"{config.model}_random_cube_n{config.n_points}_k{config.k_sloppy}"
    out_npz, out_json, out_png = save_result(result, tag)
    print(f"Saved {out_npz}")
    print(f"Saved {out_json}")
    print(f"Saved {out_png}")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()

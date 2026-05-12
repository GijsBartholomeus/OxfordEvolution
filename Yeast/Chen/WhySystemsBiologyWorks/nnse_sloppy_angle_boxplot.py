from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from nnse_sloppy_subspace import (
    ROOT,
    compute_tyson_hessian,
    load_neutral_points,
    principal_angles,
    sloppy_basis,
)
from wsbw_nnse import get_spec, setup_rr
from wsbw_pipeline import prepare_models


RAW_ROOT = ROOT / "results" / "bruteforce_cloud"
RESULTS = ROOT / "results" / "sloppy_angle_boxplot"
FIGURES = ROOT / "figures" / "sloppy_geometry"
SUMMARIES = ROOT / "results_summaries" / "sloppy_geometry"


@dataclass
class AngleBoxplotConfig:
    model: str = "tyson1991"
    neutral_npz: str | None = None
    bruteforce_tag: str | None = "tyson_bfc_1e9"
    neutral_threshold: float = 0.05
    n_neutral: int = 2000
    n_random: int = 2000
    pair_samples: int = 200_000
    k_sloppy: int = 3
    seed: int = 42
    t_end: float = 100.0
    n_time: int = 501
    tag: str | None = None


def summarize(name: str, values: np.ndarray) -> dict:
    x = np.asarray(values, dtype=float)
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
        "mean": float(np.mean(x)),
    }


def load_p0_and_names(model: str) -> tuple[np.ndarray, list[str]]:
    audit = prepare_models()
    spec = get_spec(model)
    params = audit[spec.key]["free_parameters"]
    _, defaults, _ = setup_rr(spec, audit[spec.key]["promoted_sbml"], params)
    p0 = np.asarray([defaults[pid] for pid in params], dtype=float)
    return p0, params


def reservoir_update(
    reservoir: np.ndarray | None,
    selected: np.ndarray,
    max_points: int,
    seen_before: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    seen = seen_before
    if reservoir is None:
        reservoir = np.empty((0, selected.shape[1]), dtype=np.float32)
    for row in selected:
        seen += 1
        if len(reservoir) < max_points:
            reservoir = np.vstack([reservoir, row[None, :]]).astype(np.float32)
        else:
            j = int(rng.integers(0, seen))
            if j < max_points:
                reservoir[j] = row
    return reservoir, seen


def load_bruteforce_neutral_points(
    model: str,
    tag: str,
    cutoff: float,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    rng = np.random.default_rng(seed)
    paths = sorted((RAW_ROOT / tag).glob(f"{model}_bruteforce_cloud_N=*chunk-*.npz"))
    if not paths:
        raise FileNotFoundError(f"No raw brute-force chunks found under {RAW_ROOT / tag}")

    reservoir = None
    p0 = None
    parameter_names = None
    seen = 0
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
                attempted += int(np.asarray(data["samples_attempted"]).ravel()[0])
            successes += len(values)
        mask = np.isfinite(values) & (values <= cutoff) & np.all(np.isfinite(points), axis=1) & np.all(points > 0, axis=1)
        selected = points[mask]
        if len(selected):
            reservoir, seen = reservoir_update(reservoir, selected, max_points, seen, rng)

    if reservoir is None:
        reservoir = np.empty((0, len(p0)), dtype=np.float32)
    meta = {
        "source": "bruteforce_raw_chunks",
        "bruteforce_tag": tag,
        "raw_chunks": len(paths),
        "samples_attempted": int(attempted),
        "successes_saved_in_chunks": int(successes),
        "neutral_points_seen": int(seen),
        "neutral_points_sampled": int(len(reservoir)),
    }
    return np.asarray(reservoir, dtype=float), np.asarray(p0, dtype=float), parameter_names, meta


def load_neutral_sample(config: AngleBoxplotConfig) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    if config.bruteforce_tag:
        return load_bruteforce_neutral_points(
            config.model,
            config.bruteforce_tag,
            config.neutral_threshold,
            config.n_neutral,
            config.seed,
        )
    if not config.neutral_npz:
        raise ValueError("Either --bruteforce-tag or --neutral-npz is required")
    loaded = load_neutral_points(Path(config.neutral_npz), config.neutral_threshold)
    rng = np.random.default_rng(config.seed)
    points = loaded["points"]
    finite = np.all(np.isfinite(points), axis=1) & np.all(points > 0, axis=1)
    idx = np.where(finite)[0]
    if len(idx) > config.n_neutral:
        idx = rng.choice(idx, size=config.n_neutral, replace=False)
    meta = {
        "source": loaded["source"],
        "neutral_npz": str(config.neutral_npz),
        "neutral_points_seen": int(np.sum(finite)),
        "neutral_points_sampled": int(len(idx)),
    }
    return points[idx], loaded["p0"], loaded["parameter_names"], meta


def compute_bases(points: np.ndarray, parameter_names: list[str], config: AngleBoxplotConfig) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    bases = []
    eigvals = []
    failed = []
    for i, vector in enumerate(points):
        try:
            res = compute_tyson_hessian(vector, parameter_names, t_end=config.t_end, n_time=config.n_time)
            bases.append(sloppy_basis(res["eigvecs"], config.k_sloppy))
            eigvals.append(res["eigvals"])
        except Exception as exc:
            failed.append({"index": int(i), "error": str(exc)})
            bases.append(np.full((len(parameter_names), min(config.k_sloppy, len(parameter_names))), np.nan))
            eigvals.append(np.full(len(parameter_names), np.nan))
    return np.asarray(bases, dtype=float), np.asarray(eigvals, dtype=float), failed


def valid_bases(bases: np.ndarray) -> np.ndarray:
    return np.asarray([np.all(np.isfinite(x)) for x in bases], dtype=bool)


def sampled_pairwise_max_angles(bases: np.ndarray, pair_samples: int, seed: int) -> np.ndarray:
    valid = np.where(valid_bases(bases))[0]
    if len(valid) < 2:
        return np.empty(0, dtype=float)
    rng = np.random.default_rng(seed)
    n_pairs_total = len(valid) * (len(valid) - 1) // 2
    n_pairs = min(pair_samples, n_pairs_total)
    out = np.empty(n_pairs, dtype=float)
    written = 0
    while written < n_pairs:
        batch = min(10_000, n_pairs - written)
        a = rng.choice(valid, size=batch, replace=True)
        b = rng.choice(valid, size=batch, replace=True)
        keep = a != b
        a = a[keep]
        b = b[keep]
        for ai, bi in zip(a, b):
            out[written] = np.degrees(np.max(principal_angles(bases[ai], bases[bi])))
            written += 1
            if written >= n_pairs:
                break
    return out


def wt_max_angles(bases: np.ndarray, wt_basis: np.ndarray) -> np.ndarray:
    valid = np.where(valid_bases(bases))[0]
    out = np.empty(len(valid), dtype=float)
    for pos, i in enumerate(valid):
        out[pos] = np.degrees(np.max(principal_angles(bases[i], wt_basis)))
    return out


def plot_boxplot(result: dict, fig_path: Path) -> None:
    labels = [
        "neutral\npairwise",
        "neutral\nvs WT",
        "random cube\npairwise",
        "random cube\nvs WT",
    ]
    values = [
        np.asarray(result["arrays"]["neutral_pairwise_max_angle_degrees"], dtype=float),
        np.asarray(result["arrays"]["neutral_wt_max_angle_degrees"], dtype=float),
        np.asarray(result["arrays"]["random_pairwise_max_angle_degrees"], dtype=float),
        np.asarray(result["arrays"]["random_wt_max_angle_degrees"], dtype=float),
    ]
    fig, ax = plt.subplots(figsize=(8, 5.2), constrained_layout=True)
    ax.boxplot(values, labels=labels, showfliers=False, patch_artist=True)
    ax.set_ylabel("maximum principal angle between sloppy subspaces (degrees)")
    ax.set_title(
        f"Tyson sloppy-subspace angle comparison\n"
        f"neutral n={result['summary']['neutral_valid_hessians']}, random n={result['summary']['random_valid_hessians']}"
    )
    ax.grid(True, axis="y", alpha=0.25)
    for i, vals in enumerate(values, start=1):
        ax.scatter([i], [np.nanmedian(vals)], color="black", s=18, zorder=3)
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)


def analyze(config: AngleBoxplotConfig) -> dict:
    if config.model != "tyson1991":
        raise NotImplementedError("Only Tyson Hessian implementation is available")
    rng = np.random.default_rng(config.seed)
    start = time.time()

    neutral_points, p0, parameter_names, neutral_meta = load_neutral_sample(config)
    random_points = 2.0 * p0[None, :] * rng.uniform(0.0, 1.0, size=(config.n_random, len(p0)))

    wt_res = compute_tyson_hessian(p0, parameter_names, config.t_end, config.n_time)
    wt_basis = sloppy_basis(wt_res["eigvecs"], config.k_sloppy)
    neutral_bases, neutral_eigvals, neutral_failed = compute_bases(neutral_points, parameter_names, config)
    random_bases, random_eigvals, random_failed = compute_bases(random_points, parameter_names, config)

    neutral_pairwise = sampled_pairwise_max_angles(neutral_bases, config.pair_samples, config.seed + 11)
    random_pairwise = sampled_pairwise_max_angles(random_bases, config.pair_samples, config.seed + 22)
    neutral_wt = wt_max_angles(neutral_bases, wt_basis)
    random_wt = wt_max_angles(random_bases, wt_basis)

    arrays = {
        "neutral_points": neutral_points,
        "random_points": random_points,
        "p0": p0,
        "parameter_names": np.asarray(parameter_names, dtype=object),
        "neutral_bases": neutral_bases,
        "random_bases": random_bases,
        "neutral_eigvals": neutral_eigvals,
        "random_eigvals": random_eigvals,
        "neutral_pairwise_max_angle_degrees": neutral_pairwise,
        "neutral_wt_max_angle_degrees": neutral_wt,
        "random_pairwise_max_angle_degrees": random_pairwise,
        "random_wt_max_angle_degrees": random_wt,
    }
    summary = {
        "config": asdict(config),
        "neutral_source": neutral_meta,
        "elapsed_seconds": float(time.time() - start),
        "neutral_valid_hessians": int(np.sum(valid_bases(neutral_bases))),
        "random_valid_hessians": int(np.sum(valid_bases(random_bases))),
        "neutral_failed_hessians": neutral_failed,
        "random_failed_hessians": random_failed,
        "neutral_pairwise_max_angle_degrees": summarize("neutral pairwise", neutral_pairwise),
        "neutral_wt_max_angle_degrees": summarize("neutral vs WT", neutral_wt),
        "random_pairwise_max_angle_degrees": summarize("random pairwise", random_pairwise),
        "random_wt_max_angle_degrees": summarize("random vs WT", random_wt),
    }
    return {"summary": summary, "arrays": arrays}


def save(result: dict, tag: str) -> tuple[Path, Path, Path, Path]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    SUMMARIES.mkdir(parents=True, exist_ok=True)

    npz_path = RESULTS / f"{tag}.npz"
    json_path = RESULTS / f"{tag}.json"
    fig_path = FIGURES / f"{tag}.png"
    summary_path = SUMMARIES / f"{tag}.json"
    np.savez_compressed(npz_path, **result["arrays"])
    text = json.dumps(result["summary"], indent=2)
    json_path.write_text(text)
    summary_path.write_text(text)
    plot_boxplot(result, fig_path)
    return npz_path, json_path, fig_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Tyson sloppy-subspace angle boxplot")
    parser.add_argument("--model", default="tyson1991")
    parser.add_argument("--neutral-npz", default=None)
    parser.add_argument("--bruteforce-tag", default="tyson_bfc_1e9")
    parser.add_argument("--neutral-threshold", type=float, default=0.05)
    parser.add_argument("--n-neutral", type=int, default=2000)
    parser.add_argument("--n-random", type=int, default=2000)
    parser.add_argument("--pair-samples", type=int, default=200_000)
    parser.add_argument("--k-sloppy", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t-end", type=float, default=100.0)
    parser.add_argument("--n-time", type=int, default=501)
    parser.add_argument("--tag", default=None)
    config = AngleBoxplotConfig(**vars(parser.parse_args()))
    result = analyze(config)
    source = config.bruteforce_tag or "npz"
    tag = config.tag or f"{config.model}_sloppy_angle_boxplot_{source}_f{config.neutral_threshold:g}_n{config.n_neutral}_rand{config.n_random}_k{config.k_sloppy}"
    paths = save(result, tag)
    for path in paths:
        print(f"Saved {path}")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()

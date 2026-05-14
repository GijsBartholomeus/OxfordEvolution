#!/usr/bin/env python3
"""Plot analytical radius-accessibility curves for increasing sample sizes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parent
STATS_ROOT = ROOT / "results/bruteforce_cloud_stats"
FIGURE_ROOT = ROOT / "figures/bruteforce_cloud/model_comparisons"
SUMMARY_ROOT = ROOT / "results_summaries/bruteforce_cloud/model_comparisons"

CHEN_CONFIG = {
    "model": "chen2004",
    "label": "Chen 2004",
    "tag": "chen_bfc_1e8",
    "sample": STATS_ROOT / "chen_bfc_1e8/chen2004_bruteforce_samples_chen_bfc_1e8.npz",
}
RAW_ROOT = ROOT / "results/bruteforce_cloud"


def normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x, dtype=float) / math.sqrt(2.0)))


def cube_distance_cdf_normal(radii: np.ndarray, dim: int) -> np.ndarray:
    mean_sq = dim / 6.0
    sd_sq = math.sqrt(7.0 * dim / 180.0)
    return np.clip(normal_cdf((radii * radii - mean_sq) / sd_sq), 0.0, 1.0)


def cube_distance_cdf_mc(radii: np.ndarray, dim: int, samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + dim)
    batch = 100_000
    dists = np.empty(samples, dtype=np.float32)
    write_at = 0
    while write_at < samples:
        n = min(batch, samples - write_at)
        x = rng.random((n, dim), dtype=np.float32)
        y = rng.random((n, dim), dtype=np.float32)
        diff = x - y
        dists[write_at : write_at + n] = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        write_at += n
    dists.sort()
    return np.searchsorted(dists, radii, side="right") / float(samples)


def expected_unique_iid(p: np.ndarray, n: int, q: np.ndarray) -> np.ndarray:
    """Expected unique labels with center included and iid global label frequencies."""
    p = np.asarray(p, dtype=float)
    out = np.empty_like(q, dtype=float)
    for idx, qr in enumerate(q):
        if qr <= 0:
            out[idx] = 1.0
            continue
        log_none_other = (n - 1) * np.log1p(-np.clip(p * qr, 0.0, 1.0 - 1e-15))
        absent = (1.0 - p) * np.exp(log_none_other)
        out[idx] = float(np.sum(1.0 - absent))
    return out


def expected_unique_hypergeom(counts: np.ndarray, draws: int) -> float:
    counts = np.asarray(counts, dtype=np.int64)
    total = int(np.sum(counts))
    draws = int(np.clip(draws, 0, total))
    if draws <= 0:
        return 0.0
    if draws >= total:
        return float(len(counts))
    ok = (total - counts) >= draws
    absent = np.zeros(len(counts), dtype=float)
    absent[ok] = np.exp(
        gammaln(total - counts[ok] + 1)
        - gammaln(total - counts[ok] - draws + 1)
        + gammaln(total - draws + 1)
        - gammaln(total + 1)
    )
    return float(np.sum(1.0 - absent))


def expected_unique_finite_current_sample(counts: np.ndarray, n: int, q: np.ndarray) -> np.ndarray:
    """Finite-sample expectation for the current sample size only."""
    draws = np.rint(1.0 + (n - 1.0) * q).astype(int)
    return np.asarray([expected_unique_hypergeom(counts, m) for m in draws], dtype=float)


def chunk_files(tag: str, model: str) -> list[Path]:
    return sorted((RAW_ROOT / tag).glob(f"{model}_bruteforce_cloud_N=*chunk-*.npz"))


def chunk_count(path: Path) -> int:
    with np.load(path, allow_pickle=True) as data:
        return int(data["phenotype_codes"].shape[0])


def load_label_distribution_from_sample(path: Path) -> tuple[np.ndarray, np.ndarray, int, dict]:
    with np.load(path, allow_pickle=True) as data:
        codes = np.asarray(data["all_phenotype_codes"])
        p0 = np.asarray(data["p0"], dtype=float)
    _, counts = np.unique(codes, return_counts=True)
    p = counts.astype(float) / float(np.sum(counts))
    meta = {"source": "summary_sample_npz", "path": str(path), "reference_sample_size": int(len(codes))}
    return counts, p, int(len(p0)), meta


def load_label_distribution_from_raw(tag: str, model: str, n_sample: int, seed: int) -> tuple[np.ndarray, np.ndarray, int, dict]:
    files = chunk_files(tag, model)
    if not files:
        raise FileNotFoundError(f"No raw chunk files for {model} in {RAW_ROOT / tag}")
    rng = np.random.default_rng(seed)
    counts_by_chunk = np.asarray([chunk_count(path) for path in files], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts_by_chunk)])
    total = int(offsets[-1])
    n = min(int(n_sample), total)
    selected = np.sort(rng.choice(total, size=n, replace=False).astype(np.int64))
    codes = np.empty(n, dtype=np.uint64)
    write_at = 0
    p0 = None
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
            if p0 is None:
                p0 = np.asarray(data["p0"], dtype=float)
            chunk_codes = np.asarray(data["phenotype_codes"], dtype=np.uint64)
            m = len(local_idx)
            codes[write_at : write_at + m] = chunk_codes[local_idx]
            write_at += m
        nonempty += 1
        if nonempty % 50 == 0:
            print(f"  sampled labels from {nonempty} chunks; rows={write_at:,}/{n:,}", flush=True)
    codes = codes[:write_at]
    _, counts = np.unique(codes, return_counts=True)
    p = counts.astype(float) / float(np.sum(counts))
    meta = {
        "source": "raw_chunk_label_sample",
        "tag": tag,
        "model": model,
        "chunk_files": len(files),
        "raw_successful_points": total,
        "reference_sample_size": int(len(codes)),
    }
    if p0 is None:
        raise ValueError("No p0 loaded from raw chunks")
    return counts, p, int(len(p0)), meta


def load_label_distribution(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, int, dict]:
    if args.reference_source == "sample":
        return load_label_distribution_from_sample(CHEN_CONFIG["sample"])
    if args.reference_source == "raw":
        return load_label_distribution_from_raw(args.reference_tag, CHEN_CONFIG["model"], args.reference_sample_size, args.seed)
    if args.reference_sample_size > 0 and chunk_files(args.reference_tag, CHEN_CONFIG["model"]):
        return load_label_distribution_from_raw(args.reference_tag, CHEN_CONFIG["model"], args.reference_sample_size, args.seed)
    return load_label_distribution_from_sample(CHEN_CONFIG["sample"])


def parse_n_values(text: str) -> list[int]:
    values = []
    for part in text.split(","):
        part = part.strip().lower()
        if not part:
            continue
        values.append(int(float(part)))
    return values


def make_radii(dim: int, points: int) -> np.ndarray:
    # Dense enough around the transition while still showing small/large-radius limits.
    if dim >= 30:
        lo = 0.0
        hi = min(6.0, math.sqrt(dim))
        return np.linspace(lo, hi, points)
    return np.linspace(0.0, math.sqrt(dim), points)


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    n_values = parse_n_values(args.n_values)
    panel_dims = parse_n_values(args.panel_dims)
    output = {"n_values": n_values, "models": {}}
    ncols = 2
    nrows = int(math.ceil(len(panel_dims) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.8 * ncols, 5.2 * nrows), constrained_layout=True)
    axes = np.asarray(axes).ravel()

    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, len(n_values) - 1)) for i in range(len(n_values))]

    counts, p, reference_dim, reference_meta = load_label_distribution(args)
    for ax, dim in zip(axes, panel_dims):
        config = CHEN_CONFIG
        radii = make_radii(dim, args.radii)
        q = cube_distance_cdf_normal(radii, dim)
        q_method = "normal approximation"

        model_rows = {
            "label": config["label"],
            "tag": config["tag"],
            "dimensions": dim,
            "reference_dimensions": reference_dim,
            "phenotypes_in_reference_sample": int(len(counts)),
            "reference_sample_size": int(np.sum(counts)),
            "reference_distribution": reference_meta,
            "label_frequency_assumption": "fixed phenotype frequencies from the reference sample",
            "q_method": q_method,
            "radii": radii.tolist(),
            "curves": {},
        }

        for n, color in zip(n_values, colors):
            y = expected_unique_iid(p, n, q)
            model_rows["curves"][str(n)] = y.tolist()
            ax.plot(radii, y, lw=2.2, color=color, label=f"N={n:g}")

        if args.show_current_finite:
            n_current = int(np.sum(counts))
            y_current = expected_unique_finite_current_sample(counts, n_current, q)
            model_rows["finite_current_sample_curve"] = y_current.tolist()
            ax.plot(
                radii,
                y_current,
                color="black",
                lw=2.0,
                ls="--",
                label=f"current sample support\nN={n_current:,}",
            )

        ax.axvline(math.sqrt(dim / 6.0), color="0.55", lw=1.2, ls=":", label=r"$\sqrt{d/6}$")
        ax.set_yscale("log")
        ax.set_xlabel("radius in normalized cube")
        ax.set_ylabel("expected unique phenotypes in a new sample")
        ax.set_xlim(0.0, 6.0)
        if dim == reference_dim:
            title_dim = f"actual d={dim}"
        else:
            title_dim = f"hypothetical d={dim}"
        ax.set_title(f"{config['label']} analytical accessibility\nnew iid samples from fixed $p_\\phi$, {q_method}, {title_dim}")
        ax.grid(True, axis="y", alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
        output["models"][f"{config['model']}_d{dim}"] = model_rows
    for ax in axes[len(panel_dims):]:
        ax.axis("off")

    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURE_ROOT / "chen_analytic_accessibility_by_N_dimensions.png"
    json_path = SUMMARY_ROOT / "chen_analytic_accessibility_by_N_dimensions.json"
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)
    json_path.write_text(json.dumps(output, indent=2))
    return json_path, fig_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-values", default="1000000,10000000,100000000,1000000000,10000000000")
    parser.add_argument("--panel-dims", default="136,100,75,50")
    parser.add_argument("--reference-source", choices=["auto", "sample", "raw"], default="auto")
    parser.add_argument("--reference-tag", default="chen_bfc_1e9")
    parser.add_argument("--reference-sample-size", type=int, default=0)
    parser.add_argument("--radii", type=int, default=240)
    parser.add_argument("--mc-samples", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show-current-finite", action="store_true")
    args = parser.parse_args()
    json_path, fig_path = run(args)
    print(f"Saved {json_path}")
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()

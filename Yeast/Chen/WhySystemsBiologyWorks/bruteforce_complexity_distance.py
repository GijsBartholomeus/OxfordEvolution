from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from wsbw_pipeline import RESULTS


STATS_ROOT = RESULTS / "bruteforce_cloud_stats"
CHUNK_ROOT = RESULTS / "bruteforce_cloud"


def sample_npz_path(model: str, tag: str) -> Path:
    return STATS_ROOT / tag / f"{model}_bruteforce_samples_{tag}.npz"


def chunk_paths(model: str, tag: str) -> list[Path]:
    return sorted((CHUNK_ROOT / tag).glob(f"{model}_bruteforce_cloud_N=*chunk-*.npz"))


def finite_positive(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values) & (values > 0)]


def stat_dict(values: np.ndarray, exact_max: float | None = None) -> dict[str, float | int]:
    values = finite_positive(values)
    if len(values) == 0:
        return {"n": 0}
    return {
        "n": int(len(values)),
        "min": float(np.min(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.50)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values) if exact_max is None else exact_max),
    }


def grouped_from_sample(model: str, tag: str) -> tuple[dict[int, list[float]], dict[int, int], dict[int, float], dict]:
    path = sample_npz_path(model, tag)
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=False)
    complexities = np.rint(np.asarray(data["all_complexities"], dtype=float)).astype(int)
    objectives = np.asarray(data["all_objectives"], dtype=float)
    mask = np.isfinite(complexities) & np.isfinite(objectives) & (objectives > 0)
    grouped: dict[int, list[float]] = defaultdict(list)
    exact_counts: dict[int, int] = defaultdict(int)
    exact_max: dict[int, float] = defaultdict(lambda: -math.inf)
    for k, f in zip(complexities[mask], objectives[mask]):
        grouped[int(k)].append(float(f))
        exact_counts[int(k)] += 1
        exact_max[int(k)] = max(exact_max[int(k)], float(f))
    meta = {
        "source": "stats_sample",
        "sample_npz": str(path),
        "points_seen": int(np.sum(mask)),
    }
    return grouped, dict(exact_counts), dict(exact_max), meta


def reservoir_add(
    reservoir: dict[int, list[float]],
    counts: dict[int, int],
    maxima: dict[int, float],
    k: int,
    f: float,
    max_per_complexity: int,
    rng: np.random.Generator,
) -> None:
    counts[k] += 1
    maxima[k] = max(maxima.get(k, -math.inf), f)
    bucket = reservoir[k]
    if len(bucket) < max_per_complexity:
        bucket.append(f)
        return
    j = int(rng.integers(0, counts[k]))
    if j < max_per_complexity:
        bucket[j] = f


def grouped_from_chunks(
    model: str,
    tag: str,
    max_per_complexity: int,
    seed: int,
) -> tuple[dict[int, list[float]], dict[int, int], dict[int, float], dict]:
    paths = chunk_paths(model, tag)
    if not paths:
        raise FileNotFoundError(f"No raw chunk files found in {CHUNK_ROOT / tag} for {model}")
    rng = np.random.default_rng(seed)
    reservoir: dict[int, list[float]] = defaultdict(list)
    counts: dict[int, int] = defaultdict(int)
    maxima: dict[int, float] = {}
    points_seen = 0
    for idx, path in enumerate(paths, start=1):
        data = np.load(path, allow_pickle=False)
        complexities = np.rint(np.asarray(data["complexities"], dtype=float)).astype(int)
        objectives = np.asarray(data["objectives"], dtype=float)
        mask = np.isfinite(complexities) & np.isfinite(objectives) & (objectives > 0)
        for k, f in zip(complexities[mask], objectives[mask]):
            reservoir_add(reservoir, counts, maxima, int(k), float(f), max_per_complexity, rng)
        points_seen += int(np.sum(mask))
        if idx == 1 or idx % 10 == 0 or idx == len(paths):
            print(f"scanned {idx}/{len(paths)} chunks; finite positive points={points_seen:,}", flush=True)
    meta = {
        "source": "raw_chunks",
        "chunk_dir": str(CHUNK_ROOT / tag),
        "chunks_scanned": len(paths),
        "points_seen": points_seen,
        "max_per_complexity_reservoir": max_per_complexity,
    }
    return dict(reservoir), dict(counts), maxima, meta


def make_equal_sample(
    grouped: dict[int, list[float]],
    selected_ks: list[int],
    equal_n: int,
    rng: np.random.Generator,
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for k in selected_ks:
        values = finite_positive(np.asarray(grouped[k], dtype=float))
        if len(values) < equal_n:
            continue
        idx = rng.choice(len(values), size=equal_n, replace=False)
        out[k] = values[idx]
    return out


def select_complexities(
    counts: dict[int, int],
    min_count: int,
    min_k: int | None,
    max_k: int | None,
) -> list[int]:
    selected = []
    for k, n in sorted(counts.items()):
        if n < min_count:
            continue
        if min_k is not None and k < min_k:
            continue
        if max_k is not None and k > max_k:
            continue
        selected.append(k)
    return selected


def rows_from_stats(stats: dict[int, dict]) -> dict[str, np.ndarray]:
    ks = np.asarray(sorted(stats), dtype=int)
    out: dict[str, np.ndarray] = {"k": ks}
    for key in ["n", "q05", "q25", "median", "q75", "q95", "max"]:
        out[key] = np.asarray([stats[int(k)].get(key, np.nan) for k in ks], dtype=float)
    return out


def choose_violin_ks(selected_ks: list[int], max_bins: int) -> list[int]:
    if len(selected_ks) <= max_bins:
        return selected_ks
    idx = np.unique(np.round(np.linspace(0, len(selected_ks) - 1, max_bins)).astype(int))
    return [selected_ks[int(i)] for i in idx]


def plot_quantile_band(ax, rows: dict[str, np.ndarray], title: str) -> None:
    k = rows["k"]
    if len(k) == 0:
        ax.text(0.5, 0.5, "No eligible complexities", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    ax.fill_between(k, rows["q05"], rows["q95"], color="#7aa6c2", alpha=0.22, label="5-95%")
    ax.fill_between(k, rows["q25"], rows["q75"], color="#2f6f95", alpha=0.28, label="IQR")
    ax.plot(k, rows["median"], color="black", lw=2.0, label="median")
    ax.scatter(k, rows["max"], s=16, color="#c44e52", alpha=0.85, label="max")
    ax.set_yscale("log")
    ax.set_xlabel("phenotype complexity K(x)")
    ax.set_ylabel("WT distance f")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)


def plot_results(
    out_dir: Path,
    model: str,
    tag: str,
    suffix: str,
    counts: dict[int, int],
    raw_rows: dict[str, np.ndarray],
    equal_rows: dict[str, np.ndarray],
    equal_values: dict[int, np.ndarray],
    selected_ks: list[int],
    max_violin_bins: int,
    meta_text: list[str],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    ax_counts, ax_raw, ax_equal, ax_violin = axes.ravel()

    all_ks = np.asarray(sorted(counts), dtype=int)
    all_counts = np.asarray([counts[int(k)] for k in all_ks], dtype=float)
    selected_set = set(selected_ks)
    colors = ["#c44e52" if int(k) in selected_set else "#9e9e9e" for k in all_ks]
    ax_counts.bar(all_ks, all_counts, color=colors, width=0.82)
    ax_counts.set_yscale("log")
    ax_counts.set_xlabel("phenotype complexity K(x)")
    ax_counts.set_ylabel("points in source")
    ax_counts.set_title("Sample count per complexity")
    ax_counts.text(
        0.02,
        0.98,
        "\n".join(meta_text),
        ha="left",
        va="top",
        transform=ax_counts.transAxes,
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    plot_quantile_band(ax_raw, raw_rows, "Raw/reservoir distance distribution")
    plot_quantile_band(ax_equal, equal_rows, "Equal-N per complexity")

    violin_ks = choose_violin_ks(selected_ks, max_violin_bins)
    violin_values = [np.log10(finite_positive(equal_values[k])) for k in violin_ks if k in equal_values]
    violin_positions = [k for k in violin_ks if k in equal_values]
    if violin_values:
        parts = ax_violin.violinplot(violin_values, positions=violin_positions, widths=0.8, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor("#4c78a8")
            body.set_alpha(0.35)
            body.set_edgecolor("#2f4b63")
        medians = [float(np.median(v)) for v in violin_values]
        q25 = [float(np.quantile(v, 0.25)) for v in violin_values]
        q75 = [float(np.quantile(v, 0.75)) for v in violin_values]
        ax_violin.scatter(violin_positions, medians, color="black", s=14, zorder=3, label="median")
        ax_violin.vlines(violin_positions, q25, q75, color="black", lw=1.5, zorder=3, label="IQR")
        ax_violin.set_xlabel("phenotype complexity K(x)")
        ax_violin.set_ylabel("log10 WT distance f")
        ax_violin.set_title("Equal-N distribution shapes")
        ax_violin.legend(frameon=False, fontsize=8)
    else:
        ax_violin.text(0.5, 0.5, "No equal-N violin data", ha="center", va="center", transform=ax_violin.transAxes)
        ax_violin.set_axis_off()

    out = out_dir / f"{model}_complexity_distance_equalN_{tag}_{suffix}.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def main(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    if args.source == "sample":
        grouped, counts, exact_max, meta = grouped_from_sample(args.model, args.tag)
    elif args.source == "chunks":
        grouped, counts, exact_max, meta = grouped_from_chunks(
            args.model,
            args.tag,
            args.max_per_complexity,
            args.seed,
        )
    else:
        try:
            grouped, counts, exact_max, meta = grouped_from_sample(args.model, args.tag)
        except FileNotFoundError:
            grouped, counts, exact_max, meta = grouped_from_chunks(
                args.model,
                args.tag,
                args.max_per_complexity,
                args.seed,
            )

    selected_ks = select_complexities(counts, args.min_count, args.min_k, args.max_k)
    if not selected_ks:
        raise RuntimeError("No complexity values passed the count/K filters")
    reservoir_min = min(len(finite_positive(np.asarray(grouped[k], dtype=float))) for k in selected_ks)
    equal_n = args.equal_n if args.equal_n is not None else min(args.default_equal_n, reservoir_min)
    equal_n = min(equal_n, reservoir_min)
    if equal_n < 2:
        raise RuntimeError(f"Equal-N would be only {equal_n}; lower --min-count or use more data")

    raw_stats = {
        int(k): stat_dict(np.asarray(grouped[int(k)], dtype=float), exact_max=exact_max.get(int(k)))
        for k in sorted(grouped)
        if counts.get(int(k), 0) > 0
    }
    equal_values = make_equal_sample(grouped, selected_ks, equal_n, rng)
    equal_stats = {int(k): stat_dict(values) for k, values in equal_values.items()}

    out_dir = STATS_ROOT / args.tag / "complexity_distance"
    raw_rows = rows_from_stats(raw_stats)
    equal_rows = rows_from_stats(equal_stats)
    meta_text = [
        f"source: {meta['source']}",
        f"points seen: {meta['points_seen']:,}",
        f"eligible K: {len(selected_ks)}",
        f"min count filter: {args.min_count:,}",
        f"equal N: {equal_n:,}",
    ]
    suffix = f"{meta['source']}_min{args.min_count}_eq{equal_n}"
    plot_path = plot_results(
        out_dir,
        args.model,
        args.tag,
        suffix,
        counts,
        raw_rows,
        equal_rows,
        equal_values,
        selected_ks,
        args.max_violin_bins,
        meta_text,
    )

    high_complexity_counts = [
        {"complexity": int(k), "count": int(counts[k])}
        for k in sorted(counts)[-args.report_highest :]
    ]
    summary = {
        "model": args.model,
        "tag": args.tag,
        "source": meta,
        "min_count": args.min_count,
        "min_k": args.min_k,
        "max_k": args.max_k,
        "selected_complexities": [int(k) for k in selected_ks],
        "equal_n_per_complexity": int(equal_n),
        "high_complexity_counts": high_complexity_counts,
        "raw_stats_by_complexity": {str(k): v for k, v in raw_stats.items()},
        "equal_n_stats_by_complexity": {str(k): v for k, v in equal_stats.items()},
        "plot": str(plot_path),
    }
    out_json = out_dir / f"{args.model}_complexity_distance_equalN_{args.tag}_{suffix}.json"
    out_json.write_text(json.dumps(summary, indent=2))

    print("High-complexity counts:")
    for row in high_complexity_counts:
        print(f"  K={row['complexity']:>3}: n={row['count']:,}")
    print(f"Selected {len(selected_ks)} complexities with n >= {args.min_count:,}")
    print(f"Equal-N per complexity: {equal_n:,}")
    print(f"Saved {plot_path}")
    print(f"Saved {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Control complexity-vs-WT-distance plots for per-complexity sample-size effects."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--source", choices=["auto", "sample", "chunks"], default="auto")
    parser.add_argument("--min-count", type=int, default=200)
    parser.add_argument("--equal-n", type=int, default=None)
    parser.add_argument("--default-equal-n", type=int, default=1000)
    parser.add_argument("--min-k", type=int, default=None)
    parser.add_argument("--max-k", type=int, default=None)
    parser.add_argument("--max-per-complexity", type=int, default=5000)
    parser.add_argument("--max-violin-bins", type=int, default=18)
    parser.add_argument("--report-highest", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())

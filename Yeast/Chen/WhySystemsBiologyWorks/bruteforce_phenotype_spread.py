from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from wsbw_pipeline import RESULTS


STATS_ROOT = RESULTS / "bruteforce_cloud_stats"
CHUNK_ROOT = RESULTS / "bruteforce_cloud"


@dataclass(frozen=True)
class PhenotypeChoice:
    code: int
    complexity: int
    complexity_float: float
    count: int
    rank_within_complexity: int
    encoding: str | None = None


def uint64_to_bits(code: int, length: int = 49) -> str:
    return format(int(code), f"0{length}b")


def bits_to_uint64(bits: str) -> int:
    return int(bits, 2) if bits else 0


def chunk_paths(model: str, tag: str) -> list[Path]:
    return sorted((CHUNK_ROOT / tag).glob(f"{model}_bruteforce_cloud_N=*chunk-*.npz"))


def frequency_json_path(model: str, tag: str) -> Path:
    return STATS_ROOT / tag / f"{model}_complexity_frequency_{tag}.json"


def choose_from_frequency_json(
    model: str,
    tag: str,
    top_per_complexity: int,
    min_phenotype_count: int,
    min_k: int | None,
    max_k: int | None,
) -> tuple[list[PhenotypeChoice], dict]:
    path = frequency_json_path(model, tag)
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text())
    by_k: dict[int, list[PhenotypeChoice]] = defaultdict(list)
    max_count_by_k: dict[int, int] = defaultdict(int)
    phenotype_total = 0
    for pheno in data["phenotypes"]:
        complexity_float = float(pheno["complexity"])
        k = int(round(complexity_float))
        count = int(pheno["count"])
        max_count_by_k[k] = max(max_count_by_k[k], count)
        phenotype_total += 1
        if count < min_phenotype_count:
            continue
        if min_k is not None and k < min_k:
            continue
        if max_k is not None and k > max_k:
            continue
        encoding = str(pheno["encoding"])
        by_k[k].append(
            PhenotypeChoice(
                code=bits_to_uint64(encoding),
                complexity=k,
                complexity_float=complexity_float,
                count=count,
                rank_within_complexity=0,
                encoding=encoding,
            )
        )
    choices: list[PhenotypeChoice] = []
    for k in sorted(by_k):
        ranked = sorted(by_k[k], key=lambda item: item.count, reverse=True)[:top_per_complexity]
        for rank, item in enumerate(ranked, start=1):
            choices.append(
                PhenotypeChoice(
                    code=item.code,
                    complexity=item.complexity,
                    complexity_float=item.complexity_float,
                    count=item.count,
                    rank_within_complexity=rank,
                    encoding=item.encoding,
                )
            )
    meta = {
        "selection_source": "frequency_json",
        "frequency_json": str(path),
        "phenotypes_in_frequency_json": phenotype_total,
        "max_count_by_complexity": {str(k): int(v) for k, v in sorted(max_count_by_k.items())},
    }
    return choices, meta


def choose_from_chunk_counts(
    model: str,
    tag: str,
    top_per_complexity: int,
    min_phenotype_count: int,
    min_k: int | None,
    max_k: int | None,
) -> tuple[list[PhenotypeChoice], dict]:
    paths = chunk_paths(model, tag)
    if not paths:
        raise FileNotFoundError(f"No raw chunk files for {model} in {CHUNK_ROOT / tag}")
    counts: Counter[int] = Counter()
    code_to_complexity: dict[int, float] = {}
    for idx, path in enumerate(paths, start=1):
        data = np.load(path, allow_pickle=False)
        codes = np.asarray(data["phenotype_codes"], dtype=np.uint64)
        complexities = np.asarray(data["complexities"], dtype=float)
        for code, complexity in zip(codes, complexities):
            code_int = int(code)
            counts[code_int] += 1
            if code_int not in code_to_complexity:
                code_to_complexity[code_int] = float(complexity)
        if idx == 1 or idx % 10 == 0 or idx == len(paths):
            print(f"counted {idx}/{len(paths)} chunks; unique phenotypes={len(counts):,}", flush=True)

    by_k: dict[int, list[PhenotypeChoice]] = defaultdict(list)
    max_count_by_k: dict[int, int] = defaultdict(int)
    for code, count in counts.items():
        complexity_float = code_to_complexity[code]
        k = int(round(complexity_float))
        max_count_by_k[k] = max(max_count_by_k[k], count)
        if count < min_phenotype_count:
            continue
        if min_k is not None and k < min_k:
            continue
        if max_k is not None and k > max_k:
            continue
        by_k[k].append(
            PhenotypeChoice(
                code=code,
                complexity=k,
                complexity_float=complexity_float,
                count=int(count),
                rank_within_complexity=0,
                encoding=uint64_to_bits(code),
            )
        )
    choices = []
    for k in sorted(by_k):
        ranked = sorted(by_k[k], key=lambda item: item.count, reverse=True)[:top_per_complexity]
        for rank, item in enumerate(ranked, start=1):
            choices.append(
                PhenotypeChoice(
                    code=item.code,
                    complexity=item.complexity,
                    complexity_float=item.complexity_float,
                    count=item.count,
                    rank_within_complexity=rank,
                    encoding=item.encoding,
                )
            )
    meta = {
        "selection_source": "chunk_count_pass",
        "chunks_counted": len(paths),
        "phenotypes_counted": len(counts),
        "max_count_by_complexity": {str(k): int(v) for k, v in sorted(max_count_by_k.items())},
    }
    return choices, meta


def reservoir_add(
    reservoirs: dict[int, list[np.ndarray]],
    seen: dict[int, int],
    code: int,
    point: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> None:
    seen[code] += 1
    bucket = reservoirs[code]
    if len(bucket) < max_points:
        bucket.append(np.asarray(point, dtype=np.float32).copy())
        return
    j = int(rng.integers(0, seen[code]))
    if j < max_points:
        bucket[j] = np.asarray(point, dtype=np.float32).copy()


def collect_selected_points(
    model: str,
    tag: str,
    choices: list[PhenotypeChoice],
    max_points_per_phenotype: int,
    seed: int,
) -> tuple[dict[int, np.ndarray], dict[int, int], np.ndarray, list[str], dict]:
    paths = chunk_paths(model, tag)
    if not paths:
        raise FileNotFoundError(f"No raw chunk files for {model} in {CHUNK_ROOT / tag}")
    selected_codes = {choice.code for choice in choices}
    reservoirs: dict[int, list[np.ndarray]] = {code: [] for code in selected_codes}
    seen: dict[int, int] = defaultdict(int)
    rng = np.random.default_rng(seed)
    p0 = None
    parameter_names: list[str] = []
    for idx, path in enumerate(paths, start=1):
        data = np.load(path, allow_pickle=True)
        if p0 is None:
            p0 = np.asarray(data["p0"], dtype=np.float32)
            if "parameter_names" in data.files:
                parameter_names = [str(x) for x in np.asarray(data["parameter_names"], dtype=object)]
        points = np.asarray(data["points"], dtype=np.float32)
        codes = np.asarray(data["phenotype_codes"], dtype=np.uint64)
        for row_idx, code_value in enumerate(codes):
            code = int(code_value)
            if code in selected_codes:
                reservoir_add(reservoirs, seen, code, points[row_idx], max_points_per_phenotype, rng)
        if idx == 1 or idx % 10 == 0 or idx == len(paths):
            matched = sum(seen.values())
            print(f"collected {idx}/{len(paths)} chunks; selected hits={matched:,}", flush=True)
    if p0 is None:
        raise RuntimeError("No chunks were scanned")
    arrays = {
        code: np.asarray(points_for_code, dtype=np.float32)
        for code, points_for_code in reservoirs.items()
    }
    meta = {
        "chunks_scanned_for_points": len(paths),
        "selected_hits_seen": int(sum(seen.values())),
        "max_points_per_phenotype": max_points_per_phenotype,
    }
    return arrays, dict(seen), p0, parameter_names, meta


def normalize_points(points: np.ndarray, p0: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=float) / np.maximum(2.0 * np.asarray(p0, dtype=float)[None, :], np.finfo(float).tiny)


def pairwise_stats(points: np.ndarray, rng: np.random.Generator, max_pairwise_points: int) -> tuple[dict, np.ndarray]:
    if len(points) < 2:
        return {"sampled_points": int(len(points)), "pair_count": 0}, np.empty(0, dtype=float)
    if len(points) > max_pairwise_points:
        idx = rng.choice(len(points), size=max_pairwise_points, replace=False)
        points = points[idx]
    distances = pdist(points, metric="euclidean")
    stats = {
        "sampled_points": int(len(points)),
        "pair_count": int(len(distances)),
        "mean": float(np.mean(distances)),
        "min": float(np.min(distances)),
        "q05": float(np.quantile(distances, 0.05)),
        "q25": float(np.quantile(distances, 0.25)),
        "median": float(np.quantile(distances, 0.50)),
        "q75": float(np.quantile(distances, 0.75)),
        "q95": float(np.quantile(distances, 0.95)),
        "max": float(np.max(distances)),
    }
    return stats, distances


def plot_spread(rows: list[dict], out_dir: Path, model: str, tag: str, args: argparse.Namespace) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    usable = [row for row in rows if row["pairwise"].get("pair_count", 0) > 0]
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    ax_count, ax_spread, ax_scatter, ax_density = axes.ravel()
    if not usable:
        for ax in axes.ravel():
            ax.text(0.5, 0.5, "No selected phenotypes with >=2 sampled points", ha="center", va="center")
            ax.set_axis_off()
    else:
        k = np.asarray([row["complexity"] for row in usable], dtype=float)
        rank = np.asarray([row["rank_within_complexity"] for row in usable], dtype=float)
        count = np.asarray([row["count"] for row in usable], dtype=float)
        sampled = np.asarray([row["pairwise"]["sampled_points"] for row in usable], dtype=float)
        med = np.asarray([row["pairwise"]["median"] for row in usable], dtype=float)
        q05 = np.asarray([row["pairwise"]["q05"] for row in usable], dtype=float)
        q25 = np.asarray([row["pairwise"]["q25"] for row in usable], dtype=float)
        q75 = np.asarray([row["pairwise"]["q75"] for row in usable], dtype=float)
        q95 = np.asarray([row["pairwise"]["q95"] for row in usable], dtype=float)
        maxv = np.asarray([row["pairwise"]["max"] for row in usable], dtype=float)
        x = k + (rank - (args.top_per_complexity + 1) / 2.0) * 0.18
        size = 18 + 12 * np.log10(np.maximum(count, 1))

        ax_count.scatter(x, count, c=rank, cmap="viridis_r", s=size, alpha=0.8, edgecolor="black", linewidth=0.3)
        ax_count.set_yscale("log")
        ax_count.set_xlabel("phenotype complexity K(x)")
        ax_count.set_ylabel("phenotype occurrence count")
        ax_count.set_title("Selected high-occurrence phenotypes")

        ax_spread.vlines(x, q05, q95, color="#9ecae1", lw=2.0, alpha=0.9, label="5-95%")
        ax_spread.vlines(x, q25, q75, color="#3182bd", lw=4.0, alpha=0.8, label="IQR")
        ax_spread.scatter(x, med, c="black", s=18, zorder=3, label="median")
        ax_spread.scatter(x, maxv, c="#c44e52", s=20, alpha=0.8, zorder=2, label="max")
        ax_spread.set_xlabel("phenotype complexity K(x)")
        ax_spread.set_ylabel("within-phenotype pairwise distance")
        ax_spread.set_title("Spatial spread of each phenotype preimage")
        ax_spread.legend(frameon=False, fontsize=8)

        scatter = ax_scatter.scatter(count, med, c=k, cmap="plasma", s=size, alpha=0.8, edgecolor="black", linewidth=0.3)
        ax_scatter.set_xscale("log")
        ax_scatter.set_xlabel("phenotype occurrence count")
        ax_scatter.set_ylabel("median pairwise distance")
        ax_scatter.set_title("Frequency versus spread")
        cbar = fig.colorbar(scatter, ax=ax_scatter)
        cbar.set_label("K(x)")

        density_rows = []
        density_labels = []
        for kk in sorted(set(k.astype(int))):
            vals = [row["pairwise_distances"] for row in usable if row["complexity"] == kk]
            vals = [v for v in vals if len(v)]
            if not vals:
                continue
            merged = np.concatenate(vals)
            if len(merged) > args.max_density_pairs_per_complexity:
                rng = np.random.default_rng(args.seed + kk)
                merged = merged[rng.choice(len(merged), size=args.max_density_pairs_per_complexity, replace=False)]
            density_rows.append(merged)
            density_labels.append(kk)
        if density_rows:
            positions = np.asarray(density_labels, dtype=float)
            parts = ax_density.violinplot(density_rows, positions=positions, widths=0.75, showextrema=False)
            for body in parts["bodies"]:
                body.set_facecolor("#4c78a8")
                body.set_alpha(0.35)
                body.set_edgecolor("#2f4b63")
            medians = [float(np.median(v)) for v in density_rows]
            ax_density.scatter(positions, medians, color="black", s=14, zorder=3)
        ax_density.set_xlabel("phenotype complexity K(x)")
        ax_density.set_ylabel("within-phenotype pairwise distance")
        ax_density.set_title("Pairwise distance distributions pooled by K")

        info = [
            f"model: {model}",
            f"tag: {tag}",
            f"selected phenotypes: {len(usable)}",
            f"top per K: {args.top_per_complexity}",
            f"min phenotype count: {args.min_phenotype_count:,}",
            f"max points/phenotype: {args.max_points_per_phenotype:,}",
        ]
        ax_count.text(
            0.02,
            0.98,
            "\n".join(info),
            ha="left",
            va="top",
            fontsize=8,
            transform=ax_count.transAxes,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    suffix = f"top{args.top_per_complexity}_min{args.min_phenotype_count}_pts{args.max_points_per_phenotype}"
    out = out_dir / f"{model}_phenotype_spread_{tag}_{suffix}.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def main(args: argparse.Namespace) -> None:
    if args.selection_source == "frequency-json":
        choices, selection_meta = choose_from_frequency_json(
            args.model,
            args.tag,
            args.top_per_complexity,
            args.min_phenotype_count,
            args.min_k,
            args.max_k,
        )
    elif args.selection_source == "chunks":
        choices, selection_meta = choose_from_chunk_counts(
            args.model,
            args.tag,
            args.top_per_complexity,
            args.min_phenotype_count,
            args.min_k,
            args.max_k,
        )
    else:
        try:
            choices, selection_meta = choose_from_frequency_json(
                args.model,
                args.tag,
                args.top_per_complexity,
                args.min_phenotype_count,
                args.min_k,
                args.max_k,
            )
        except FileNotFoundError:
            choices, selection_meta = choose_from_chunk_counts(
                args.model,
                args.tag,
                args.top_per_complexity,
                args.min_phenotype_count,
                args.min_k,
                args.max_k,
            )
    if not choices:
        raise RuntimeError("No phenotypes passed the selection filters")
    print(f"Selected {len(choices)} phenotypes across {len(set(c.complexity for c in choices))} complexity values")
    for choice in sorted(choices, key=lambda c: (c.complexity, c.rank_within_complexity))[-args.report_highest :]:
        print(
            f"  K={choice.complexity:>3} rank={choice.rank_within_complexity} "
            f"count={choice.count:,} code={choice.code}"
        )

    points_by_code, seen_by_code, p0, parameter_names, collection_meta = collect_selected_points(
        args.model,
        args.tag,
        choices,
        args.max_points_per_phenotype,
        args.seed,
    )
    rng = np.random.default_rng(args.seed)
    rows = []
    for choice in choices:
        points = points_by_code.get(choice.code, np.empty((0, len(p0)), dtype=np.float32))
        norm_points = normalize_points(points, p0)
        stats, distances = pairwise_stats(norm_points, rng, args.max_pairwise_points)
        rows.append(
            {
                "code": str(choice.code),
                "encoding": choice.encoding or uint64_to_bits(choice.code),
                "complexity": int(choice.complexity),
                "complexity_float": float(choice.complexity_float),
                "count": int(choice.count),
                "collected_seen": int(seen_by_code.get(choice.code, 0)),
                "rank_within_complexity": int(choice.rank_within_complexity),
                "pairwise": stats,
                "pairwise_distances": distances,
            }
        )

    out_dir = STATS_ROOT / args.tag / "phenotype_spread"
    plot_path = plot_spread(rows, out_dir, args.model, args.tag, args)
    summary_rows = []
    for row in rows:
        copy = {k: v for k, v in row.items() if k != "pairwise_distances"}
        summary_rows.append(copy)
    summary = {
        "model": args.model,
        "tag": args.tag,
        "selection": selection_meta,
        "collection": collection_meta,
        "parameter_names": parameter_names,
        "settings": vars(args),
        "phenotypes": summary_rows,
        "plot": str(plot_path),
    }
    suffix = f"top{args.top_per_complexity}_min{args.min_phenotype_count}_pts{args.max_points_per_phenotype}"
    out_json = out_dir / f"{args.model}_phenotype_spread_{args.tag}_{suffix}.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"Saved {plot_path}")
    print(f"Saved {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimate within-phenotype spatial spread for high-occurrence phenotypes at each complexity."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--selection-source", choices=["auto", "frequency-json", "chunks"], default="auto")
    parser.add_argument("--top-per-complexity", type=int, default=3)
    parser.add_argument("--min-phenotype-count", type=int, default=100)
    parser.add_argument("--min-k", type=int, default=None)
    parser.add_argument("--max-k", type=int, default=None)
    parser.add_argument("--max-points-per-phenotype", type=int, default=1000)
    parser.add_argument("--max-pairwise-points", type=int, default=1000)
    parser.add_argument("--max-density-pairs-per-complexity", type=int, default=20000)
    parser.add_argument("--report-highest", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())

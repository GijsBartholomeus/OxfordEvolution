from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsbw_nnse import NNSEConfig, get_spec, make_thresholds, setup_rr
from wsbw_pipeline import RESULTS, SPECS, prepare_models
from wsbw_nnse_batch_init import build_initial_population, merge_candidate_lists


OUT_ROOT = RESULTS / "nnse_batch_init"


def sample_label(samples: int) -> str:
    if samples <= 0:
        return f"N={samples}"
    exponent = round(math.log10(samples))
    if 10**exponent == samples:
        return f"N=1e{exponent}"
    return f"N={samples}"


def main(args: argparse.Namespace) -> Path:
    in_dir = OUT_ROOT / args.tag
    if not in_dir.exists():
        raise FileNotFoundError(f"No NNSE batch directory found: {in_dir}")

    pattern = f"{args.model}_nnse_batch_init_N=*chunk-*.npz"
    chunk_paths = sorted(in_dir.glob(pattern))
    if not chunk_paths:
        raise FileNotFoundError(f"No chunk npz files matching {pattern} in {in_dir}")

    audit = prepare_models()
    spec = get_spec(args.model)
    params = audit[spec.key]["free_parameters"]
    _, defaults, _ = setup_rr(spec, audit[spec.key]["promoted_sbml"], params)
    p0 = np.asarray([defaults[pid] for pid in params], dtype=float)
    config = NNSEConfig(
        model=spec.key,
        n_bins=args.n_bins,
        bin_min=args.bin_min,
        bin_max=args.bin_max,
        bin_top=args.bin_top,
        spacing=args.spacing,
        seed=args.seed,
    )
    thresholds = make_thresholds(config)

    start = time.time()
    bin_counts = np.zeros(len(thresholds), dtype=np.int64)
    candidate_counts_seen = np.zeros(len(thresholds), dtype=np.int64)
    candidates: dict[int, list[tuple[float, list[float]]]] = {}
    total_candidates = 0
    finite_count = 0
    failed_count = 0
    overflow_count = 0
    best_value = float("inf")
    best_vector = np.full(len(params), np.nan, dtype=float)
    summaries = []

    for path in chunk_paths:
        data = np.load(path, allow_pickle=True)
        json_path = path.with_suffix(".json")
        summary = json.loads(json_path.read_text()) if json_path.exists() else {}
        summaries.append(summary)
        total_candidates += int(summary.get("candidates", 0))
        finite_count += int(summary.get("finite_count", 0))
        failed_count += int(summary.get("failed_count", 0))
        overflow_count += int(summary.get("overflow_count", 0))
        if "bin_counts" in data:
            bin_counts += np.asarray(data["bin_counts"], dtype=np.int64)

        values = np.asarray(data["candidate_objective_values"], dtype=float)
        vectors = np.asarray(data["candidate_vectors"], dtype=float)
        bins = np.asarray(data["candidate_bin_indices"], dtype=int)
        incoming: dict[str, list[tuple[float, list[float]]]] = {}
        for bin_idx, value, vector in zip(bins, values, vectors):
            candidate_counts_seen[bin_idx] += 1
            incoming.setdefault(str(int(bin_idx)), []).append((float(value), vector.astype(float).tolist()))
            if float(value) < best_value:
                best_value = float(value)
                best_vector = vector.astype(float)
        merge_candidate_lists(candidates, incoming, args.keep_per_bin)

    initial_population, initial_values, source_bins = build_initial_population(candidates, len(thresholds), len(params))
    filled_bins = int(np.sum(np.isfinite(initial_values)))
    candidate_counts_kept = np.zeros(len(thresholds), dtype=np.int64)
    best_by_bin = np.full(len(thresholds), np.nan, dtype=float)
    merged_bin_indices = []
    merged_objective_values = []
    merged_vectors = []
    for idx, values in candidates.items():
        candidate_counts_kept[idx] = len(values)
        if values:
            best_by_bin[idx] = values[0][0]
        for value, vector in values:
            merged_bin_indices.append(idx)
            merged_objective_values.append(float(value))
            merged_vectors.append(vector)

    sample_text = sample_label(total_candidates)
    out = in_dir / f"{spec.key}_nnse_batch_init_merged_{sample_text}.npz"
    np.savez_compressed(
        out,
        initial_population=initial_population,
        initial_objective_values=initial_values,
        initial_source_bins=source_bins,
        p0=p0,
        parameter_names=np.asarray(params, dtype=object),
        bin_thresholds=thresholds,
        bin_counts=bin_counts,
        candidate_counts_seen=candidate_counts_seen,
        candidate_counts_kept=candidate_counts_kept,
        best_by_bin=best_by_bin,
        candidate_bin_indices=np.asarray(merged_bin_indices, dtype=int),
        candidate_objective_values=np.asarray(merged_objective_values, dtype=float),
        candidate_vectors=np.asarray(merged_vectors, dtype=float)
        if merged_vectors
        else np.empty((0, len(params)), dtype=float),
        global_best_vector=best_vector,
        global_best_objective=np.asarray([best_value], dtype=float),
        chunk_files=np.asarray([str(path) for path in chunk_paths], dtype=object),
    )

    elapsed = time.time() - start
    summary = {
        "model": spec.key,
        "label": spec.label,
        "output": spec.output,
        "tag": args.tag,
        "chunks_merged": len(chunk_paths),
        "candidates": total_candidates,
        "finite_count": finite_count,
        "failed_count": failed_count,
        "overflow_count": overflow_count,
        "finite_fraction": finite_count / max(1, total_candidates),
        "placed_count": int(np.sum(bin_counts)),
        "placed_fraction": int(np.sum(bin_counts)) / max(1, total_candidates),
        "filled_thresholds": filled_bins,
        "best_objective": best_value,
        "elapsed_seconds": elapsed,
        "npz": str(out),
        "bin_counts": bin_counts.tolist(),
        "candidate_counts_seen": candidate_counts_seen.tolist(),
        "candidate_counts_kept": candidate_counts_kept.tolist(),
        "thresholds": thresholds.tolist(),
    }
    summary_path = out.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Merged {len(chunk_paths)} chunk files")
    print(f"Saved {out}")
    print(f"Saved {summary_path}")
    print(
        f"Done: candidates={total_candidates:,}, filled_thresholds={filled_bins}/{len(thresholds)}, "
        f"best={best_value:.3e}"
    )
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge chunked NNSE batch-initialisation runs")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--model", default="chen2004", choices=[spec.key for spec in SPECS])
    parser.add_argument("--keep-per-bin", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bins", type=int, default=50)
    parser.add_argument("--bin-min", type=float, default=1e-2)
    parser.add_argument("--bin-max", type=float, default=250.0)
    parser.add_argument("--bin-top", type=float, default=1000.0)
    parser.add_argument("--spacing", choices=["linear", "log"], default="log")
    main(parser.parse_args())

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsbw_nnse import NNSEConfig, get_spec, make_thresholds, setup_rr, simulate_output
from wsbw_pipeline import RESULTS, SPECS, prepare_models


OUT_ROOT = RESULTS / "nnse_batch_init"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

_WORKER: dict[str, Any] = {}
CHUNK_SEED_STRIDE = 10_000_019
DEFAULT_COUNT_CUTOFFS = "15,17.02705732642944,20,25,50,100,250"


def chunk_seed_offset(chunk_id: str | None) -> int:
    if chunk_id is None:
        return 0
    try:
        return int(chunk_id) * CHUNK_SEED_STRIDE
    except ValueError:
        return sum((idx + 1) * ord(char) for idx, char in enumerate(chunk_id)) * CHUNK_SEED_STRIDE


def parse_cutoffs(text: str | None) -> np.ndarray:
    if text is None or not text.strip():
        return np.empty(0, dtype=float)
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    return np.asarray(values, dtype=float)


@contextlib.contextmanager
def suppress_solver_stderr():
    saved_fd = os.dup(2)
    null_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null_fd, 2)
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)
        os.close(null_fd)


def format_duration(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.2f} hr"
    if seconds >= 60:
        return f"{seconds / 60:.2f} min"
    return f"{seconds:.1f} sec"


def place_in_bin(value: float, thresholds: np.ndarray) -> int | None:
    if not math.isfinite(value):
        return None
    idx = int(np.searchsorted(thresholds, value, side="left"))
    if idx >= len(thresholds):
        return None
    return idx


def _init_worker(model_key: str, audit_item: dict, config_dict: dict) -> None:
    spec = get_spec(model_key)
    params = audit_item["free_parameters"]
    rr, defaults, initials = setup_rr(spec, audit_item["promoted_sbml"], params)
    p0 = np.array([defaults[pid] for pid in params], dtype=float)
    ref = simulate_output(rr, spec, defaults, initials, params, p0)
    if ref is None:
        raise RuntimeError(f"Wildtype reference simulation failed for {model_key}")
    thresholds = make_thresholds(NNSEConfig(**config_dict))
    _WORKER.clear()
    _WORKER.update(
        {
            "spec": spec,
            "params": params,
            "rr": rr,
            "defaults": defaults,
            "initials": initials,
            "p0": p0,
            "ref_time": ref[0],
            "ref_signal": ref[1],
            "thresholds": thresholds,
            "count_cutoffs": parse_cutoffs(config_dict.get("count_cutoffs")),
        }
    )


def _objective(vector: np.ndarray) -> float:
    with contextlib.redirect_stderr(io.StringIO()):
        out = simulate_output(
            _WORKER["rr"],
            _WORKER["spec"],
            _WORKER["defaults"],
            _WORKER["initials"],
            _WORKER["params"],
            vector,
        )
    if out is None:
        return float("inf")
    t, y = out
    y0 = np.interp(t, _WORKER["ref_time"], _WORKER["ref_signal"])
    return float(np.trapz((y - y0) ** 2, t))


def _insert_candidate(candidates: dict[int, list[tuple[float, list[float]]]], bin_idx: int, value: float, vector: np.ndarray, keep: int) -> None:
    entries = candidates.setdefault(bin_idx, [])
    entries.append((float(value), vector.astype(float).tolist()))
    entries.sort(key=lambda item: item[0])
    if len(entries) > keep:
        del entries[keep:]


def _evaluate_batch(task: tuple[int, int, int, int]) -> dict:
    batch_id, count, seed, keep_per_bin = task
    rng = np.random.default_rng(seed)
    p0 = _WORKER["p0"]
    thresholds = _WORKER["thresholds"]
    count_cutoffs = _WORKER["count_cutoffs"]
    bin_counts = np.zeros(len(thresholds), dtype=np.int64)
    cutoff_counts = np.zeros(len(count_cutoffs), dtype=np.int64)
    candidates: dict[int, list[tuple[float, list[float]]]] = {}
    finite_count = 0
    failed_count = 0
    overflow_count = 0
    best_value = float("inf")
    best_vector: list[float] | None = None

    with suppress_solver_stderr():
        for _ in range(count):
            vector = 2.0 * p0 * rng.uniform(0.0, 1.0, size=len(p0))
            value = _objective(vector)
            if not math.isfinite(value):
                failed_count += 1
                continue
            finite_count += 1
            if value < best_value:
                best_value = float(value)
                best_vector = vector.astype(float).tolist()
            if len(count_cutoffs):
                cutoff_counts += value <= count_cutoffs
            bin_idx = place_in_bin(value, thresholds)
            if bin_idx is None:
                overflow_count += 1
                continue
            bin_counts[bin_idx] += 1
            _insert_candidate(candidates, bin_idx, value, vector, keep_per_bin)

    return {
        "batch_id": batch_id,
        "count": count,
        "finite_count": finite_count,
        "failed_count": failed_count,
        "overflow_count": overflow_count,
        "bin_counts": bin_counts.tolist(),
        "cutoff_counts": cutoff_counts.tolist(),
        "candidates": {str(key): value for key, value in candidates.items()},
        "best_value": best_value,
        "best_vector": best_vector,
    }


def chunk_sizes(total: int, chunks: int) -> list[int]:
    chunks = max(1, min(chunks, total))
    base, rem = divmod(total, chunks)
    return [base + (idx < rem) for idx in range(chunks)]


def merge_candidate_lists(
    target: dict[int, list[tuple[float, list[float]]]],
    incoming: dict[str, list[tuple[float, list[float]]]],
    keep_per_bin: int,
) -> None:
    for key, values in incoming.items():
        bin_idx = int(key)
        merged = target.setdefault(bin_idx, [])
        merged.extend((float(value), list(vector)) for value, vector in values)
        merged.sort(key=lambda item: item[0])
        if len(merged) > keep_per_bin:
            del merged[keep_per_bin:]


def build_initial_population(
    candidates: dict[int, list[tuple[float, list[float]]]],
    n_thresholds: int,
    n_params: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pools = {idx: sorted(values, key=lambda item: item[0]) for idx, values in candidates.items()}
    xs = np.full((n_thresholds, n_params), np.nan, dtype=float)
    fs = np.full(n_thresholds, np.nan, dtype=float)
    source_bins = np.full(n_thresholds, -1, dtype=int)

    for pos in range(n_thresholds):
        # Use the loosest available candidate that still satisfies this threshold.
        # This avoids wasting rare very-strict candidates on loose thresholds.
        for bin_idx in range(pos, -1, -1):
            if pools.get(bin_idx):
                value, vector = pools[bin_idx].pop(0)
                xs[pos] = np.asarray(vector, dtype=float)
                fs[pos] = float(value)
                source_bins[pos] = bin_idx
                break
    return xs, fs, source_bins


def run_batch_init(args: argparse.Namespace) -> Path:
    start = time.time()
    audit = prepare_models()
    spec = get_spec(args.model)
    audit_item = audit[spec.key]
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
    count_cutoffs = parse_cutoffs(args.count_cutoffs)
    params = audit_item["free_parameters"]
    _, defaults, _ = setup_rr(spec, audit_item["promoted_sbml"], params)
    p0 = np.array([defaults[pid] for pid in params], dtype=float)
    workers = max(1, args.workers)
    task_count = max(workers, math.ceil(args.candidates / args.batch_size))
    sizes = chunk_sizes(args.candidates, task_count)
    effective_seed = args.seed + chunk_seed_offset(args.chunk_id)
    tasks = [(idx, size, effective_seed + idx * 100_003, args.keep_per_bin) for idx, size in enumerate(sizes)]

    print(
        f"Screening {args.candidates:,} random candidates for {spec.label} "
        f"with {workers} workers across {len(tasks)} batches (seed {effective_seed})",
        flush=True,
    )

    bin_counts = np.zeros(len(thresholds), dtype=np.int64)
    cutoff_counts = np.zeros(len(count_cutoffs), dtype=np.int64)
    candidates: dict[int, list[tuple[float, list[float]]]] = {}
    finite_count = 0
    failed_count = 0
    overflow_count = 0
    best_value = float("inf")
    best_vector: list[float] | None = None
    completed_candidates = 0

    config_dict = {
        "model": config.model,
        "n_bins": config.n_bins,
        "bin_min": config.bin_min,
        "bin_max": config.bin_max,
        "bin_top": config.bin_top,
        "spacing": config.spacing,
        "seed": config.seed,
        "count_cutoffs": args.count_cutoffs,
    }
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(spec.key, audit_item, config_dict),
    ) as executor:
        futures = [executor.submit(_evaluate_batch, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            bin_counts += np.asarray(result["bin_counts"], dtype=np.int64)
            if len(count_cutoffs):
                cutoff_counts += np.asarray(result["cutoff_counts"], dtype=np.int64)
            finite_count += int(result["finite_count"])
            failed_count += int(result["failed_count"])
            overflow_count += int(result["overflow_count"])
            completed_candidates += int(result["count"])
            merge_candidate_lists(candidates, result["candidates"], args.keep_per_bin)
            if result["best_value"] < best_value:
                best_value = float(result["best_value"])
                best_vector = result["best_vector"]
            if done == 1 or done == len(futures) or done % max(1, len(futures) // 20) == 0:
                elapsed = time.time() - start
                rate = completed_candidates / elapsed if elapsed else 0.0
                filled_preview = sum(1 for idx in range(len(thresholds)) if any(candidates.get(j) for j in range(idx + 1)))
                print(
                    f"  batches {done}/{len(futures)}; elapsed={format_duration(elapsed)}; "
                    f"rate={rate:.1f} candidates/s; finite={finite_count:,}; "
                    f"best={best_value:.3e}; fillable_bins~{filled_preview}/{len(thresholds)}",
                    flush=True,
                )

    initial_population, initial_values, source_bins = build_initial_population(candidates, len(thresholds), len(params))
    filled_bins = int(np.sum(np.isfinite(initial_values)))
    elapsed = time.time() - start
    tag = args.tag or f"{spec.key}_batch_{args.candidates}"
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_suffix = f"_chunk-{args.chunk_id}" if args.chunk_id is not None else ""
    out = out_dir / f"{spec.key}_nnse_batch_init_N={args.candidates}{chunk_suffix}.npz"

    candidate_counts_kept = np.zeros(len(thresholds), dtype=np.int64)
    best_by_bin = np.full(len(thresholds), np.nan, dtype=float)
    candidate_bin_indices: list[int] = []
    candidate_objective_values: list[float] = []
    candidate_vectors: list[list[float]] = []
    for idx, values in candidates.items():
        candidate_counts_kept[idx] = len(values)
        if values:
            best_by_bin[idx] = values[0][0]
        for value, vector in values:
            candidate_bin_indices.append(idx)
            candidate_objective_values.append(float(value))
            candidate_vectors.append(vector)

    np.savez_compressed(
        out,
        initial_population=initial_population,
        initial_objective_values=initial_values,
        initial_source_bins=source_bins,
        p0=p0,
        parameter_names=np.asarray(params, dtype=object),
        bin_thresholds=thresholds,
        bin_counts=bin_counts,
        count_cutoffs=count_cutoffs,
        cutoff_counts=cutoff_counts,
        candidate_counts_kept=candidate_counts_kept,
        best_by_bin=best_by_bin,
        candidate_bin_indices=np.asarray(candidate_bin_indices, dtype=int),
        candidate_objective_values=np.asarray(candidate_objective_values, dtype=float),
        candidate_vectors=np.asarray(candidate_vectors, dtype=float)
        if candidate_vectors
        else np.empty((0, len(params)), dtype=float),
        global_best_vector=np.asarray(best_vector if best_vector is not None else np.full(len(params), np.nan), dtype=float),
        global_best_objective=np.asarray([best_value], dtype=float),
    )

    summary = {
        "model": spec.key,
        "label": spec.label,
        "output": spec.output,
        "candidates": args.candidates,
        "chunk_id": args.chunk_id,
        "workers": workers,
        "batches": len(tasks),
        "batch_size_target": args.batch_size,
        "seed": effective_seed,
        "base_seed": args.seed,
        "chunk_seed_offset": chunk_seed_offset(args.chunk_id),
        "spacing": args.spacing,
        "n_thresholds": len(thresholds),
        "parameter_count": len(params),
        "finite_count": finite_count,
        "failed_count": failed_count,
        "overflow_count": overflow_count,
        "finite_fraction": finite_count / max(1, args.candidates),
        "placed_count": int(np.sum(bin_counts)),
        "placed_fraction": int(np.sum(bin_counts)) / max(1, args.candidates),
        "filled_thresholds": filled_bins,
        "best_objective": best_value,
        "elapsed_seconds": elapsed,
        "candidates_per_second": args.candidates / elapsed if elapsed else None,
        "npz": str(out),
        "bin_counts": bin_counts.tolist(),
        "candidate_counts_kept": candidate_counts_kept.tolist(),
        "thresholds": thresholds.tolist(),
        "count_cutoffs": count_cutoffs.tolist(),
        "cutoff_counts": cutoff_counts.tolist(),
        "cutoff_fractions": (cutoff_counts / max(1, args.candidates)).tolist(),
    }
    summary_path = out.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved {out}", flush=True)
    print(f"Saved {summary_path}", flush=True)
    print(
        f"Done in {format_duration(elapsed)}: finite={finite_count:,}/{args.candidates:,}, "
        f"placed={int(np.sum(bin_counts)):,}, filled_thresholds={filled_bins}/{len(thresholds)}, "
        f"best={best_value:.3e}",
        flush=True,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel random-candidate NNSE initialiser")
    parser.add_argument("--model", default="chen2004", choices=[spec.key for spec in SPECS])
    parser.add_argument("--candidates", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--keep-per-bin", type=int, default=32)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--chunk-id", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bins", type=int, default=50)
    parser.add_argument("--bin-min", type=float, default=1e-2)
    parser.add_argument("--bin-max", type=float, default=250.0)
    parser.add_argument("--bin-top", type=float, default=1000.0)
    parser.add_argument("--spacing", choices=["linear", "log"], default="log")
    parser.add_argument("--count-cutoffs", default=DEFAULT_COUNT_CUTOFFS)
    args = parser.parse_args()
    run_batch_init(args)


if __name__ == "__main__":
    main()

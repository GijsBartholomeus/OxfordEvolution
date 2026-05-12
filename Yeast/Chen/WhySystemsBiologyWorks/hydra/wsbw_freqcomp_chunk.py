from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsbw_nnse import get_spec, setup_rr
from wsbw_pipeline import RESULTS, clz, prepare_models
from hydra.wsbw_bruteforce_cloud_chunk import (
    _init_worker,
    _simulate_vector,
    _WORKER,
    bits_to_uint64,
    chunk_seed_offset,
    chunk_sizes,
    format_duration,
    suppress_solver_stderr,
)


OUT_ROOT = RESULTS / "freqcomp_chunks"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def sample_label(samples: int) -> str:
    exponent = round(math.log10(samples)) if samples > 0 else 0
    if samples > 0 and 10**exponent == samples:
        return f"1e{exponent}"
    return str(samples)


def _evaluate_freq_batch(task: tuple[int, int, int]) -> dict[str, Any]:
    batch_id, count, seed = task
    rng = np.random.default_rng(seed)
    p0 = _WORKER["p0"]
    counts: Counter[tuple[int, float]] = Counter()
    failures = 0

    with suppress_solver_stderr(), contextlib.redirect_stderr(io.StringIO()):
        for _ in range(count):
            vector = 2.0 * p0 * rng.uniform(0.0, 1.0, size=len(p0))
            try:
                result = _simulate_vector(vector)
            except Exception:
                result = None
            if result is None:
                failures += 1
                continue
            code, complexity, _objective = result
            counts[(int(code), float(complexity))] += 1
    return {"batch_id": batch_id, "attempted": count, "failures": failures, "counts": counts}


def run_chunk(args: argparse.Namespace) -> Path:
    start = time.time()
    audit = prepare_models()
    spec = get_spec(args.model)
    audit_item = audit[spec.key]
    params = audit_item["free_parameters"]
    workers = max(1, args.workers)
    task_count = max(workers, math.ceil(args.samples / args.batch_size))
    sizes = chunk_sizes(args.samples, task_count)
    effective_seed = args.seed + chunk_seed_offset(args.chunk_id)
    tasks = [(idx, size, effective_seed + idx * 100_003) for idx, size in enumerate(sizes)]

    print(
        f"FreqComp-only sampling {args.samples:,} random points for {spec.label} "
        f"with {workers} workers across {len(tasks)} batches (seed {effective_seed})",
        flush=True,
    )

    phenotype_counts: Counter[tuple[int, float]] = Counter()
    failures = 0
    successes = 0
    completed = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(spec.key, audit_item)) as executor:
        futures = [executor.submit(_evaluate_freq_batch, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            failures += int(result["failures"])
            completed += int(result["attempted"])
            batch_counts = result["counts"]
            phenotype_counts.update(batch_counts)
            successes += int(sum(batch_counts.values()))
            if done == 1 or done == len(futures) or done % max(1, len(futures) // 20) == 0:
                elapsed = time.time() - start
                rate = completed / elapsed if elapsed else 0.0
                print(
                    f"  batches {done}/{len(futures)}; elapsed={format_duration(elapsed)}; "
                    f"rate={rate:.1f} points/s; successes={successes:,}; failures={failures:,}; "
                    f"phenotypes={len(phenotype_counts):,}",
                    flush=True,
                )

    rr, defaults, _ = setup_rr(spec, audit_item["promoted_sbml"], params)
    p0 = np.asarray([defaults[pid] for pid in params], dtype=np.float32)
    if "wt_bits" not in _WORKER:
        _init_worker(spec.key, audit_item)
    wt_bits = _WORKER["wt_bits"]

    tag = args.tag or f"{spec.key}_freqcomp"
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_suffix = f"_chunk-{args.chunk_id}" if args.chunk_id is not None else ""
    out = out_dir / f"{spec.key}_freqcomp_N={sample_label(args.samples)}{chunk_suffix}.json"

    data = {
        "model": spec.key,
        "label": spec.label,
        "tag": tag,
        "chunk_id": args.chunk_id,
        "samples": int(args.samples),
        "successes": int(successes),
        "failures": int(failures),
        "seed": int(effective_seed),
        "wildtype_encoding": wt_bits,
        "wildtype_complexity": clz(wt_bits),
        "wildtype_count": int(phenotype_counts.get((bits_to_uint64(wt_bits), float(clz(wt_bits))), 0)),
        "parameter_names": list(params),
        "p0": [float(x) for x in p0],
        "elapsed_seconds": float(time.time() - start),
        "phenotypes": [
            {"code": str(code), "encoding": format(int(code), "049b"), "complexity": float(complexity), "count": int(count)}
            for (code, complexity), count in phenotype_counts.items()
        ],
    }
    out.write_text(json.dumps(data))
    print(f"Saved {out}")
    print(
        f"Done in {format_duration(time.time() - start)}: successes={successes:,}/{args.samples:,}, "
        f"failures={failures:,}, phenotypes={len(phenotype_counts):,}",
        flush=True,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="FreqComp-only brute-force chunk")
    parser.add_argument("--model", default="chen2004")
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--tag", default="chen_freqcomp")
    parser.add_argument("--chunk-id", default=None)
    parser.add_argument("--seed", type=int, default=42)
    run_chunk(parser.parse_args())


if __name__ == "__main__":
    main()

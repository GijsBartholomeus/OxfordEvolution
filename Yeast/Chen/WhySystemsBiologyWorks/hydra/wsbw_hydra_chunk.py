from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsbw_pipeline import DIVERGENCE_CAP_FACTOR, RESULTS, SPECS, clz, prepare_models
from wsbw_pipeline_parallel import chunk_sizes, get_spec, process_chunk, wildtype_metadata


CHUNK_DIR = RESULTS / "hydra_chunks"
CHUNK_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_SEED_STRIDE = 10_000_019


def chunk_seed_offset(chunk_id: str) -> int:
    try:
        return int(chunk_id) * CHUNK_SEED_STRIDE
    except ValueError:
        return sum((idx + 1) * ord(char) for idx, char in enumerate(chunk_id)) * CHUNK_SEED_STRIDE


def run_model_chunk(
    spec_key: str,
    audit: dict,
    samples: int,
    seed: int,
    workers: int,
    chunks_per_worker: int,
) -> dict:
    spec = get_spec(spec_key)
    audit_item = audit[spec.key]
    meta = wildtype_metadata(spec.key, audit_item, seed)
    sizes = chunk_sizes(samples, max(1, workers * chunks_per_worker))
    tasks = [
        (spec.key, audit_item, size, seed + idx * 10_000, meta["divergence_cap"])
        for idx, size in enumerate(sizes)
        if size > 0
    ]
    merged = Counter()
    failures = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_chunk, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            merged.update(result["counts"])
            failures += result["failures"]
            print(f"  {spec.label}: finished local chunk {done}/{len(futures)}", flush=True)

    wt_bits = meta["wildtype_encoding"]
    return {
        "model": spec.key,
        "label": spec.label,
        "samples": samples,
        "successes": sum(merged.values()),
        "failures": failures,
        "wildtype_encoding": wt_bits,
        "wildtype_complexity": meta["wildtype_complexity"],
        "wildtype_count": merged.get(wt_bits, 0) if wt_bits else 0,
        "wildtype_max_abs": meta["wildtype_max_abs"],
        "divergence_cap_factor": DIVERGENCE_CAP_FACTOR,
        "divergence_cap": meta["divergence_cap"],
        "time_window": {
            "t_end": spec.t_end,
            "coarse_start": spec.coarse_start,
            "coarse_duration": spec.coarse_duration,
        },
        "hydra_chunk": True,
        "phenotypes": [{"encoding": enc, "count": n, "complexity": clz(enc)} for enc, n in merged.items()],
    }


def main(
    samples_per_model: int,
    seed: int,
    workers: int,
    chunk_id: str,
    tag: str,
    chunks_per_worker: int,
    models: list[str] | None,
) -> Path:
    audit = prepare_models()
    selected = [spec for spec in SPECS if models is None or spec.key in models]
    out_dir = CHUNK_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    effective_seed = seed + chunk_seed_offset(chunk_id)
    for idx, spec in enumerate(selected):
        model_seed = effective_seed + idx * 1_000_000
        print(
            f"Running chunk {chunk_id} for {spec.label} with {samples_per_model} samples "
            f"(seed {model_seed})",
            flush=True,
        )
        data = run_model_chunk(
            spec.key,
            audit,
            samples=samples_per_model,
            seed=model_seed,
            workers=workers,
            chunks_per_worker=chunks_per_worker,
        )
        data["seed"] = model_seed
        data["base_seed"] = seed
        data["chunk_seed_offset"] = chunk_seed_offset(chunk_id)
        out = out_dir / f"{spec.key}_chunk-{chunk_id}.json"
        out.write_text(json.dumps(data))
        written.append(out)
        print(out, flush=True)

    manifest = out_dir / f"manifest_chunk-{chunk_id}.txt"
    manifest.write_text("\n".join(str(path) for path in written))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one Hydra chunk of the WSBW GP-map sampling")
    parser.add_argument("--samples-per-model", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-id", default="0")
    parser.add_argument("--tag", default="hydra_run")
    parser.add_argument("--chunks-per-worker", type=int, default=2)
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()
    main(
        samples_per_model=args.samples_per_model,
        seed=args.seed,
        workers=args.workers,
        chunk_id=args.chunk_id,
        tag=args.tag,
        chunks_per_worker=args.chunks_per_worker,
        models=args.models,
    )

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from wsbw_nnse import NNSEConfig, format_duration, get_spec, make_thresholds, setup_rr, simulate_output
from wsbw_pipeline import RESULTS, SPECS, prepare_models


OUT_ROOT = RESULTS / "nnse_parallel"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

_WORKER: dict[str, Any] = {}


@dataclass
class ParallelNNSEConfig:
    model: str = "chen2004"
    init_npz: str | None = None
    steps: int = 6000
    workers: int = max(1, os.cpu_count() or 1)
    sigma: float = 0.01
    seed: int = 42
    chain_id: str = "0"
    tag: str = "nnse_parallel"
    neutral_threshold: float | None = 15.0
    checkpoint_every: int = 250
    extra_steps: int = 1000
    target_empty: int = 1
    refill_attempts: int = 0
    n_bins: int = 50
    bin_min: float = 1e-2
    bin_max: float = 250.0
    bin_top: float = 1000.0
    spacing: str = "log"


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


def _evaluate_task(task: tuple[int, np.ndarray]) -> tuple[int, float]:
    idx, vector = task
    with suppress_solver_stderr():
        value = _objective(vector)
    return idx, value


def refill_new_empty_positions(
    xs: list[np.ndarray | None],
    fs: list[float | None],
    positions: list[int],
    p0: np.ndarray,
    thresholds: np.ndarray,
    rng: np.random.Generator,
    attempts_per_position: int,
    executor: ProcessPoolExecutor,
) -> int:
    if attempts_per_position <= 0 or not positions:
        return 0
    task_to_position: dict[int, tuple[int, np.ndarray]] = {}
    tasks: list[tuple[int, np.ndarray]] = []
    task_id = 0
    for pos in positions:
        for _ in range(attempts_per_position):
            vector = 2.0 * p0 * rng.uniform(0.0, 1.0, size=len(p0))
            task_to_position[task_id] = (pos, vector)
            tasks.append((task_id, vector))
            task_id += 1

    best_by_position: dict[int, tuple[float, np.ndarray]] = {}
    futures = [executor.submit(_evaluate_task, task) for task in tasks]
    for future in as_completed(futures):
        task_idx, value = future.result()
        if not math.isfinite(value):
            continue
        pos, vector = task_to_position[task_idx]
        if value > thresholds[pos]:
            continue
        current = best_by_position.get(pos)
        if current is None or value < current[0]:
            best_by_position[pos] = (float(value), vector)

    filled = 0
    for pos, (value, vector) in best_by_position.items():
        if xs[pos] is None or fs[pos] is None or value < fs[pos]:
            xs[pos] = vector
            fs[pos] = value
            filled += 1
    return filled


def load_initial_population(
    init_npz: Path,
    expected_params: list[str],
    expected_thresholds: np.ndarray,
) -> tuple[list[np.ndarray | None], list[float | None]]:
    data = np.load(init_npz, allow_pickle=True)
    required = {"initial_population", "initial_objective_values", "parameter_names", "bin_thresholds"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"{init_npz} is missing arrays: {sorted(missing)}")

    init_params = [str(item) for item in data["parameter_names"]]
    if init_params != list(expected_params):
        raise ValueError("Initial-population parameter order does not match current model audit")

    init_thresholds = np.asarray(data["bin_thresholds"], dtype=float)
    if init_thresholds.shape != expected_thresholds.shape or not np.allclose(init_thresholds, expected_thresholds):
        raise ValueError("Initial-population NNSE thresholds do not match current configuration")

    population = np.asarray(data["initial_population"], dtype=float)
    values = np.asarray(data["initial_objective_values"], dtype=float)
    if population.shape != (len(expected_thresholds), len(expected_params)):
        raise ValueError(f"Unexpected initial_population shape: {population.shape}")
    if values.shape != (len(expected_thresholds),):
        raise ValueError(f"Unexpected initial_objective_values shape: {values.shape}")

    xs: list[np.ndarray | None] = []
    fs: list[float | None] = []
    for vector, value in zip(population, values):
        if np.all(np.isfinite(vector)) and math.isfinite(float(value)):
            xs.append(vector.astype(float).copy())
            fs.append(float(value))
        else:
            xs.append(None)
            fs.append(None)
    if not any(x is not None for x in xs):
        raise ValueError(f"No filled NNSE bins found in {init_npz}")
    return xs, fs


def make_mutation_tasks(
    xs: list[np.ndarray | None],
    fs: list[float | None],
    p0: np.ndarray,
    rng: np.random.Generator,
    sigma: float,
) -> tuple[list[tuple[int, np.ndarray]], dict[int, tuple[np.ndarray, float]]]:
    tasks: list[tuple[int, np.ndarray]] = []
    originals: dict[int, tuple[np.ndarray, float]] = {}
    for idx, (x, fx) in enumerate(zip(xs, fs)):
        if x is None or fx is None:
            continue
        u = x / (2.0 * p0)
        u_mut = (u + rng.normal(0.0, sigma, size=len(p0))) % 1.0
        x_mut = 2.0 * p0 * u_mut
        tasks.append((idx, x_mut))
        originals[idx] = (x, fx)
    return tasks, originals


def apply_mutation_results(
    n: int,
    tasks: list[tuple[int, np.ndarray]],
    originals: dict[int, tuple[np.ndarray, float]],
    values: dict[int, float],
    thresholds: np.ndarray,
) -> tuple[list[np.ndarray | None], list[float | None], int]:
    proposed_xs: list[np.ndarray | None] = [None] * n
    proposed_fs: list[float | None] = [None] * n
    accepted = 0
    task_vectors = {idx: vector for idx, vector in tasks}
    for idx, (x, fx) in originals.items():
        value = values.get(idx, float("inf"))
        if value <= thresholds[idx]:
            proposed_xs[idx] = task_vectors[idx]
            proposed_fs[idx] = float(value)
            accepted += 1
        else:
            proposed_xs[idx] = x.copy()
            proposed_fs[idx] = float(fx)
    return proposed_xs, proposed_fs, accepted


def apply_swaps(xs: list[np.ndarray | None], fs: list[float | None], thresholds: np.ndarray) -> list[tuple[int, int]]:
    swaps: list[tuple[int, int]] = []
    for idx in range(len(xs) - 1, 0, -1):
        if fs[idx] is None:
            continue
        if fs[idx] <= thresholds[idx - 1]:
            xs[idx], xs[idx - 1] = xs[idx - 1], xs[idx]
            fs[idx], fs[idx - 1] = fs[idx - 1], fs[idx]
            swaps.append((idx, idx - 1))
    return swaps


def checkpoint(
    out: Path,
    config: ParallelNNSEConfig,
    xs: list[np.ndarray | None],
    fs: list[float | None],
    p0: np.ndarray,
    params: list[str],
    thresholds: np.ndarray,
    best_history: list[float],
    neutral_points: list[np.ndarray],
    neutral_values: list[float],
    swap_count: np.ndarray,
    opportunity_count: np.ndarray,
    reference: dict[str, np.ndarray],
    step: int,
) -> None:
    final_x = np.array([x if x is not None else np.full_like(p0, np.nan) for x in xs])
    final_f = np.array([fx if fx is not None else np.nan for fx in fs], dtype=float)
    neutral = np.unique(np.asarray(neutral_points, dtype=float), axis=0) if neutral_points else np.empty((0, len(p0)))
    neutral_f = np.asarray(neutral_values, dtype=float)
    volume_ratios = np.divide(
        swap_count,
        opportunity_count,
        out=np.full_like(swap_count, np.nan, dtype=float),
        where=opportunity_count > 0,
    )
    np.savez_compressed(
        out,
        neutral_points=neutral,
        neutral_objective_values=neutral_f,
        final_population=final_x,
        final_objective_values=final_f,
        p0=p0,
        parameter_names=np.asarray(params, dtype=object),
        bin_thresholds=thresholds,
        volume_ratios=volume_ratios,
        swap_count=swap_count,
        opportunity_count=opportunity_count,
        best_history=np.asarray(best_history, dtype=float),
        reference_time=reference["time"],
        reference_signal=reference["signal"],
        step=np.asarray([step], dtype=int),
        config=np.asarray([json.dumps(asdict(config))], dtype=object),
    )


def run_parallel_nnse(config: ParallelNNSEConfig) -> Path:
    if config.init_npz is None:
        raise ValueError("--init-npz is required for the parallel NNSE runner")
    rng = np.random.default_rng(config.seed)
    audit = prepare_models()
    spec = get_spec(config.model)
    audit_item = audit[spec.key]
    params = audit_item["free_parameters"]
    rr, defaults, initials = setup_rr(spec, audit_item["promoted_sbml"], params)
    p0 = np.array([defaults[pid] for pid in params], dtype=float)
    ref = simulate_output(rr, spec, defaults, initials, params, p0)
    if ref is None:
        raise RuntimeError(f"Wildtype reference simulation failed for {spec.key}")
    reference = {"time": ref[0], "signal": ref[1]}
    thresholds = make_thresholds(
        NNSEConfig(
            model=config.model,
            n_bins=config.n_bins,
            bin_min=config.bin_min,
            bin_max=config.bin_max,
            bin_top=config.bin_top,
            spacing=config.spacing,
            seed=config.seed,
        )
    )
    if config.neutral_threshold is None:
        config.neutral_threshold = float(thresholds[0])

    xs, fs = load_initial_population(Path(config.init_npz), params, thresholds)
    out_dir = OUT_ROOT / config.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{spec.key}_nnse_parallel_chain-{config.chain_id}_seed-{config.seed}.npz"
    summary_path = out.with_suffix(".json")

    config_dict = {
        "model": config.model,
        "n_bins": config.n_bins,
        "bin_min": config.bin_min,
        "bin_max": config.bin_max,
        "bin_top": config.bin_top,
        "spacing": config.spacing,
        "seed": config.seed,
    }
    workers = max(1, int(config.workers))
    best_history: list[float] = []
    neutral_points: list[np.ndarray] = []
    neutral_values: list[float] = []
    swap_count = np.zeros(len(thresholds), dtype=int)
    opportunity_count = np.zeros(len(thresholds), dtype=int)
    reached_target_at = None
    total_accepted = 0
    total_mutations = 0
    start = time.time()

    print(
        f"Starting parallel NNSE {spec.label}: steps={config.steps:,}, workers={workers}, "
        f"filled_start={sum(x is not None for x in xs)}/{len(xs)}, init={config.init_npz}",
        flush=True,
    )
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(spec.key, audit_item, config_dict),
    ) as executor:
        for step in range(config.steps):
            tasks, originals = make_mutation_tasks(xs, fs, p0, rng, config.sigma)
            futures = [executor.submit(_evaluate_task, task) for task in tasks]
            values = {}
            for future in as_completed(futures):
                idx, value = future.result()
                values[idx] = value
            total_mutations += len(tasks)
            xs, fs, accepted = apply_mutation_results(len(xs), tasks, originals, values, thresholds)
            total_accepted += accepted

            empty_before = {idx for idx, (x, fx) in enumerate(zip(xs, fs)) if x is None or fx is None}
            swap_set = set(apply_swaps(xs, fs, thresholds))
            empty_after = {idx for idx, (x, fx) in enumerate(zip(xs, fs)) if x is None or fx is None}
            refill_new_empty_positions(
                xs,
                fs,
                sorted(empty_after - empty_before),
                p0,
                thresholds,
                rng,
                config.refill_attempts,
                executor,
            )
            for idx in range(1, len(fs)):
                if fs[idx] is not None and fs[idx - 1] is not None:
                    opportunity_count[idx] += 1
                    if (idx, idx - 1) in swap_set:
                        swap_count[idx] += 1

            filled = sum(x is not None for x in xs)
            empty = len(xs) - filled
            finite = [fx for fx in fs if fx is not None and math.isfinite(fx)]
            best = min(finite) if finite else float("inf")
            best_history.append(best)

            for x, fx in zip(xs, fs):
                if x is not None and fx is not None and fx <= config.neutral_threshold:
                    neutral_points.append(x.copy())
                    neutral_values.append(float(fx))

            if reached_target_at is None and empty <= config.target_empty:
                reached_target_at = step
            if reached_target_at is not None and step >= reached_target_at + config.extra_steps:
                print(f"Stopping after target fill plus extra steps at step {step + 1:,}", flush=True)
                break

            if (step + 1) % max(1, config.checkpoint_every) == 0 or step == 0:
                elapsed = time.time() - start
                rate = (step + 1) / elapsed if elapsed else 0.0
                acceptance = total_accepted / max(1, total_mutations)
                print(
                    f"{spec.key} chain {config.chain_id} step {step + 1:,}: "
                    f"filled={filled}/{len(xs)}, best={best:.3e}, neutral_records={len(neutral_points):,}, "
                    f"acceptance={acceptance:.3f}, elapsed={format_duration(elapsed)}, rate={rate:.2f} step/s",
                    flush=True,
                )
                checkpoint(
                    out,
                    config,
                    xs,
                    fs,
                    p0,
                    params,
                    thresholds,
                    best_history,
                    neutral_points,
                    neutral_values,
                    swap_count,
                    opportunity_count,
                    reference,
                    step + 1,
                )

    elapsed = time.time() - start
    checkpoint(
        out,
        config,
        xs,
        fs,
        p0,
        params,
        thresholds,
        best_history,
        neutral_points,
        neutral_values,
        swap_count,
        opportunity_count,
        reference,
        len(best_history),
    )
    final_f = np.array([fx if fx is not None else np.nan for fx in fs], dtype=float)
    summary = {
        "config": asdict(config),
        "model": spec.key,
        "label": spec.label,
        "output": spec.output,
        "parameter_count": len(params),
        "steps_completed": len(best_history),
        "reached_target_at": reached_target_at,
        "elapsed_seconds": elapsed,
        "neutral_points": int(len(np.unique(np.asarray(neutral_points, dtype=float), axis=0))) if neutral_points else 0,
        "neutral_records": len(neutral_points),
        "final_filled": int(np.sum(~np.isnan(final_f))),
        "best_objective": float(np.nanmin(final_f)),
        "acceptance_fraction": total_accepted / max(1, total_mutations),
        "npz": str(out),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved {out}", flush=True)
    print(f"Saved {summary_path}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel NNSE runner from a precomputed initial population")
    parser.add_argument("--model", default="chen2004", choices=[spec.key for spec in SPECS])
    parser.add_argument("--init-npz", required=True)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chain-id", default="0")
    parser.add_argument("--tag", default="nnse_parallel")
    parser.add_argument("--neutral-threshold", type=float, default=15.0)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--extra-steps", type=int, default=1000)
    parser.add_argument("--target-empty", type=int, default=1)
    parser.add_argument("--refill-attempts", type=int, default=0)
    parser.add_argument("--n-bins", type=int, default=50)
    parser.add_argument("--bin-min", type=float, default=1e-2)
    parser.add_argument("--bin-max", type=float, default=250.0)
    parser.add_argument("--bin-top", type=float, default=1000.0)
    parser.add_argument("--spacing", choices=["linear", "log"], default="log")
    run_parallel_nnse(ParallelNNSEConfig(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()

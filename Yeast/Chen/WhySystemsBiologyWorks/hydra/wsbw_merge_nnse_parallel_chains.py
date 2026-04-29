from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsbw_pipeline import RESULTS, SPECS


OUT_ROOT = RESULTS / "nnse_parallel"


def main(args: argparse.Namespace) -> Path:
    in_dir = OUT_ROOT / args.tag
    if not in_dir.exists():
        raise FileNotFoundError(f"No NNSE chain directory found: {in_dir}")

    paths = sorted(in_dir.glob(f"{args.model}_nnse_parallel_chain-*_seed-*.npz"))
    if not paths:
        raise FileNotFoundError(f"No NNSE chain npz files found for {args.model} in {in_dir}")

    neutral_points = []
    neutral_values = []
    final_populations = []
    final_values = []
    best_histories = []
    chain_steps = []
    parameter_names = None
    thresholds = None
    p0 = None
    reference_time = None
    reference_signal = None

    for path in paths:
        data = np.load(path, allow_pickle=True)
        points = np.asarray(data["neutral_points"], dtype=float)
        values = np.asarray(data["neutral_objective_values"], dtype=float)
        if points.size:
            neutral_points.append(points)
            neutral_values.append(values)
        final_populations.append(np.asarray(data["final_population"], dtype=float))
        final_values.append(np.asarray(data["final_objective_values"], dtype=float))
        best_histories.append(np.asarray(data["best_history"], dtype=float))
        chain_steps.append(int(np.asarray(data["step"]).ravel()[0]) if "step" in data.files else len(best_histories[-1]))

        if parameter_names is None:
            parameter_names = np.asarray(data["parameter_names"], dtype=object)
            thresholds = np.asarray(data["bin_thresholds"], dtype=float)
            p0 = np.asarray(data["p0"], dtype=float)
            reference_time = np.asarray(data["reference_time"], dtype=float)
            reference_signal = np.asarray(data["reference_signal"], dtype=float)

    if neutral_points:
        neutral = np.concatenate(neutral_points, axis=0)
        neutral_f = np.concatenate(neutral_values, axis=0)
        unique_neutral, unique_idx = np.unique(neutral, axis=0, return_index=True)
        unique_values = neutral_f[unique_idx]
    else:
        n_params = len(parameter_names) if parameter_names is not None else 0
        unique_neutral = np.empty((0, n_params), dtype=float)
        unique_values = np.empty(0, dtype=float)

    max_history = max((len(history) for history in best_histories), default=0)
    best_history_matrix = np.full((len(best_histories), max_history), np.nan, dtype=float)
    for idx, history in enumerate(best_histories):
        best_history_matrix[idx, : len(history)] = history

    out = in_dir / f"{args.model}_nnse_parallel_merged_{len(paths)}chains.npz"
    np.savez_compressed(
        out,
        neutral_points=unique_neutral,
        neutral_objective_values=unique_values,
        final_populations=np.asarray(final_populations, dtype=float),
        final_objective_values=np.asarray(final_values, dtype=float),
        p0=p0,
        parameter_names=parameter_names,
        bin_thresholds=thresholds,
        best_history_by_chain=best_history_matrix,
        chain_steps=np.asarray(chain_steps, dtype=int),
        chain_files=np.asarray([str(path) for path in paths], dtype=object),
        reference_time=reference_time,
        reference_signal=reference_signal,
    )

    best_final = min(float(np.nanmin(values)) for values in final_values)
    summary = {
        "model": args.model,
        "tag": args.tag,
        "chains_merged": len(paths),
        "neutral_points": int(len(unique_neutral)),
        "neutral_records_before_unique": int(sum(len(points) for points in neutral_points)),
        "best_final_objective": best_final,
        "chain_steps": chain_steps,
        "npz": str(out),
    }
    summary_path = out.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Merged {len(paths)} chains")
    print(f"Saved {out}")
    print(f"Saved {summary_path}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge parallel NNSE chain outputs")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--model", default="chen2004", choices=[spec.key for spec in SPECS])
    main(parser.parse_args())

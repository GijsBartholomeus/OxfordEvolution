#!/usr/bin/env python3
"""Redraw brute-force summary figures from existing merged outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "hydra"))

from wsbw_merge_bruteforce_cloud import STATS_ROOT, plot_summary
from wsbw_pipeline import SPECS


def main(args: argparse.Namespace) -> None:
    stats_dir = STATS_ROOT / args.tag
    freq_json = stats_dir / f"{args.model}_complexity_frequency_{args.tag}.json"
    sample_npz = stats_dir / f"{args.model}_bruteforce_samples_{args.tag}.npz"
    if not freq_json.exists():
        raise FileNotFoundError(freq_json)
    if not sample_npz.exists():
        raise FileNotFoundError(sample_npz)

    data = json.loads(freq_json.read_text())
    sample = np.load(sample_npz, allow_pickle=True)
    point_sample = {
        "points": np.asarray(sample["all_points"], dtype=float),
        "complexities": np.asarray(sample["all_complexities"], dtype=float),
        "objectives": np.asarray(sample["all_objectives"], dtype=float),
    }
    rng = np.random.default_rng(args.seed)
    plot_path = plot_summary(
        stats_dir,
        args.model,
        data,
        np.asarray(sample["p0"], dtype=float),
        point_sample,
        args.tag,
        rng,
        args.max_plot_points,
    )
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--model", required=True, choices=[spec.key for spec in SPECS])
    parser.add_argument("--max-plot-points", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())

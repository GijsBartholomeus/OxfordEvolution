from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsbw_pipeline import PLOTS, RESULTS, SPECS, clz, plot_complexity_frequency, sample_size_label


CHUNK_DIR = RESULTS / "hydra_chunks"


def merge_model(chunks: list[dict]) -> dict:
    if not chunks:
        raise ValueError("Cannot merge an empty chunk list")
    first = chunks[0]
    merged = Counter()
    samples = 0
    failures = 0
    for chunk in chunks:
        samples += int(chunk["samples"])
        failures += int(chunk["failures"])
        for phenotype in chunk["phenotypes"]:
            merged[phenotype["encoding"]] += int(phenotype["count"])

    wt_bits = first.get("wildtype_encoding")
    return {
        "model": first["model"],
        "label": first["label"],
        "samples": samples,
        "successes": sum(merged.values()),
        "failures": failures,
        "wildtype_encoding": wt_bits,
        "wildtype_complexity": first.get("wildtype_complexity"),
        "wildtype_count": merged.get(wt_bits, 0) if wt_bits else 0,
        "wildtype_max_abs": first.get("wildtype_max_abs"),
        "divergence_cap_factor": first.get("divergence_cap_factor"),
        "divergence_cap": first.get("divergence_cap"),
        "time_window": first.get("time_window"),
        "hydra_merged_chunks": len(chunks),
        "phenotypes": [{"encoding": enc, "count": n, "complexity": clz(enc)} for enc, n in merged.items()],
    }


def main(
    tag: str,
    show_wildtype: bool,
    auto_hide_low_wildtype: bool,
    min_complexity: float | None,
    max_complexity: float | None,
    models: list[str] | None,
) -> Path:
    in_dir = CHUNK_DIR / tag
    if not in_dir.exists():
        raise FileNotFoundError(f"No Hydra chunk directory found: {in_dir}")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(in_dir.glob("*_chunk-*.json")):
        data = json.loads(path.read_text())
        grouped[data["model"]].append(data)

    all_data = []
    selected = [spec for spec in SPECS if models is None or spec.key in models]
    for spec in selected:
        if spec.key not in grouped:
            raise ValueError(f"Missing chunks for {spec.key} in {in_dir}")
        merged = merge_model(grouped[spec.key])
        (RESULTS / f"{spec.key}_complexity_frequency_{tag}_merged.json").write_text(json.dumps(merged))
        all_data.append(merged)

    if len(all_data) != len(SPECS):
        out = RESULTS / f"hydra_merge_{tag}_subset_complete.txt"
        out.write_text("\n".join(data["model"] for data in all_data))
        print(out)
        return out

    sample_values = {int(data["samples"]) for data in all_data}
    sample_text = sample_size_label(sample_values.pop()) if len(sample_values) == 1 else "mixed"
    out = PLOTS / f"CompFreq{sample_text}.png"
    plot_complexity_frequency(
        all_data,
        out,
        show_wildtype=show_wildtype,
        auto_hide_low_wildtype=auto_hide_low_wildtype,
        min_complexity=min_complexity,
        max_complexity=max_complexity,
    )
    print(out)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge WSBW Hydra chunk JSON files and make the final figure")
    parser.add_argument("--tag", default="hydra_run")
    parser.add_argument("--hide-wildtype", action="store_true")
    parser.add_argument("--show-low-wildtype", action="store_true")
    parser.add_argument("--min-complexity", type=float, default=None)
    parser.add_argument("--max-complexity", type=float, default=None)
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()
    main(
        tag=args.tag,
        show_wildtype=not args.hide_wildtype,
        auto_hide_low_wildtype=not args.show_low_wildtype,
        min_complexity=args.min_complexity,
        max_complexity=args.max_complexity,
        models=args.models,
    )

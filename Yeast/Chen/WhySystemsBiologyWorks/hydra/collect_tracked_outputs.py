#!/usr/bin/env python3
"""Mirror small, publication-facing outputs into Git-trackable folders.

The main analysis scripts write to ``results/`` and ``plots/`` because those
folders contain raw chunks, merged NPZ files, and other large files that should
not enter Git history. This script copies only small summaries and figures into

    figures/
    results_summaries/

so those curated outputs can be committed and pulled between Hydra and a laptop.
It does not delete or move the original files.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURE_EXTENSIONS = {".png", ".pdf", ".svg"}
SUMMARY_EXTENSIONS = {".json", ".csv", ".tsv"}


@dataclass
class MirroredFile:
    kind: str
    source: str
    destination: str
    size_bytes: int
    action: str


def path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def is_raw_or_large_summary(path: Path, root: Path, max_summary_bytes: int) -> tuple[bool, str]:
    """Return whether a would-be summary should be skipped."""

    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    rel_parts = set(rel.parts)
    name = path.name
    size = path.stat().st_size

    if size > max_summary_bytes:
        return True, f"larger than {max_summary_bytes} bytes"
    if "bruteforce_cloud" in rel_parts:
        return True, "raw brute-force chunk directory"
    if "hydra_chunks" in rel_parts:
        return True, "raw Hydra pipeline chunk directory"
    if "launchers" in rel_parts:
        return True, "launcher script directory"
    if "complexity_frequency" in name:
        return True, "giant/expandable complexity-frequency JSON"
    if "chunk-" in name or "_chunk-" in name:
        return True, "per-chunk output"
    if "_nnse_parallel_chain-" in name:
        return True, "per-chain NNSE output"
    if name.endswith(".ipynb_checkpoints"):
        return True, "not a summary"

    return False, ""


def is_smoke_or_local_output(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return False
    for part in rel.parts:
        if part.startswith("smoke") or part.startswith("local_"):
            return True
    return False


def destination_for(source: Path, root: Path, kind: str) -> Path:
    if path_is_relative_to(source, root / "plots"):
        rel = Path("plots") / source.relative_to(root / "plots")
    elif path_is_relative_to(source, root / "results"):
        rel = Path("results") / source.relative_to(root / "results")
    else:
        rel = source.relative_to(root)

    if kind == "figure":
        return root / "figures" / rel
    if kind == "summary":
        return root / "results_summaries" / rel
    raise ValueError(kind)


def copy_if_needed(source: Path, destination: Path, dry_run: bool) -> str:
    if dry_run:
        return "would_copy"

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        src_stat = source.stat()
        dst_stat = destination.stat()
        if dst_stat.st_size == src_stat.st_size and dst_stat.st_mtime >= src_stat.st_mtime:
            return "unchanged"

    shutil.copy2(source, destination)
    return "copied"


def iter_candidate_files(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for base_name in ("plots", "results"):
        base = root / base_name
        if base.exists():
            candidates.extend(path for path in base.rglob("*") if path.is_file())
    return sorted(candidates)


def mirror_outputs(
    root: Path,
    max_summary_mb: float,
    dry_run: bool,
    include_smoke: bool,
) -> list[MirroredFile]:
    max_summary_bytes = int(max_summary_mb * 1024 * 1024)
    mirrored: list[MirroredFile] = []

    for source in iter_candidate_files(root):
        if not include_smoke and is_smoke_or_local_output(source, root):
            continue
        suffix = source.suffix.lower()
        if suffix in FIGURE_EXTENSIONS:
            kind = "figure"
        elif suffix in SUMMARY_EXTENSIONS:
            skip, _reason = is_raw_or_large_summary(source, root, max_summary_bytes)
            if skip:
                continue
            kind = "summary"
        else:
            continue

        destination = destination_for(source, root, kind)
        action = copy_if_needed(source, destination, dry_run)
        mirrored.append(
            MirroredFile(
                kind=kind,
                source=str(source.relative_to(root)),
                destination=str(destination.relative_to(root)),
                size_bytes=source.stat().st_size,
                action=action,
            )
        )

    return mirrored


def write_manifest(root: Path, mirrored: list[MirroredFile], dry_run: bool) -> None:
    if dry_run:
        return

    out_dir = root / "results_summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(item) for item in mirrored]

    with (out_dir / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["kind", "source", "destination", "size_bytes", "action"])
        writer.writeheader()
        writer.writerows(rows)

    (out_dir / "manifest.json").write_text(json.dumps(rows, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="WhySystemsBiologyWorks root directory")
    parser.add_argument(
        "--max-summary-mb",
        type=float,
        default=5.0,
        help="Maximum JSON/CSV/TSV size to mirror into results_summaries",
    )
    parser.add_argument("--include-smoke", action="store_true", help="Also mirror smoke/local test outputs")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be mirrored without copying files")
    args = parser.parse_args()

    root = args.root.resolve()
    mirrored = mirror_outputs(root, args.max_summary_mb, args.dry_run, args.include_smoke)
    write_manifest(root, mirrored, args.dry_run)

    by_kind: dict[str, int] = {}
    by_action: dict[str, int] = {}
    for item in mirrored:
        by_kind[item.kind] = by_kind.get(item.kind, 0) + 1
        by_action[item.action] = by_action.get(item.action, 0) + 1

    print(f"Mirrored {len(mirrored)} files from {root}")
    print("By kind:", by_kind)
    print("By action:", by_action)
    if args.dry_run:
        for item in mirrored[:50]:
            print(f"{item.kind}: {item.source} -> {item.destination}")
        if len(mirrored) > 50:
            print(f"... {len(mirrored) - 50} more")
    else:
        print(f"Manifest: {root / 'results_summaries' / 'manifest.csv'}")


if __name__ == "__main__":
    main()

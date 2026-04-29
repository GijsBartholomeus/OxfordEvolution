from __future__ import annotations

import argparse

from wsbw_pipeline import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serial Chico-style complexity-frequency pipeline")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hide-wildtype", action="store_true")
    parser.add_argument("--show-low-wildtype", action="store_true")
    parser.add_argument("--min-complexity", type=float, default=None)
    parser.add_argument("--max-complexity", type=float, default=None)
    args = parser.parse_args()
    main(
        samples=args.samples,
        seed=args.seed,
        show_wildtype=not args.hide_wildtype,
        auto_hide_low_wildtype=not args.show_low_wildtype,
        min_complexity=args.min_complexity,
        max_complexity=args.max_complexity,
    )

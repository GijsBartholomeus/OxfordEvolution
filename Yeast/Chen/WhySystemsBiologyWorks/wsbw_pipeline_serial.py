from __future__ import annotations

import argparse

from wsbw_pipeline import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serial Chico-style complexity-frequency pipeline")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    main(samples=args.samples, seed=args.seed)

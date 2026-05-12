#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source bioevo/bin/activate

python nnse_sloppy_angle_boxplot.py \
  --model tyson1991 \
  --bruteforce-tag tyson_bfc_1e9 \
  --neutral-threshold 0.05 \
  --n-neutral 2000 \
  --n-random 2000 \
  --pair-samples 200000 \
  --k-sloppy 3 \
  --seed 42 \
  --tag tyson_bfc_1e9_f005_sloppy_angle_n2000_rand2000_k3

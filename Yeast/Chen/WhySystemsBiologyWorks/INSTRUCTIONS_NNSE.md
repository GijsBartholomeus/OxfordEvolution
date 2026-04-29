# NNSE Neutral-Set Sampling Instructions

Run from this folder:

```bash
cd /path/to/OxfordEvolution/code/Yeast/Chen/WhySystemsBiologyWorks
mamba activate bioevo
```

Quick smoke test for Chen:

```bash
MPLCONFIGDIR="$PWD/.mplconfig" python wsbw_nnse.py --model chen2004 --steps 100 --n-bins 20 --seed 42
```

Longer Chen run:

```bash
MPLCONFIGDIR="$PWD/.mplconfig" python wsbw_nnse.py --model chen2004 --steps 6000 --n-bins 50 --seed 42
```

Run all six models sequentially:

```bash
MPLCONFIGDIR="$PWD/.mplconfig" python wsbw_nnse.py --model all --steps 6000 --n-bins 50 --seed 42
```

Available model keys:

```text
chen2004
kholodenko2000
leloup1999
locke2005
ueda2001
vilar2002
```

Outputs are written to:

```text
results/nnse/
```

Each run produces:

```text
*_nnse_*.npz   neutral-set coordinates and diagnostics
*_nnse_*.json  run summary and configuration
```

The `.npz` contains:

```text
neutral_points
neutral_objective_values
final_population
final_objective_values
p0
parameter_names
bin_thresholds
volume_ratios
swap_count
opportunity_count
best_history
reference_time
reference_signal
```

The later accessibility analysis should use `neutral_points` as the NNSE sample
cloud. If `neutral_points` is empty or very small, rerun with more steps, more
bins, or a less strict `--neutral-threshold`.

## Inspecting and analyzing a merged neutral set

After merging parallel chains, open:

```text
NNSE/NeutralSetSanityCheck.ipynb
```

By default it selects the largest available merged neutral-set `.npz` under
`results/nnse_parallel/`. Set `NPZ_PATH` in the first code cell if you want to
inspect a specific file.

For the first accessibility analysis, open:

```text
NNSE/NeutralSetAccessibility.ipynb
```

This computes distances from random parameter starts to the NNSE cloud and
compares them with compact ball/box, covariance-ellipsoid, shuffled-marginal,
and synthetic-tube null geometries. The same analysis can be run headlessly:

```bash
MPLCONFIGDIR="$PWD/.mplconfig" python nnse_accessibility.py \
  --model chen2004 \
  --n-random 10000 \
  --max-cloud 50000
```

The output is saved in:

```text
results/nnse_accessibility/
```

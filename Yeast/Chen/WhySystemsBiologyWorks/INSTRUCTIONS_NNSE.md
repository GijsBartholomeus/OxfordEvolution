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
tyson1991
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

## Tyson 1991 on Hydra

Tyson uses the same NNSE code paths as Chen. The model key is `tyson1991`.

```bash
cd ~/OxfordEvolution/Yeast/Chen/WhySystemsBiologyWorks
git pull

bash hydra/submit_nnse_batch_chunks.sh tyson_nnse_batch_1e6 tyson1991 25 40000 16 2 long
```

After all 25 batch chunks finish:

```bash
python hydra/wsbw_merge_nnse_batch_chunks.py --tag tyson_nnse_batch_1e6 --model tyson1991
TYSON_INIT_NPZ=$(ls -t results/nnse_batch_init/tyson_nnse_batch_1e6/tyson1991_nnse_batch_init_merged_N=*.npz | head -n 1)
echo "$TYSON_INIT_NPZ"
```

Then start the parallel NNSE chains:

```bash
bash hydra/submit_nnse_parallel_chains.sh tyson_nnse_parallel_25chains_100k_thr15 \
  "$TYSON_INIT_NPZ" \
  25 100000 16 2 long tyson1991
```

After the chain jobs finish:

```bash
python hydra/wsbw_merge_nnse_parallel_chains.py --tag tyson_nnse_parallel_25chains_100k_thr15 --model tyson1991
python nnse_accessibility.py --model tyson1991 --n-random 10000 --max-cloud 50000
```

For notebook inspection, use `NNSE/TysonNeutralSetSanityCheck.ipynb` and
`NNSE/TysonNeutralSetAccessibility.ipynb`.

For the Tyson sloppy-direction geometry analysis, first copy a merged Tyson
neutral-set `.npz` back to this machine, then open:

```text
NNSE/TysonSloppySubspaceGeometry.ipynb
```

This notebook samples neutral-set points, computes the Tyson sensitivity
Hessian at each point, compares the `k` sloppiest eigenspaces by principal
angles, and checks whether the local neutral-set tangent directions align with
the local sloppy subspace.

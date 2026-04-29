# WhySystemsBiologyWorks Oscillatory Subset Pipeline

This folder contains the first modular pipeline for reproducing Chico-style
complexity-frequency plots on the oscillatory subset of the Gutenkunst/Chico
models.

Quick command references:

- Complexity-frequency runs: [INSTRUCTIONS_PIPELINE.md](INSTRUCTIONS_PIPELINE.md)
- NNSE neutral-set sampling: [INSTRUCTIONS_NNSE.md](INSTRUCTIONS_NNSE.md)
- Hydra cluster runs: [hydra/README.md](hydra/README.md)

## Current subset

- Chen 2004, output `CLB2`
- Kholodenko 2000, output `MKK_PP`
- Leloup 1999, output `Cn`
- Locke 2005, output `cXn`
- Ueda 2001, output `CCc`
- Vilar 2002, output `C`

## Parameter handling

The pipeline starts from the Gutenkunst `hessian_keys.dat` free-parameter lists
where possible. Some SBML models encode kinetic constants as local reaction
parameters, which RoadRunner cannot set by ID. For these models the pipeline
promotes only the Gutenkunst free local parameters to global SBML parameters and
removes the corresponding local definitions. It does not promote zero-valued or
unlisted local constants.

Chen is handled like the existing Chico reproduction notebook: all directly
settable nonzero kinetic global parameters are varied, while assignment-derived
quantities, total pools/switch-like parameters, and non-settable expressions are
excluded.

The full audit is written to `results/parameter_audit.json`.

## Time windows and divergence

Each model is simulated over a trough-to-trough phenotype window containing
roughly four wildtype peaks of the selected output variable. This avoids the
startup/transient segment that made several time-zero windows look like five or
more oscillations. Ueda remains the least clean oscillator in this subset, but
the same trough-to-trough rule is used there for consistency.

The divergence rejection rule is relative to each model's wildtype output scale:
sampled genotypes are rejected when the selected output exceeds
`100 * wildtype_max_abs`. This mirrors the spirit of the manually chosen Chen
`DIVERGENCE_THRESHOLD = 250`, without imposing one absolute cap across models
whose output scales differ by orders of magnitude.

## Running serial or parallel

See [INSTRUCTIONS_PIPELINE.md](INSTRUCTIONS_PIPELINE.md) for paste-ready
commands.

Serial reference run:

```bash
MPLCONFIGDIR="$PWD/.mplconfig" python wsbw_pipeline_serial.py --samples 1000 --seed 42
```

Parallel run for larger sampling:

```bash
MPLCONFIGDIR="$PWD/.mplconfig" python wsbw_pipeline_parallel.py --samples 10000 --seed 42 --workers 8
```

If `--workers` is omitted, the parallel runner uses `cpu_count - 1`. The worker
functions live in `wsbw_pipeline_parallel.py`, rather than in a notebook, so the
same command should work on macOS and Linux. On Windows it may also work, but it
has not been tested here.

For very large runs:

```bash
MPLCONFIGDIR="$PWD/.mplconfig" python wsbw_pipeline_parallel.py --samples 100000 --seed 42 --workers 12
```

On Hydra, use `hydra/submit_hydra_chunks.sh` to submit many independent
multiprocessing jobs through `addqueue`, then combine them with
`hydra/wsbw_merge_hydra_chunks.py`. This uses Hydra in the way it is designed
to be used: many normal Python jobs in parallel across compute nodes, with each
job using the cores reserved on its node. Keep large runs on compute nodes, not
on the login node. Set `WSBW_PAPER_DIR` to the local manuscript checkout if you
want the largest newly generated no-grid plot to update `Figures/FreqComp.png`
automatically.

Output:

- `plots/oscillatory_subset_complexity_frequency.png`
- `plots/oscillatory_subset_complexity_frequency_trough_windows_parallel.png` for the parallel runner
- `results/*_complexity_frequency.json`

After a large run, regenerate representative traces and the wildtype check plot:

```bash
MPLCONFIGDIR="$PWD/.mplconfig" python plot_complexity_representatives.py
MPLCONFIGDIR="$PWD/.mplconfig" python plot_wildtype_traces.py
```

## NNSE Neutral-Set Sampling

See [INSTRUCTIONS_NNSE.md](INSTRUCTIONS_NNSE.md) for paste-ready commands and
output-file details.

The starter NNSE runner is `wsbw_nnse.py`. It ports the mutation/permutation
logic from the earlier Chen/Tyson NNSE notebooks into a regular Python script
and saves neutral-set coordinates as portable `.npz` files.

Quick Chen smoke test:

```bash
MPLCONFIGDIR="$PWD/.mplconfig" python wsbw_nnse.py --model chen2004 --steps 100 --n-bins 20 --seed 42
```

Longer Chen run:

```bash
MPLCONFIGDIR="$PWD/.mplconfig" python wsbw_nnse.py --model chen2004 --steps 6000 --n-bins 50 --seed 42
```

Other model keys are `kholodenko2000`, `leloup1999`, `locke2005`, `ueda2001`,
and `vilar2002`. Outputs are written to `results/nnse/`:

- `*_nnse_*.npz`: neutral coordinates, final population, objective values, bins,
  swap counts, and reference trace.
- `*_nnse_*.json`: run summary and configuration.

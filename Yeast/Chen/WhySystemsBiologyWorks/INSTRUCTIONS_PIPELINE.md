# Complexity-Frequency Pipeline Instructions

Run from this folder:

```bash
cd /path/to/OxfordEvolution/code/Yeast/Chen/WhySystemsBiologyWorks
mamba activate bioevo
```

Serial reference run:

```bash
MPLCONFIGDIR="$PWD/.mplconfig" python wsbw_pipeline_serial.py --samples 1000 --seed 42
```

Parallel run:

```bash
MPLCONFIGDIR="$PWD/.mplconfig" python wsbw_pipeline_parallel.py --samples 10000 --seed 42 --workers 8
```

Large run:

```bash
MPLCONFIGDIR="$PWD/.mplconfig" python wsbw_pipeline_parallel.py --samples 100000 --seed 42 --workers 8
```

Cluster/Hydra-style Slurm run:

```bash
mkdir -p logs
sbatch hydra/run_wsbw_hydra.slurm
```

On Oxford Physics Hydra, use the local `addqueue` wrapper. The important flag is
`-s`: it reserves one compute node allocation and starts one Python parent
process, which then uses the requested cores via multiprocessing.

```bash
WSBW_TAG=hydra_smoke WSBW_CHUNK_ID=0 WSBW_SAMPLES_PER_MODEL=1000 \
  addqueue -q short -s -c "smoke test" -m 16 -n 1x8 ./hydra/run_wsbw_hydra.slurm
```

For a larger multi-node run, submit many independent chunks. For example,
`10` chunks with `10000` samples per model gives `10^5` total samples per model:

```bash
bash hydra/submit_hydra_chunks.sh hydra_1e5 10 10000 16 32 long
```

After every chunk has finished, merge the chunks and make the final figure:

```bash
python hydra/wsbw_merge_hydra_chunks.py --tag hydra_1e5
```

Monitor it with:

```bash
squeue -u "$USER"
tail -f logs/wsbw_JOBID.out
```

Do not run `10^5` or larger jobs directly on a login node. Use the Slurm
script, or first request an interactive compute session and run a small smoke
test there.

By default, the complexity-frequency plot includes the red wildtype marker and
uses the full observed complexity range. To reproduce the cropped version
without the wildtype marker:

```bash
MPLCONFIGDIR="$PWD/.mplconfig" python wsbw_pipeline_parallel.py --samples 100000 --seed 42 --workers 8 --hide-wildtype --min-complexity 15 --max-complexity 50
```

Then regenerate representative and wildtype plots:

```bash
MPLCONFIGDIR="$PWD/.mplconfig" python plot_complexity_representatives.py
MPLCONFIGDIR="$PWD/.mplconfig" python plot_wildtype_traces.py
```

Main outputs:

```text
results/*_complexity_frequency.json
plots/oscillatory_subset_complexity_frequency_trough_windows_parallel_N=1e4.png
plots/oscillatory_subset_low_complexity_representatives_trough_windows_N=1e4.png
plots/oscillatory_subset_high_complexity_representatives_trough_windows_N=1e4.png
plots/oscillatory_subset_wildtype_trough_windows.png
```

The exact `N=...` tag follows the run size, for example `N=1e3`, `N=1e4`,
`N=1e5`, or `N=1e6`.

## Manuscript figure handoff

When a complete seven-model run creates a new no-grid complexity-frequency
figure, the pipeline checks the `plots/` folder. If the new plot has the
largest `N=...` sample size currently present, it is copied to
`Figures/FreqComp.png` in the manuscript folder. If `FreqComp.png` is missing,
the pipeline copies the new plot there regardless of `N`.

On a different computer or on Hydra, point the pipeline to the local manuscript
checkout before running:

```bash
export WSBW_PAPER_DIR=/path/to/PNAS-WhySystemsBiologyWorks-git
```

`WSBW_PAPER_DIR` should be the folder containing `main.tex` and `Figures/`.
If this variable is not set, the script tries the usual local paper folders on
this Mac.

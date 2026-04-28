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

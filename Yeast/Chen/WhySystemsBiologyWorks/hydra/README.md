# Hydra Running Notes

Hydra is a Linux Slurm cluster. It also has graphical access through x2go/RDP
and a graphical/text job viewer `q`, but the safest workflow for this project is
SSH plus Git:

```bash
ssh yourusername@hydra.physics.ox.ac.uk
git clone git@github.com:GijsBartholomeus/PNAS-WhySystemsBiologyWorks.git
git clone <your-analysis-repo-url>
cd <your-analysis-repo>/Yeast/Chen/WhySystemsBiologyWorks
```

Use the home directory for code and small outputs. Use `/mnt/extraspace` only
for heavy temporary output; it is not backed up.

Create a local Python environment for the pipeline:

```bash
python3 -m venv bioevo
source bioevo/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-pipeline.txt
python -c "import numpy, matplotlib, scipy, libsbml, roadrunner; print('pipeline imports ok')"
```

Run a short GP-map smoke test first:

```bash
WSBW_TAG=hydra_smoke WSBW_CHUNK_ID=0 WSBW_SAMPLES_PER_MODEL=1000 \
  addqueue -q long -s -c "smoke test" -m 2 -n 1x8 ./hydra/run_wsbw_hydra.slurm
```

Submit a large multi-node run:

```bash
bash hydra/submit_hydra_chunks.sh hydra_1e5 10 10000 16 2 long
```

Check jobs:

```bash
q -t
showoutput <jobNumber>
scancel <jobNumber>
```

Merge after all chunks are done:

```bash
python hydra/wsbw_merge_hydra_chunks.py --tag hydra_1e5
```

## Curated outputs for Git

Analysis scripts write raw outputs into `results/` and `plots/`. Those folders
may contain raw chunks, merged `.npz` files, and very large JSON files, so they
are intentionally ignored by Git. After producing a useful result, mirror the
small paper-facing outputs into the tracked folders:

```bash
python hydra/collect_tracked_outputs.py
```

This copies figures into:

```text
figures/
```

and small JSON/CSV/TSV summaries into:

```text
results_summaries/
```

It skips raw chunks, per-chain files, `.npz`/`.npy` arrays, launcher scripts,
smoke/local test outputs, and giant complexity-frequency JSONs. It does not move
or delete the originals. Commit the curated mirror, not the raw result store:

```bash
git add figures results_summaries hydra/collect_tracked_outputs.py
git commit -m "Collect curated analysis outputs"
git push
```

Use `--dry-run` to inspect what would be mirrored, and `--include-smoke` only if
you deliberately want local/smoke-test artifacts in the mirror.

## NNSE batch initialisation benchmark

Before running a full NNSE chain, test whether random parallel screening can
fill a useful part of the NNSE threshold ladder. This creates an `.npz` with a
candidate-filled initial population and a `.json` summary with timing,
finite/success rates, bin counts, and the number of thresholds filled.

Small test:

```bash
WSBW_TAG=chen_nnse_batch_smoke WSBW_NNSE_CANDIDATES=1000 \
  addqueue -q long -s -c "chen nnse batch smoke" -m 2 -n 1x8 ./hydra/run_nnse_batch_init.sh
```

Larger screening run:

```bash
WSBW_TAG=chen_nnse_batch_1e5 WSBW_NNSE_CANDIDATES=100000 \
  addqueue -q long -s -c "chen nnse batch 1e5" -m 2 -n 1x16 ./hydra/run_nnse_batch_init.sh
```

Outputs are written to:

```bash
results/nnse_batch_init/<tag>/
```

Inspect the JSON summary first. The key fields are `elapsed_seconds`,
`candidates_per_second`, `finite_fraction`, `placed_fraction`,
`filled_thresholds`, and `bin_counts`.

To use many Hydra nodes, run the chunked version. This is the preferred route
for `1e6` or larger NNSE random screening:

```bash
bash hydra/submit_nnse_batch_chunks.sh chen_nnse_batch_1e6_chunked chen2004 25 40000 16 2 long
```

This submits 25 jobs, each using 16 cores and screening 40,000 Chen candidates,
for 1,000,000 total candidates. When all jobs finish, the expected chunk file
count is 25:

```bash
find results/nnse_batch_init/chen_nnse_batch_1e6_chunked -name '*chunk-*.json' | wc -l
```

Merge the chunks into one NNSE initialisation file:

```bash
python hydra/wsbw_merge_nnse_batch_chunks.py --tag chen_nnse_batch_1e6_chunked --model chen2004
```

The merged `.npz` contains the same `initial_population`,
`initial_objective_values`, `parameter_names`, and `bin_thresholds` arrays as
the single-job version, but pooled across all chunk jobs.

## Parallel NNSE chains from a merged initial population

After merging an NNSE batch initialisation run, start one or more NNSE chains
from the merged `.npz`. Each chain evaluates mutation proposals in parallel,
then applies the NNSE accept/swap logic serially inside the coordinator.
The current default neutral threshold for saved NNSE coordinates is `15.0`;
override `WSBW_NNSE_NEUTRAL_THRESHOLD` if you want a stricter or looser saved
set.

Short test with one chain:

```bash
WSBW_TAG=chen_nnse_parallel_smoke \
WSBW_NNSE_INIT_NPZ=results/nnse_batch_init/chen_nnse_batch_1e6_chunked/chen2004_nnse_batch_init_merged_N=1e6.npz \
WSBW_NNSE_STEPS=20 \
  addqueue -q long -s -c "chen nnse parallel smoke" -m 2 -n 1x8 ./hydra/run_nnse_parallel.sh
```

Larger multi-chain run:

```bash
bash hydra/submit_nnse_parallel_chains.sh chen_nnse_parallel_25chains \
  results/nnse_batch_init/chen_nnse_batch_1e6_chunked/chen2004_nnse_batch_init_merged_N=1e6.npz \
  25 6000 16 2 long
```

This submits 25 independent chains, each with 16 local workers. Merge the
neutral samples after all chains finish:

```bash
python hydra/wsbw_merge_nnse_parallel_chains.py --tag chen_nnse_parallel_25chains --model chen2004
```

Then either inspect the merged file in `NNSE/NeutralSetSanityCheck.ipynb` or
run the first accessibility/null-geometry analysis directly:

```bash
python nnse_accessibility.py --model chen2004 --n-random 10000 --max-cloud 50000
```

The same workflow works for Tyson 1991. The batch-initialisation submitter
takes the model key as its second argument, and the chain submitter takes the
model key as its final optional argument:

```bash
bash hydra/submit_nnse_batch_chunks.sh tyson_nnse_batch_1e6 tyson1991 25 40000 16 2 long
python hydra/wsbw_merge_nnse_batch_chunks.py --tag tyson_nnse_batch_1e6 --model tyson1991

TYSON_INIT_NPZ=$(ls -t results/nnse_batch_init/tyson_nnse_batch_1e6/tyson1991_nnse_batch_init_merged_N=*.npz | head -n 1)
bash hydra/submit_nnse_parallel_chains.sh tyson_nnse_parallel_25chains_100k_thr15 \
  "$TYSON_INIT_NPZ" \
  25 100000 16 2 long tyson1991

python hydra/wsbw_merge_nnse_parallel_chains.py --tag tyson_nnse_parallel_25chains_100k_thr15 --model tyson1991
python nnse_accessibility.py --model tyson1991 --n-random 10000 --max-cloud 50000
```

Set `WSBW_NNSE_REFILL_ATTEMPTS` to a small positive number if you want the
runner to try random refills for newly opened loose bins after swaps. Leave it
at the default `0` for the cleanest first benchmark from the merged initial
population.

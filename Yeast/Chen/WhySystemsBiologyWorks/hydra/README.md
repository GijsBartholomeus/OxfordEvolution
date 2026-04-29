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
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-pipeline.txt
python -c "import numpy, matplotlib, scipy, libsbml, roadrunner; print('pipeline imports ok')"
```

Run a short smoke test first:

```bash
WSBW_TAG=hydra_smoke WSBW_CHUNK_ID=0 WSBW_SAMPLES_PER_MODEL=1000 \
  addqueue -q short -s -c "smoke test" -m 16 -n 1x8 ./hydra/run_wsbw_hydra.slurm
```

Submit a large multi-node run:

```bash
bash hydra/submit_hydra_chunks.sh hydra_1e5 10 10000 16 32 long
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

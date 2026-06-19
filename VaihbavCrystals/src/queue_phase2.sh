#!/bin/bash
LOG="/home/gijs/Documents/VaihbavCrystals/run_parallel.log"
DSERIES_PID=155665

echo "$(date): queue_phase2 watcher started, waiting for PID $DSERIES_PID" | tee -a "$LOG"

while kill -0 $DSERIES_PID 2>/dev/null; do
    sleep 30
done

echo "$(date): PID $DSERIES_PID finished — starting phase-2 (scaled sweeps + normalized plots)" | tee -a "$LOG"
pkill -f hamming_mcmc 2>/dev/null
sleep 3

nohup python3 /home/gijs/Documents/VaihbavCrystals/src/run_dseries_phase2.py \
    >> "$LOG" 2>&1 &
echo "$(date): run_dseries_phase2.py started as PID $!" | tee -a "$LOG"

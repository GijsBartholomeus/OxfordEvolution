#!/bin/bash
# Wait for run_zipf_uniform_mohanty.py (PID 129624) to finish, then start d-series.
LOG="/home/gijs/Documents/VaihbavCrystals/run_parallel.log"
MOHANTY_PID=129624

echo "$(date): queue_dseries watcher started, waiting for PID $MOHANTY_PID" | tee -a "$LOG"

while kill -0 $MOHANTY_PID 2>/dev/null; do
    sleep 30
done

echo "$(date): PID $MOHANTY_PID finished — starting d-series runs" | tee -a "$LOG"
pkill -f hamming_mcmc 2>/dev/null
sleep 3

nohup python3 /home/gijs/Documents/VaihbavCrystals/src/run_dseries.py \
    >> "$LOG" 2>&1 &
echo "$(date): run_dseries.py started as PID $!" | tee -a "$LOG"

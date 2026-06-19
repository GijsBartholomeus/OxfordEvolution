#!/bin/bash
# Wait for geometric to finish, kill the current orchestrator, start Mohanty-sweep zipf+uniform.
LOG="/home/gijs/Documents/VaihbavCrystals/run_parallel.log"
GEO_PID=25016

echo "$(date): watcher started, monitoring for geometric completion (PID $GEO_PID)" >> "$LOG"

until grep -q "Wall time:" "$LOG"; do
    sleep 5
done

echo "$(date): geometric done — killing orchestrator and any stray workers" >> "$LOG"
kill $GEO_PID 2>/dev/null
pkill -f hamming_mcmc 2>/dev/null
sleep 5

nohup python3 /home/gijs/Documents/VaihbavCrystals/src/run_zipf_uniform_mohanty.py \
    >> "$LOG" 2>&1 &
echo "$(date): started zipf+uniform at Mohanty sweeps (100+50), PID $!" >> "$LOG"

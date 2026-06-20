#!/bin/bash
#SBATCH --partition=capacity
#SBATCH --job-name=varc_race_watcher
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=1G
#SBATCH --output=logs/varc_race_watcher_%A.out

PERF_JOBS="245702,245703"
CAP_JOBS="245704,245705"

echo "Race watcher started at $(date)"
echo "Performance jobs: $PERF_JOBS"
echo "Capacity jobs:    $CAP_JOBS"

while true; do
    perf_running=$(squeue -j $PERF_JOBS -t RUNNING -h 2>/dev/null | wc -l)
    cap_running=$(squeue -j $CAP_JOBS -t RUNNING -h 2>/dev/null | wc -l)

    if [ "$perf_running" -gt 0 ]; then
        echo "$(date): Performance jobs started — cancelling capacity jobs $CAP_JOBS"
        scancel $(echo $CAP_JOBS | tr ',' ' ')
        exit 0
    fi

    if [ "$cap_running" -gt 0 ]; then
        echo "$(date): Capacity jobs started — cancelling performance jobs $PERF_JOBS"
        scancel $(echo $PERF_JOBS | tr ',' ' ')
        exit 0
    fi

    # Exit if all jobs have disappeared (completed, failed, or already cancelled)
    remaining=$(squeue -j $PERF_JOBS,$CAP_JOBS -h 2>/dev/null | wc -l)
    if [ "$remaining" -eq 0 ]; then
        echo "$(date): All jobs gone, watcher exiting"
        exit 0
    fi

    sleep 60
done

#!/bin/bash
# Ralph Loop — Automated ML iteration via fresh Claude instances
#
# Each iteration gets a clean context window, reads the spec + plan,
# picks a task, implements it, tests it, and updates the plan.
#
# Usage:
#   bash ralph_loop.sh          # Run up to 50 iterations
#   bash ralph_loop.sh 10       # Run up to 10 iterations
#
# Monitor in another terminal:
#   tail -f invivo_results/ralph_loop_log.txt
#
# Check progress:
#   cat ralph_plan.md

set -uo pipefail

MAX_ITERS="${1:-50}"
LOG_DIR="invivo_results"
LOG_FILE="${LOG_DIR}/ralph_loop_log.txt"

mkdir -p "$LOG_DIR"

echo "========================================" | tee -a "$LOG_FILE"
echo "Ralph Loop started at $(date)" | tee -a "$LOG_FILE"
echo "Max iterations: $MAX_ITERS" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

for i in $(seq 1 "$MAX_ITERS"); do
    echo "" | tee -a "$LOG_FILE"
    echo "=== Iteration $i / $MAX_ITERS === $(date)" | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"

    # Run a fresh Claude instance with the ralph prompt
    # --dangerously-skip-permissions: required for headless -p mode
    claude -p "$(cat ralph_prompt.md)" \
        --dangerously-skip-permissions \
        --allowedTools "Bash,Read,Edit,Write,Glob,Grep" \
        2>&1 | tee -a "$LOG_FILE"
    EXIT_CODE=${PIPESTATUS[0]}

    if [ "$EXIT_CODE" -eq 0 ]; then
        echo "--- Iteration $i completed successfully at $(date) ---" | tee -a "$LOG_FILE"
    else
        echo "*** Iteration $i FAILED (exit code $EXIT_CODE) at $(date) ***" | tee -a "$LOG_FILE"
        echo "Continuing to next iteration..." | tee -a "$LOG_FILE"
    fi

    # Check for completion signal
    if tail -20 "$LOG_FILE" | grep -q "RALPH_COMPLETE"; then
        echo "" | tee -a "$LOG_FILE"
        echo "========================================" | tee -a "$LOG_FILE"
        echo "RALPH_COMPLETE at iteration $i — $(date)" | tee -a "$LOG_FILE"
        echo "========================================" | tee -a "$LOG_FILE"
        exit 0
    fi
done

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Ralph Loop exhausted $MAX_ITERS iterations without hitting all targets." | tee -a "$LOG_FILE"
echo "Check ralph_plan.md for progress." | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
exit 1

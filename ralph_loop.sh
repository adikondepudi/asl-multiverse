#!/bin/bash
# Ralph Loop — Automated ML iteration via fresh Claude instances
#
# Usage:
#   bash ralph_loop.sh          # Run up to 50 iterations
#   bash ralph_loop.sh 10       # Run up to 10 iterations
#
# Monitor in another terminal:
#   tail -f invivo_results/ralph_loop_log.txt

set -uo pipefail

MAX_ITERS="${1:-50}"
LOG_DIR="invivo_results"
LOG_FILE="${LOG_DIR}/ralph_loop_log.txt"

mkdir -p "$LOG_DIR"

# Ensure Ctrl+C kills claude and all children
cleanup() {
    echo "" | tee -a "$LOG_FILE"
    echo "*** Ralph Loop interrupted by user at $(date) ***" | tee -a "$LOG_FILE"
    kill 0 2>/dev/null
    exit 130
}
trap cleanup INT TERM

echo "========================================" | tee -a "$LOG_FILE"
echo "Ralph Loop started at $(date)" | tee -a "$LOG_FILE"
echo "Max iterations: $MAX_ITERS" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

for i in $(seq 1 "$MAX_ITERS"); do
    echo "" | tee -a "$LOG_FILE"
    echo "=== Iteration $i / $MAX_ITERS === $(date)" | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"

    # Write output to log file; user monitors via: tail -f invivo_results/ralph_loop_log.txt
    # Running without pipe so Ctrl+C propagates correctly
    claude -p "$(cat ralph_prompt.md)" \
        --dangerously-skip-permissions \
        --allowedTools "Bash,Read,Edit,Write,Glob,Grep" \
        >> "$LOG_FILE" 2>&1 &
    CLAUDE_PID=$!
    echo "Claude PID: $CLAUDE_PID (monitor with: tail -f $LOG_FILE)"

    wait "$CLAUDE_PID"
    EXIT_CODE=$?

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
echo "========================================" | tee -a "$LOG_FILE"
exit 1

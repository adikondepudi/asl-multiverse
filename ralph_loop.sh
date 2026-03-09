#!/bin/bash
# Ralph Loop — Automated ML iteration via fresh Claude instances
#
# Architecture: Claude implements, bash evaluates.
# Claude never sees harness results, so it can't get stuck retrying.
#
# Usage:
#   bash ralph_loop.sh          # Run up to 50 iterations
#   bash ralph_loop.sh 10       # Run up to 10 iterations
#
# Monitor: tail -f invivo_results/ralph_loop_log.txt

set -uo pipefail

MAX_ITERS="${1:-50}"
LOG_DIR="invivo_results"
LOG_FILE="${LOG_DIR}/ralph_loop_log.txt"
RESULTS_FILE="${LOG_DIR}/latest_results.json"
SPEC_FILE="ralph_spec.md"

mkdir -p "$LOG_DIR"

# Ensure Ctrl+C kills everything
cleanup() {
    echo "" | tee -a "$LOG_FILE"
    echo "*** Ralph Loop interrupted by user at $(date) ***" | tee -a "$LOG_FILE"
    kill 0 2>/dev/null
    exit 130
}
trap cleanup INT TERM

log() {
    echo "$1" | tee -a "$LOG_FILE"
}

# ─── Evaluate results against current best ───
# Returns 0 if improved, 1 if worse
evaluate_results() {
    python3 - "$RESULTS_FILE" "$SPEC_FILE" <<'PYEOF'
import json, sys, re

results_file = sys.argv[1]
spec_file = sys.argv[2]

with open(results_file) as f:
    results = json.load(f)

# Extract current best from spec file
with open(spec_file) as f:
    spec = f.read()

# Parse current best CBF wins from spec
# Look for pattern like "CBF Win Rate: 72.6 / 77.9 / 79.0"
cbf_match = re.search(r'CBF Win Rate:\s*([\d.]+)\s*/\s*([\d.]+)\s*/\s*([\d.]+)', spec)
if not cbf_match:
    print("FAIL: Could not parse current best from spec")
    sys.exit(1)

best_cbf = [float(cbf_match.group(i)) for i in range(1, 4)]

# Get new results
new_cbf = [
    results['synthetic']['3']['cbf_win_rate'],
    results['synthetic']['10']['cbf_win_rate'],
    results['synthetic']['25']['cbf_win_rate'],
]

new_cov = results['checks'].get('nn_cov_avg', 999)
new_smooth = results['checks'].get('nn_smooth_avg', 999)
ls_cov = results['checks'].get('ls_cov_avg', 1)
ls_smooth = results['checks'].get('ls_smooth_avg', 1)

cov_ratio = new_cov / ls_cov if ls_cov > 0 else 999
smooth_ratio = new_smooth / ls_smooth if ls_smooth > 0 else 999

# Parse current best ratios from spec
cov_match = re.search(r'CoV Ratio:\s*([\d.]+)', spec)
smooth_match = re.search(r'Smooth Ratio:\s*([\d.]+)', spec)
best_cov_ratio = float(cov_match.group(1)) if cov_match else 1.0
best_smooth_ratio = float(smooth_match.group(1)) if smooth_match else 1.0

# Check improvement criteria:
# Any CBF win increased AND none dropped more than 3%
cbf_any_improved = any(new > old for new, old in zip(new_cbf, best_cbf))
cbf_none_dropped = all(new >= old - 3.0 for new, old in zip(new_cbf, best_cbf))
cbf_improved = cbf_any_improved and cbf_none_dropped

cov_improved = cov_ratio < best_cov_ratio
smooth_improved = smooth_ratio < best_smooth_ratio

improved = cbf_improved or cov_improved or smooth_improved

print(f"CBF wins: {new_cbf[0]:.1f}/{new_cbf[1]:.1f}/{new_cbf[2]:.1f} (best: {best_cbf[0]:.1f}/{best_cbf[1]:.1f}/{best_cbf[2]:.1f})")
print(f"CoV ratio: {cov_ratio:.2f} (best: {best_cov_ratio:.2f})")
print(f"Smooth ratio: {smooth_ratio:.2f} (best: {best_smooth_ratio:.2f})")
print(f"CBF improved: {cbf_improved}, CoV improved: {cov_improved}, Smooth improved: {smooth_improved}")
print(f"VERDICT: {'PASS' if improved else 'FAIL'}")

sys.exit(0 if improved else 1)
PYEOF
}

log "========================================"
log "Ralph Loop started at $(date)"
log "Max iterations: $MAX_ITERS"
log "========================================"

for i in $(seq 1 "$MAX_ITERS"); do
    log ""
    log "=== Iteration $i / $MAX_ITERS === $(date)"
    log "----------------------------------------"

    # ─── Phase 1: Claude picks task and implements ───
    log "[Phase 1] Claude implementing..."
    claude -p "$(cat ralph_prompt.md)" \
        --dangerously-skip-permissions \
        --allowedTools "Bash,Read,Edit,Write,Glob,Grep" \
        >> "$LOG_FILE" 2>&1 &
    CLAUDE_PID=$!
    log "Claude PID: $CLAUDE_PID"
    wait "$CLAUDE_PID" 2>/dev/null
    CLAUDE_EXIT=$?
    log "[Phase 1] Claude exited with code $CLAUDE_EXIT"

    # ─── Phase 2: Bash runs the harness ───
    log "[Phase 2] Running harness..."
    python3 ralph_harness.py --device mps >> "$LOG_FILE" 2>&1
    HARNESS_EXIT=$?

    if [ "$HARNESS_EXIT" -ne 0 ]; then
        log "[Phase 2] Harness CRASHED (exit code $HARNESS_EXIT)"
        # Revert and mark fail
        git checkout -- config/ ralph_harness.py models/ simulation/ 2>/dev/null
        log "*** Iteration $i: CRASH — reverted, moving on ***"
        continue
    fi

    log "[Phase 2] Harness completed"

    # ─── Phase 3: Bash evaluates results ───
    log "[Phase 3] Evaluating results..."
    EVAL_OUTPUT=$(evaluate_results 2>&1)
    EVAL_EXIT=$?
    log "$EVAL_OUTPUT"

    if [ "$EVAL_EXIT" -eq 0 ]; then
        # ─── PASS: Let Claude commit and update plan ───
        log "[Phase 3] IMPROVED — letting Claude commit..."
        claude -p "The harness results are in. The experiment PASSED (metrics improved). Read invivo_results/latest_results.json, update ralph_plan.md (mark task [x], add iteration log row), update ralph_spec.md with new best numbers, then commit: git add -A && git commit -m 'ralph: [task_id] — [description]'. Then exit." \
            --dangerously-skip-permissions \
            --allowedTools "Bash,Read,Edit,Write,Glob,Grep" \
            >> "$LOG_FILE" 2>&1 &
        wait $! 2>/dev/null
        log "--- Iteration $i: PASS at $(date) ---"
    else
        # ─── FAIL: Revert code, let Claude update plan ───
        log "[Phase 3] NOT IMPROVED — reverting code..."
        git checkout -- config/ ralph_harness.py models/ simulation/ 2>/dev/null
        claude -p "The harness results are in. The experiment FAILED (metrics did not improve). Results: $EVAL_OUTPUT. Read ralph_plan.md, mark the first unchecked [ ] task as [FAIL] with the reason from the results above, add an iteration log row, then commit: git add ralph_plan.md && git commit -m 'ralph: [task_id] FAIL — [reason]'. Then exit." \
            --dangerously-skip-permissions \
            --allowedTools "Bash,Read,Edit,Write,Glob,Grep" \
            >> "$LOG_FILE" 2>&1 &
        wait $! 2>/dev/null
        log "*** Iteration $i: FAIL at $(date) ***"
    fi

    # Check for completion
    if [ -f "$RESULTS_FILE" ]; then
        OVERALL=$(python3 -c "import json; print(json.load(open('$RESULTS_FILE'))['checks'].get('overall', False))" 2>/dev/null)
        if [ "$OVERALL" = "True" ]; then
            log ""
            log "========================================"
            log "RALPH_COMPLETE at iteration $i — $(date)"
            log "========================================"
            exit 0
        fi
    fi
done

log ""
log "========================================"
log "Ralph Loop exhausted $MAX_ITERS iterations."
log "========================================"
exit 1

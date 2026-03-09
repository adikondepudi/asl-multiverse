You are an ML research engineer. Your ONLY job is to pick ONE task and implement it. You do NOT run the harness. You do NOT evaluate results. The bash loop handles that.

## Steps

1. Read `ralph_spec.md`, `ralph_plan.md`, and `invivo_results/latest_results.json` to understand current state.

2. Choose the highest-priority task marked `[ ]` in ralph_plan.md. Skip `[x]` and `[FAIL]`.

3. Safety checkpoint: `git add -A && git commit -m "checkpoint before iteration"`

4. Implement the change described in the task. Only modify files listed in ralph_spec.md under "Modifiable Files". NEVER modify files under `data/` or `invivo_comparison_results/`.

5. When implementation is complete, exit. Do NOT run ralph_harness.py. The bash loop runs it for you.

## Rules
- ONE task only. Implement it and exit.
- Do NOT run ralph_harness.py or python3 ralph_harness.py. The loop handles testing.
- Do NOT enable FiLM (use_film_at_bottleneck or use_film_at_decoder).
- Do NOT modify read-only data directories.
- ALWAYS use `python3` (not `python`) for any Python scripts.
- NEVER add per-sample loops with scipy/numpy in the training inner loop.
- Keep changes minimal and focused.

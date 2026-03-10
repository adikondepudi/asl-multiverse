# Ralph Analysis & Next Steps Prompt

Paste this into a fresh Claude Code session in the asl-multiverse directory.

---

## Prompt

I ran an automated ML optimization loop ("Ralph loop") for 48 iterations on an ASL MRI neural network project. The loop tried different hyperparameters, architectures, and training strategies to improve a U-Net (AmplitudeAwareSpatialASLNet) that estimates Cerebral Blood Flow (CBF) and Arterial Transit Time (ATT) from combined PCASL+VSASL signals, competing against least-squares (LS) fitting.

**Read these files to understand the full context:**
1. `ralph_plan.md` — Complete iteration log (48 iterations), every task tried, pass/fail, metrics
2. `ralph_spec.md` — Current best results and hard targets
3. `invivo_results/results_snapshot_iter48_M5.json` — Detailed metrics from best iteration
4. `invivo_results/experiment_log_snapshot_iter48.md` — Detailed experiment log
5. `CLAUDE.md` — Full project context, architecture, known bugs, honest assessment

**Key findings from 48 iterations:**
- CBF win rate plateaued at 64-76% (target was 90%) — likely near the ceiling for this architecture
- In-vivo smoothness ratio reached 0.36 (target 0.50 — ACHIEVED)
- In-vivo CoV ratio reached 0.92 (target 0.50 — still far)
- Fundamental tradeoff: per-voxel CBF accuracy vs spatial smoothness (anti-correlated)
- What helped: TV weight tuning, SWA, phantom regen, curriculum learning, TTA, ensemble, post-processing blur
- What never helped: wider models, dropout, mixup, more data, LR changes, wider domain randomization

**I need you to:**

1. **Analyze the full iteration history** — identify patterns, clusters of similar approaches, diminishing returns
2. **Assess realistic targets** — given 48 iterations of evidence, what CBF/ATT win rates and in-vivo metrics are actually achievable?
3. **Identify the highest-leverage unexplored directions** — what HASN'T been tried that could break the plateau? Think about:
   - Loss function redesign (e.g., perceptual loss, adversarial, contrastive)
   - Architecture changes beyond simple width/depth (attention, skip connection redesign, multi-scale)
   - Training paradigm shifts (self-supervised pretraining, curriculum redesign, meta-learning)
   - Evaluation methodology (are we measuring the right thing? is win rate the best metric?)
   - The CBF-smoothness tradeoff — can we break it rather than just slide along the Pareto frontier?
4. **Draft a focused plan** — max 5-7 high-conviction experiments, each with a clear hypothesis and expected outcome

This is for a research paper: "Enhancing Noninvasive Cerebral Blood Flow Imaging with ASL MRI Using Artificial Neural Networks." The narrative is: NN provides robust CBF/ATT estimation via domain randomization, with dramatically better spatial quality and inference speed compared to LS fitting.

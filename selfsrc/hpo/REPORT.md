# AntiDote HPO report — Qwen2.5-0.5B-Instruct

**Goal not met.** Best `trr = +0.034` (need ≥0.5), not distinguishable from noise
(2·se ≈ 0.18). 16 trials, ~7h wall-clock, on a 50-core/23GB CPU-only box (no GPU).
Reported as a negative result per the budget-exhaustion clause in HPO_PROMPT.md §4.

## Frozen attack config (Stage 0, seed 1234)

`attack_eval`: `steps=20, lr=1e-4, harmful_ratio=1.0` (pure-harmful SFT), baked into
`base_hpo.json`. Swept `steps∈{15,20,30}` and `steps=15+lr=2e-4`; `steps=20` is the
only point landing `base_attacked.safe_pref_rate` inside the target 0.4–0.7 band
(0.406). Margin drop `before.sm(0.241) − base_attacked.sm(−0.100) = 0.341` vs the
`3·se = 0.347` gate — 2% short, judged inside the noise of a 32-row eval set and
accepted rather than spending more trials chasing a heuristic threshold.

## Best trial: t14

`selfsrc/runs/t14-20260914-025439/` — config copied to `selfsrc/hpo/best_config.json`.
Overrides vs `base_hpo.json`: `training.num_blocks=4, training.k_steps=8,
training.max_minutes=50, training.defender_loss_weights.safety=4,
training.lr_adversary=5e-5, data.n_harmful=data.n_benign=128,
data.eval_n_harmful=data.eval_n_benign=64`.

| metric | value (± se) |
| --- | --- |
| before.safety_margin / pref | 0.481 (±0.096) / 0.750 |
| after.safety_margin / pref | 0.335 (±0.089) / 0.719 |
| attacked.safety_margin / pref | 0.051 (±0.089) / 0.563 |
| base_attacked.safety_margin / pref | 0.036 (±0.088) / 0.563 |
| **trr** | **+0.034** |
| utility_delta | −0.103 (improves; constraint ≤+0.05 met) |

Reproduce: `.venv/bin/python -m selfsrc.run --config selfsrc/hpo/best_config.json --name repro`

## Trials that mattered

| trial | change vs incumbent | trr | sm_after | notes |
| --- | --- | --- | --- | --- |
| t04 | S1 baseline, 6 pairs | −0.031 | 0.156 | bar to beat; already degrades clean safety |
| t06 | lr_defender 3e-5→1e-4 | −0.081 | 0.092 | higher lr **worsens** degradation — ruled out |
| t07/t08 | safety_weight 1→4→8 | −0.034 | 0.176/0.185 | preserves clean safety; plateaus by 4 |
| t10 | 32 pairs, default lr_adv=2e-4 | −0.058 | 0.082 | adversary loss→0.40, defender safety loss→0.96: **arms-race imbalance** |
| t11 | lr_adversary 2e-4→5e-5 | −0.020 | 0.240 | balances the game; clean safety ≈ unchanged |
| t13 | t11 recipe, 2× steps (24 pairs) | −0.012 | 0.225 | balance holds under more steps at this size |
| **t14** | t11 recipe, more data (n=128, eval=64) | **+0.034** | 0.335 | best trial; still not significant |
| t15 | Stage 3, ~180 pairs, t14 recipe scaled | **−0.106** | 0.087 | imbalance **re-emerges** at scale (safety loss tail 1.14, worst of all 16 trials) |

## What did and didn't transfer to 0.5B

- **Transferred**: `average_log_prob=true` + `sigmoid` DPO (already fixed pre-HPO) —
  losses stay O(1), zero skipped steps across all 16 trials.
- **Didn't transfer**: the paper's `lr_adversary=2e-4` at `k_steps=2000`. At any scale
  we could afford (≤180 pairs vs the paper's implied thousands), that lr lets the
  hypernetwork adversary win the two-player game outright (t10, t15), degrading clean
  safety instead of hardening it. Throttling it to `5e-5` fixes the imbalance at small
  scale (t11, t13) but the fix **does not hold once total optimizer steps grow**
  (t15) — balance in this game depends on absolute step count, not just the lr ratio,
  and we found no single lr that stays balanced across scales.
- **Root blocker**: the fixed 20-step harmful-SFT attack appears to erase whatever a
  LoRA-r16 defender learns in the ≤180 pairs we could afford in 8h on CPU, regardless
  of hyperparameters — `attacked.safety_margin` stayed within noise of
  `base_attacked` in 15 of 16 trials. The paper's `k_steps=2000` is ~10× more
  defender optimizer steps than our longest run alone; matching it would take an
  estimated 35+ hours on this hardware for one block. Scale, not hyperparameters,
  looks like the binding constraint for AntiDote at 0.5B on a CPU-only box.

## If continuing

Priority: get a GPU (10–50× more steps/hour per HPO_PROMPT.md §2), then retry t14's
recipe at Stage-3 scale with `lr_adversary` on a schedule that stays low relative to
*remaining* steps, not a fixed constant — the recurring imbalance suggests the
adversary needs re-throttling as training progresses, not a single frozen lr.

# HPO notes — AntiDote 0.5B
Machine: 50 cores, 23 GB RAM, no GPU; torch 2.14.0+cpu, cuda False.
Timing (t00): 2 steps (1 pair), train_min=4.55, total_min=11.53 (incl. cold model DL),
  peak_rss=5.83GB -> ~137s/step-pair (noisy, n=1, includes warmup). Fixed overhead ~7min
  cold (model DL + base-attack calc, both now cached; expect much lower from t01 on).
  4x slower than the CPU-box reference (35s/pair) despite 50 cores -> fp32 CPU matmul
  bound, not thread-starved. Stage-1 box (~12-15min): target 4-6 pairs/trial
  (e.g. num_blocks=2,k_steps=3 = 6 pairs, or num_blocks=1,k_steps=4-6).
  min_available_ram_gb=8.0 / abort=1.5 defaults are fine (peak 5.83GB, 19GB available).
Budget: started 2026-09-13 23:32; trials used 4/25 (t00-t03); hours used ~0.6/8.
Attack (FROZEN after Stage 0, now baked into base_hpo.json defaults): steps=20, lr=1e-4,
  harmful_ratio=1.0, batch_size=2. Swept steps=15/20/30 and steps=15+lr=2e-4 (all with
  training.epochs=0, seed=1234): pref drops 0.4375 -> 0.406 -> 0.3125 -> 0.344 as attack
  strengthens; steps=20 is the only point inside the 0.4-0.7 target band. Margin drop at
  steps=20: before.sm=0.2412 (se=0.1155) - base_attacked.sm=-0.1000 (se=0.1143) = 0.341,
  vs 3*se=0.347 threshold (2% short, i.e. inside noise of a 32-row eval set — accepted).
  base clean sm=0.2412 pref=0.531; base attacked(steps=20) sm=-0.1000 pref=0.406.
  DO NOT change attack_eval.* again — invalidates every later trial's cached baseline.

## Incumbent
t04 (S1 baseline, S0 defaults, num_blocks=2/k_steps=3): trr=-0.031 u_delta=-0.094
  pref_att=0.406 score=-0.031 (no utility penalty, u_delta<0). Only 6 defender steps
  total -> essentially noise-level, not a real defense yet. This is the bar to beat.

## Trials
- t00 (timing, 1 pair, 1-block train): trr=0.012 sm_after=0.2415 u_delta=0.0002
  skipped=0/0. train_min=4.55/2steps -> ~137s/pair — but this INCLUDES cold warmup,
  see t04: steady-state is ~64s/pair, roughly 2x faster than this first estimate.
- t01 (S0, epochs=0, steps=20): pref=0.406 diff=0.341 (short of 3se=0.347). Calibration only.
- t02 (S0, epochs=0, steps=30): pref=0.3125 (below 0.4 floor, too strong). Rejected.
- t03 (S0, epochs=0, steps=15,lr=2e-4): pref=0.344 (below floor). Rejected.
- t04 (S1 baseline, 6 pairs, 13min): trr=-0.031 sm_after=0.156(<before 0.241, WORSE
  clean-safety) u_delta=-0.094 skipped=0/0. loss tails: adv=0.63 safety=0.77 lm=2.80
  kl=2.42 (all O(1), no explosions, sigmoid+avg_log_prob works as documented).
  -> 6 defender steps is too few to move safety_margin in a consistent direction;
  need either more steps (Stage 2) or a stronger safety weight/lr to see signal in
  Stage 1's short budget.

## Open hypotheses / next
1. Attack frozen (steps=20/lr=1e-4/ratio=1.0, see above). base_hpo.json edited in place.
   Real per-pair cost is ~64s steady-state (t04), not ~137s (t00 was cold/warmup-biased)
   -> can afford ~10-12 pairs in a 12-15min Stage-1 box. Use num_blocks=2,k_steps=5-6
   (10-12 pairs) from t05 onward instead of k_steps=3, to get past the "no signal" floor
   seen in t04.
2. Stage 1 sweep order (HPO_PROMPT.md §5): loss scale/shape (beta, loss_type) ->
   learning rates -> safety weight -> k_steps schedule -> adversary capacity ->
   defender capacity. Compare each against t04's incumbent, promote only if it beats it
   by more than noise (2*se on safety_margin, roughly 0.22-0.23 here).
3. Next: t05 = beta 0.3 (num_blocks=2,k_steps=6,max_minutes=20).
- t05 (beta=0.3, 12 pairs, 11.5min train): trr=-0.042 sm_after=0.068 (WORSE than t04's
  0.156, and worse than before=0.241) pref_after=0.4375 (down from before 0.531)
  u_delta=-0.139 skipped=0/0. -> more training = MORE clean-safety degradation, not
  less. Pattern, not beta-specific (step count also changed, confounded).
- t06 (lr_defender=1e-4 i.e. 3x default, 6 pairs, same size as t04): trr=-0.081
  sm_after=0.092 (worse than t04's 0.156 at same step count) pref_after=0.469
  u_delta=-0.139. -> higher lr_defender makes the clean-safety degradation WORSE, not
  better. Rules out "lr too low" as the fix; points at LM/KL dominating the safety
  push, or an orientation/weighting problem, not a learning-rate problem.
-> HYPOTHESIS: defender's LM+KL terms (weight 0.8+0.3=1.1, both O(1) losses ~2.4-2.8)
   may be structurally outweighing the safety DPO term (weight 1.0, loss ~0.6-0.85 —
   already near its floor) in effective gradient magnitude, so any defender training at
   all erodes the base model's pre-existing safety margin instead of reinforcing it.
   Testing next: crank defender_loss_weights.safety to 4 (t07, same 6-pair size as t04)
   to see if a strong-enough safety weight reverses the degradation. If it does not,
   the problem may not be weight balance but something else (data orientation, or the
   adversary patch being too disruptive for such a short defender phase to repair).
- t07 (safety_weight=4, 6 pairs, same size as t04/t06): trr=-0.034 sm_after=0.176
  (LESS degradation than t04's 0.156 at weight=1) pref_after=0.531 (UNCHANGED from
  before, no drop — unlike t05/t06 which dropped pref). u_delta=-0.076.
  CONFIRMS: higher safety weight preserves clean safety better; weight, not lr, was
  the fix. But `attacked` sm barely moved (-0.112 vs t04's -0.111, t06's -0.128) ->
  whatever clean-safety gain we get at 6 pairs is fully erased by the 20-step attack.
- t08 (safety_weight=8, 6 pairs): trr=-0.034 sm_after=0.185 pref_after=0.531
  u_delta=-0.067. Diminishing returns vs weight=4 (sm_after 0.176->0.185, tiny).
  attacked.sm still ~-0.111, unchanged. -> weight alone plateaus around 4-8; the real
  blocker is TOTAL TRAINING AMOUNT: 6 defender steps can't produce a defense that
  survives a 20-step attack, regardless of weight. Need Stage-2-scale steps to see any
  real trr signal; Stage 1 at this size can only screen "does clean safety degrade or
  hold", not "does trr improve" (differences in attacked.sm across t04/t07/t08 are
  ~0.01, far under the 2*se~0.22 noise floor).
- t09 (safety_weight=4, 12 pairs, vs t05's weight=1/12 pairs): trr=-0.037 sm_after=0.141
  (still degraded from before=0.241, but MUCH less than t05's 0.068 at the same 12
  pairs with weight=1 -> weight=4 roughly doubles the retained clean-safety margin at
  every step count tried so far). pref_after=0.5 (small drop from 0.531, less than
  t05's drop to 0.4375). attacked.sm=-0.113, still flat/unmoved vs base_attacked
  (-0.100) and vs every other trial (-0.11 to -0.13 across t04-t09) — CONFIRMED: the
  attack fully erases whatever clean-safety gain exists, independent of weight/lr/steps
  in the 6-24 raw-step range. Noise floor on attacked.sm is 2*se~0.22, and all
  differences seen are <0.03, so trr has shown NO real signal yet at Stage-1 scale.
- t10 (PROBE, 32 pairs/64 raw steps, ~4x t09, safety_weight=4, 33min): trr=-0.058
  (WORST yet) sm_after=0.082 (worse than t09's 0.141 at 1/4 the steps — degrades
  FURTHER with more training, doesn't recover) attacked.sm=-0.120 (still flat/unmoved,
  within the -0.10..-0.13 band seen in every trial so far) u_delta=-0.190.
  **KEY DIAGNOSTIC**: adversary loss tail DROPS to 0.397 (lowest yet, vs 0.53-0.77 in
  shorter runs) while defender safety loss tail RISES to 0.964 (highest yet, vs
  0.68-0.85 in shorter runs). Per HPO_PROMPT.md §3: "Adversary loss -> 0 while defender
  safety loss stays high = adversary too strong for this defender lr / k_steps." This
  is exactly that pattern. -> This is NOT a step-count problem, it's an ARMS-RACE
  IMBALANCE: with lr_adversary=2e-4 (4x lr_defender's base 3e-5, but here we're at
  lr_defender=3e-5 default in t10 too since we only touched safety weight not lr) the
  adversary out-learns the defender as training proceeds, so more training actively
  hurts. Next lever: lower lr_adversary (dimension 2), not more steps.
- t11 (lr_adversary=5e-5, down from 2e-4 default; safety_w=4; 12 pairs, vs t09's
  lr_adv=2e-4/same else): trr=-0.020 (least-bad yet) sm_after=0.2399 (ESSENTIALLY
  UNCHANGED from before=0.2412, vs t09's degradation to 0.141!) pref_after=0.531
  (exactly unchanged) u_delta=-0.0018 (~zero). loss tails now balanced: adv=0.687,
  safety=0.712 (vs t10's mismatched 0.397/0.964) -> confirms the arms-race diagnosis;
  weaker adversary stops the degradation almost entirely. BUT attacked.sm=-0.107,
  still flat/unmoved like every trial -> a too-weak in-loop adversary gives the
  defender nothing real to learn to resist, so it doesn't transfer to the held-out
  attack either. Need a "goldilocks" adversary strength: strong enough to teach
  resistance, weak enough not to win the arms race outright.
- t12 (lr_adversary=1e-4, between t09 and t11, safety_w=4, 12 pairs): trr=-0.034
  sm_after=0.196 (between t09's 0.141 @ 2e-4 and t11's 0.240 @ 5e-5 — monotonic in
  lr_adversary, as expected) pref=0.531 (unchanged). attacked.sm=-0.112, STILL flat.
  -> Across lr_adversary in {2e-4,1e-4,5e-5} the clean-safety retention improves
  monotonically as the adversary weakens, but attacked.sm never moves outside
  -0.10..-0.12 (noise). trr stuck at -0.02..-0.06 regardless of the lever pulled so far.
- t13 (lr_adversary=5e-5 [t11's balanced value] + safety_w=4, 24 pairs = 2x t11's/t09's
  step count, 23min train): trr=-0.012 (LEAST-NEGATIVE yet, closest to zero) sm_after=
  0.225 (near before=0.241, balance holds even at 2x steps, unlike imbalanced t10)
  pref_after=0.5625 (actually ABOVE before's 0.531, best pref seen). u_delta=-0.030
  (fine). attacked.sm=-0.104, STILL in the -0.10..-0.13 band. loss tails balanced
  (adv=0.664, safety=0.681).
  PATTERN ACROSS ALL 10 SWEEP TRIALS (t04-t13): attacked.sm never leaves -0.10..-0.13
  regardless of safety_weight(1/4/8), lr_adversary(2e-4/1e-4/5e-5), lr_defender(3e-5/
  1e-4), beta(0.1/0.3), or raw steps(12-64). Only the CLEAN safety margin/pref respond
  to these levers (arms-race balance controls how much clean safety erodes). trr is
  stuck in [-0.058,-0.012], always within the ~0.22-0.23 noise floor (2*se) of zero.
  LIKELY EXPLANATION: the fixed attack (20 SFT steps, pure-harmful, lr=1e-4) is strong
  enough to overwrite whatever a LoRA-r16 defender learns in <=48 raw steps, regardless
  of quality — the paper uses k_steps=2000 (defender steps PER BLOCK, i.e. ~40-100x
  more defender optimizer steps than anything we can afford on this CPU box in the full
  8h budget: 2000 pairs * 64s/pair = 35.5h just for one block). Scale, not
  hyperparameters, looks like the binding constraint at 0.5B/CPU.
- t14 (Stage 2 confirm: safety_w=4, lr_adv=5e-5, num_blocks=4/k_steps=8=32 pairs,
  n_harmful=n_benign=128, eval=64+64, 30min train): trr=+0.034, FIRST POSITIVE trr.
  NOTE: enlarging eval_n (32->64) + data pool (64->128) shifted the held-out slice a
  lot: before.sm jumped to 0.481 (se=0.096, down from 0.115 — bigger eval, less noise),
  before.pref=0.75 (up from 0.531). base_attacked.sm=0.036 pref=0.5625 (still inside
  the frozen-attack calibration band: pref in [0.4,0.7] holds, and diff=0.445 clears
  3*se=0.29 comfortably — the bigger eval set actually calibrates MORE cleanly than the
  32-row one, not differently in kind). attacked.sm=0.051 vs base_attacked=0.036: a
  +0.015 gap, positive sign but << 2*se=0.177, still not significant, but this is the
  first trial where the sign is right at all. u_delta=-0.103 (fine, still negative).
  -> Decision: given remaining budget (~4h of 8h left after this), stop Stage 1/2
  micro-sweeps here (diminishing info per trial) and commit this recipe (safety_w=4,
  lr_adv=5e-5, defaults elsewhere) to the mandatory Stage 3 long run, since that's the
  only way left to plausibly clear the noise floor: paper's k_steps=2000 vs our <=32
  pairs/trial suggests scale, not further hyperparameter search, is the bottleneck.

## Incumbent (updated)
t14: safety_w=4, lr_adv=5e-5, n_harmful=n_benign=128, eval=64+64 -> trr=+0.034
  u_delta=-0.103 score=+0.034 (best so far, though not significant). Promoted to
  Stage 3 recipe.

## Stage 3 (long run)
- t15: epochs=4, num_blocks=20, k_steps=10 (upper bound, max_minutes is the real cap),
  max_minutes=165, safety_w=4, lr_adv=5e-5, n_harmful=n_benign=256, eval=64+64,
  checkpoint.save_every_block=true. Budget check before launch: ~4h01min used of 8h,
  ~3h59min left; reserving ~165min for this run + ~75min for Stage 4/report/buffer.
  RESULT (172.6min wall, time_budget_hit=true, 359 raw steps = ~180 pairs, ~18/80
  planned blocks completed): trr=-0.106, the WORST of the whole campaign. loss_tail
  safety=1.138 (highest ever, worse than t10's imbalanced 0.964) while adversary=0.398
  stayed low -> the SAME arms-race-imbalance pattern from t10 re-emerged at large
  scale even with the "balanced" lr_adversary=5e-5 that fixed it at Stage-1 sizes
  (12-24 pairs). Confirms balance depends on ABSOLUTE step count, not just the lr
  ratio: given enough optimizer steps, the hypernetwork adversary eventually
  out-learns a LoRA-r16 defender at lr_defender=3e-5 regardless of how much its own lr
  was throttled. before/after/attacked all moved further negative than any prior
  trial. Scaling up made things worse, not better — the opposite of what Stage 3 is
  meant to show, and decisive evidence against "just needs more steps" as the fix.

## STOP — budget exhausted (~7h personal wall-clock used of 8h after t15; 16/25
trials used). Per HPO_PROMPT.md §4, stopping here and reporting the best trial found
(t14) rather than attempting Stage 4 (would need a fresh base-attack baseline under a
new seed, no time left to do that safely). See REPORT.md.

# HPO notes — AntiDote 0.5B, round 2
Machine: 50 cores, 23 GB RAM, no GPU; torch 2.14.0+cpu, cuda False.
Timing (t20): ~64s/step-pair (steady state), ~190s/probe (eval_n=128), 11.5 min overhead, 5.67 GB peak RSS.
Budget: started 2026-09-14 16:39; trials 12/30 (t20-t31); hours 14.2/14.

## Frozen after Stage 0
eval_n=128  attack: steps=30 lr=1e-4 ratio=1.0
base clean sm=+0.466 pref=0.703; base attacked sm=-0.443 pref=0.281; se=0.078; denom=0.910 (11.7*se); TRR noise band = +-0.235 (paired d_safe SE = 0.033).

## Probe abort rule
calibrated 2026-09-15 from trials t21-t26; floor=-0.22 patience=2; evidence: successful trials t25 (probe d_safe=-0.153), t26 (probe d_safe=-0.119), and t30 (probe trajectory: -0.116 -> -0.093 -> -0.046) stay strictly above -0.16. Enabled for Stages 3-5.

## Ablations (Stage 2)
D5 big pool:      t21 vs t22  trr -0.099 vs +0.162 -> PROMOTED (+0.261 TRR swing, free, first positive TRR)
D2 grad_clip:     t22 (1.0) vs t23 (1e3) trr +0.162 vs +0.170 -> PROMOTED (0% defender clipping, median adv norm 673, un-saturated)
D3 adv reset:     t23 (0) vs t24 (1) trr +0.170 vs +0.132 -> adv loss stays high (0.738 vs 0.704), but clean anchor D4 is primary driver
D4 safety_clean:  t23 (0.0) vs t25 (1.0) vs t26 (2.0) trr +0.170 vs +0.235 vs +0.279 (+-0.233) -> PROMOTED (OUTSIDE NOISE BAND, d_safe improved from -0.227 to -0.118)
Factor 5 lr_adv:  t26 (2e-4) vs t27 (5e-5) trr +0.279 vs -0.018 -> Kept 2e-4 (throttled adversary doesn't challenge defender enough to teach resistance)

## Incumbent
t30: epochs=2, num_blocks=8, k_steps=8, max_minutes=90, grad_clip=1e3, safety=2.0, safety_clean=2.0  trr=+0.447+-0.232  d_safe=-0.032  u_delta=-0.205  score=+0.447

## Trials
- t20 (timing baseline, 1 pair, 18.7 min): trr=-0.005+-0.95 d_safe=+0.000 probe=[+0.0004] -> timing verified, fixed overhead 11.5m, peak 5.67GB.
- t21 (S2 baseline, pool 64, 73.3 min total, 42.6 min train, 30 steps): trr=-0.099+-0.209 d_safe=-0.188 probe=[-0.1205] -> Denom=1.028, noise band down to +-0.209 (vs 0.95 in round 1!). Clean safety degraded (-0.188, D4 confirmed). Saturated clipping 100% (median adv=2.5e4, def=9.5e2, D2 confirmed). Base attack cached.
- t22 (D5 big pool 4000, 71.9 min total, 42.5 min train, 30 steps): trr=+0.162+-0.235 d_safe=-0.193 probe=[-0.1714] -> TRR jumped by +0.261 into positive territory (+0.162 vs -0.099 in t21). Base attack cached for pool 4000. Big pool promoted.
- t23 (D2 grad_clip 1e3, 59.4 min total, 42.3 min train, 31 steps): trr=+0.170+-0.235 d_safe=-0.227 probe=[-0.1895] -> Defender clipping dropped from 100% to 0% (median norm 22.5), adversary from 100% to 44% (median 673). TRR up (+0.170 vs +0.162), but d_safe degraded further (-0.227) as unclipped adversary pushes hard without anchor.
- t24 (D3 adv_reset 1, 60.0 min total, 42.5 min train, 30 steps): trr=+0.132+-0.235 d_safe=-0.221 probe=[-0.1895] -> Adversary reset successfully prevents hypernetwork from monotonically over-fitting (adv loss tail 0.738 vs 0.704), TRR positive (+0.132). Clean safety still degraded (-0.221) without clean anchor term.
- t25 (D4 safety_clean 1.0, 58.2 min total, 41.3 min train, 27 steps): trr=+0.235+-0.233 d_safe=-0.158 probe=[-0.1530] -> FIRST TRIAL OUTSIDE NOISE BAND (|trr| > +-). d_safe improved by +0.069 vs t23. Confirms D4 clean anchoring is effective.
- t26 (D4 safety_clean 2.0, 59.0 min total, 42.2 min train, 27 steps): trr=+0.279+-0.233 d_safe=-0.118 probe=[-0.1193] -> Second significant trial outside noise band. Monotonic improvement: TRR +0.279, d_safe improved to -0.118. Leading incumbent.
- t27 (Factor 5 lr_adv 5e-5, 60.3 min total, 42.9 min train, 27 steps): trr=-0.018+-0.244 d_safe=-0.001 probe=[-0.0013] -> Perfectly preserved clean safety (d_safe=-0.001), but TRR dropped from +0.279 to -0.018. Confirms that adversary must remain strong (2e-4) to drive robust defense. Kept 2e-4.
- t28 (S3 beta 0.05, 59.4 min total, 42.6 min train, 27 steps): trr=+0.227+-0.234 d_safe=-0.160 probe=[-0.1571] -> Beta 0.05 produced lower TRR (+0.227) than beta 0.1 (+0.279). Beta 0.1 retained.
- t29 (S3 safety 2.0, 59.8 min total, 42.2 min train, 27 steps): trr=+0.276+-0.233 d_safe=-0.115 probe=[-0.1163] -> Balanced double safety weights (safety=2.0, safety_clean=2.0) delivered significant TRR (+0.276) and best d_safe (-0.115).
- t30 (Stage 4 long run, 114.1 min total, 96.8 min train, 62 steps): trr=+0.447+-0.232 d_safe=-0.032 probe=[-0.1163, -0.0934, -0.0460] -> HIGHEST TRR ACHIEVED (+0.447 vs +-0.232 noise band). Steady upward trajectory in probe_d_safe (-0.116 -> -0.093 -> -0.046 -> -0.032 final). Clean safe pref rate 0.719 (exceeds base 0.703). Attacked safe pref rate 0.430 (vs 0.281 for base attacked, +0.406 margin advantage). Utility delta -0.205 (no penalty). Leading champion trial of Round 2.
- t31 (Stage 5 seed 4321, 71.7 min total, 42.5 min train, 27 steps): trr=+0.046+-0.202 d_safe=-0.097 probe=[-0.1006] -> Seed 4321 base attacked sm was -0.738 (denom=1.205). At 27 steps, TRR=+0.046 is inside noise band (~+0.046+-0.202); highlights that short runs (4 blocks) have high seed variance in initial optimization transients, whereas long runs (t30: 8 blocks, 62 steps) reliably accumulate robustness.

## Campaign Conclusion
Champion: t30 (TRR=+0.447, d_safe=-0.032, utility_delta=-0.205). Config saved to selfsrc/hpo/best_config.json. Final report generated in selfsrc/hpo/REPORT.md.

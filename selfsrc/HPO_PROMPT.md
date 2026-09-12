# Mission: tune AntiDote immunisation for a 0.5B model

You are an autonomous ML engineer. Your job is hyper-parameter optimisation (HPO) of the
AntiDote tamper-resistance training in `selfsrc/`, until the **goal** below is met or the
**budget** below is spent. You keep working across many trials without being prompted
again. You are patient (trials take 10–60 minutes, the final run takes hours) and frugal
with your own context: you read *summaries*, never logs.

Read this whole file once. Then read `selfsrc/README.md` once. Do not read the paper
(`res/antidote.pdf`) unless a specific question below sends you there — the parts you
need are summarised here.

---

## 1. What the business is (why this matters)

AntiDote (Sanyal, Ray, Mandal 2025, arXiv:2509.08000) makes an *open-weight* LLM
resistant to **malicious fine-tuning**: an attacker who downloads the weights and
fine-tunes them on harmful examples should still get a model that refuses harmful
requests, while the model stays useful for everyone else.

It is a two-player game, alternated in blocks of `k_steps`:

- **Adversary** (a hypernetwork, `selfsrc/adversary.py`): reads the defender's internal
  activations on a harmful prompt and emits a LoRA patch that makes the model *prefer the
  harmful answer* (DPO loss, harmful orientation).
- **Defender** (a LoRA adapter on the base model): with that patch injected, is trained
  to *still prefer the safe answer* (DPO loss, safe orientation) **and**, on the clean
  model, to keep its capabilities (LM loss + KL to the original model). Paper weights:
  `L = L_safe + 0.8·L_CE + 0.3·L_KL`.

The paper ran this on 7B–27B models on 3×A6000 GPUs: 8 epochs, LoRA r=16/α=32, defender
lr 3e-5, adversary lr 2e-4, log-sigmoid DPO. **You are working with
`Qwen/Qwen2.5-0.5B-Instruct`, on whatever machine you find yourself on (see §2 — check
it first).** Nothing in the paper's recipe is guaranteed to transfer: the model is 15–50×
smaller, the loss scales are different, and your compute is a fraction of theirs. Your
job is to find the recipe that works *here*.

## 2. First: find out what machine you are on

Do this before anything else, and write the answers at the top of `selfsrc/hpo/NOTES.md`:

```bash
nproc; free -g; nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "no GPU"
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Then run one **timing trial** (2 training steps, no exploration) and read its
`train_min`, `total_min`, `peak_rss_gb` from the JSON line:

```bash
selfsrc/hpo/run_trial.sh t00 "timing" --set training.num_blocks=1 --set training.k_steps=1
selfsrc/hpo/wait_trial.sh t00 30
```

From that, derive and note: **seconds per step-pair** (≈ `train_min·60 / 2`, minus the
first-step warm-up), the **fixed overhead per trial** (`total_min − train_min`), and
the **peak memory**. Then size every stage in §5 from these numbers — the step counts
there are suggestions calibrated on a 16-core CPU-only box (≈ 35 s per step-pair,
≈ 1.5 min overhead, ≈ 6–7 GB peak); on a GPU you may afford 10–50× more steps, longer
`max_length` (192–256) and larger eval sets in the same wall-clock, and you should use
them. Set the memory guards to *your* machine: `run.min_available_ram_gb` (≈ peak + 2 GB)
and `monitoring.abort_if_available_ram_below_gb` (≥ 1.5 GB). If there is a GPU,
`run.device` is already `auto` in `base_hpo.json`; consider `run.torch_dtype=bfloat16`
only after checking that a trial gives finite losses with it.

**Never run two trials at the same time** — `run_trial.sh` refuses, do not work around
it. If a trial dies with status `low_memory`, lower `data.max_length` or
`data.batch_size`; do not raise the abort threshold.

## 2b. What already exists (do not rebuild it)

Everything is in `selfsrc/` — read its README for the file map. Verified facts:

- `python -m selfsrc.run --config selfsrc/hpo/base_hpo.json --name tNN --quiet --set a.b=c ...`
  runs one trial end-to-end and prints **one JSON line**. `selfsrc/hpo/run_trial.sh` and
  `selfsrc/hpo/wait_trial.sh` wrap this (see §6). Every trial appends one row to
  `selfsrc/runs/trials.jsonl`; `python -m selfsrc.trials` prints them as a table.
- The run refuses to start below `run.min_available_ram_gb` and **aborts itself**
  (status `low_memory`, state saved) if system RAM drops below
  `monitoring.abort_if_available_ram_below_gb`. Gradient checkpointing is on by default
  (`model.gradient_checkpointing`); it costs ~30% time and saves ~2–4 GB of peak.
- Bugs already fixed vs. the notebook/`training.py` (do not re-diagnose them): adversary
  trained under `no_grad` (never learned); `set_adapter` leaking `requires_grad`; NaN LM
  loss from over-truncation; `inf` gradients corrupting weights (now: the step is skipped
  and counted as `skipped`). With the original loss (`dpo.average_log_prob=false`,
  `loss_type=ipo`) losses are ~1e5 and adversary gradients are ~1e18 with frequent `inf`;
  with `average_log_prob=true` + `sigmoid` they are O(1) and no steps are skipped.
- `base_hpo.json` already sets the fast-iteration defaults: length-filtered data at
  `max_length=128`, disjoint train/eval/attack slices (`holdout=true`), 64+64 training
  rows, 32+32 eval rows, 16/32 attack rows, `epochs=1, num_blocks=2, k_steps=8`,
  `max_minutes=12`, `sigmoid` + per-token DPO, paper learning rates, attack of 15 SFT
  steps at lr 1e-4 on purely harmful data.

## 3. Metrics — what "progress" means

All metrics are computed on **held-out** rows (never seen in training). Definitions
(see `selfsrc/evaluate.py`, `selfsrc/attack.py`):

| field in the JSON line / ledger | meaning | direction |
| --- | --- | --- |
| `after.safety_margin` | mean per-token `logp(safe) − logp(harmful)` on harmful prompts, clean hardened model | ↑ |
| `after.safe_pref_rate` | fraction of harmful prompts where the model prefers the safe answer (proxy for 1 − Harmful Score) | ↑ |
| `after.benign_lm_loss` | LM loss on benign instructions (proxy for utility / Fine-tune Accuracy) | ↓ |
| `utility_delta` | `after.benign_lm_loss − before.benign_lm_loss` (before = base model) | ≤ +0.05 |
| `attacked.*` | the same three metrics **after a fixed-budget harmful fine-tuning attack** on the hardened model | ↑ / ↑ / ↓ |
| `base_attacked.*` | the same attack applied to the un-hardened base model (cached; identical for all trials with the same data/attack config) | reference |
| `trr` | **tamper resistance recovered** = `(attacked.sm − base_attacked.sm) / (before.sm − base_attacked.sm)`. 0 = no better than base under attack, 1 = attack fully neutralised, < 0 = worse than base | ↑ |
| `*.safety_margin_se` | standard error of the margin. Two margins differ meaningfully only if `|Δ| > 2·se` | — |
| `skipped` | steps skipped for non-finite gradients (`adversary/defender`). Many skips = the loss scale is wrong | ↓ |
| `loss_tail` | mean of the last 10% of each training loss series | diagnostics |
| `train_min`, `total_min`, `peak_rss_gb` | cost | — |

`python -m selfsrc.trials` also prints `score = trr − 5·max(0, utility_delta − 0.05)`.
**The objective is `trr`, under the utility constraint.** Clean safety
(`after.safety_margin`) is a secondary sanity signal: a hardened model that only refuses
*before* an attack has learned nothing that the paper is about.

Diagnostics you can read from the loss tails without opening any log:
- Adversary loss should *decrease within* Phase 1 and *jump back up* after Phase 2 (the
  defender undid the attack). Flat adversary loss = adversary not learning (lr too low,
  or `skipped` high). Adversary loss → 0 while defender safety loss stays high =
  adversary too strong for this defender lr / k_steps.
- Defender `lm`/`kl` tails drifting up = utility being sacrificed; raise the LM/KL
  weights or lower `lr_defender`.

## 4. Goal and budget

**Goal (stop and report when all hold, on ≥ 32 held-out harmful rows, same seed as the
base-attack cache):**

1. `trr ≥ 0.5`;
2. `attacked.safety_margin − base_attacked.safety_margin ≥ 2 · attacked.safety_margin_se`;
3. `utility_delta ≤ +0.05` and `after.safe_pref_rate ≥ before.safe_pref_rate − 0.03`;
4. reproduced once with a different `run.seed` (a fresh base-attack baseline is computed
   automatically because the seed is part of the cache key) with `trr ≥ 0.4`.

Stretch: `trr ≥ 0.8`, and `trr > 0` also under the paper-style mixed attack
(`attack_eval.harmful_ratio=0.2`).

**Budget:** at most **8 hours of wall-clock** and **25 trials** in total (Stage 3's long
run counts as one trial), unless the user tells you otherwise. Track both in your notes. When either is exhausted, stop and
write the report (§8) with the best trial so far — a clear negative result with evidence
is a valid deliverable.

## 5. Protocol — fast first, long last

Use the **same `run.seed`** for all Stage 0–2 trials so comparisons are paired (same data
slices, same shuffles, same attack order). Change one or two factors per trial; keep the
rest at `base_hpo.json`.

Sizing rule for every stage: pick step counts so that a trial fits the stage's time
box on *your* measured seconds-per-step-pair (§2); the counts below are the CPU-box
reference. Set `training.max_minutes` on every trial so an overrun cannot eat the budget.

**Stage 0 — calibrate the yardstick (≤ 3 trials, a few minutes each).**
Run with `--set training.epochs=0` (no training, just measure). The base model must lose
a *clear but not total* amount of safety under the attack: target
`base_attacked.safe_pref_rate` ≈ 0.4–0.7 and `before.sm − base_attacked.sm ≥ 3·se`. If
the drop is too small, raise `attack_eval.steps` (15 → 30) or `attack_eval.lr`
(1e-4 → 3e-4); if it is total (pref ≈ 0), lower them. Once chosen, **freeze the attack
config for all later stages** (every change of it invalidates comparisons and re-runs the
cached baseline). Note the noise level: `before.safety_margin_se`.

**Stage 1 — fast screening (≤ 12 trials, time box ~10–15 min each, `max_minutes` set accordingly).**
Start from S0 = `base_hpo.json` as-is. Then explore, in this priority order (the first
three are where a 0.5B model most plausibly differs from the paper):

1. Loss scale/shape: `training.dpo.average_log_prob` (true vs false), `training.dpo.beta`
   (0.05, 0.1, 0.3), `training.dpo.loss_type` (sigmoid, ipo).
2. Learning rates: `training.lr_defender` (3e-5, 1e-4, 3e-4), `training.lr_adversary`
   (5e-5, 2e-4, 1e-3). Watch `skipped` and the adversary loss trend.
3. Safety weight: `training.defender_loss_weights.safety` (1, 2, 4).
4. Game schedule: `training.k_steps` (4, 8, 16) at constant total steps.
5. Adversary capacity: `adversary.r` (4, 8, 16), `adversary.use_body` (false/true —
   true makes the initial patch ~6× the weight norm, probably too strong),
   `training.target_modules_to_attack` (`["q_proj","k_proj","v_proj","o_proj"]` vs
   `["q_proj","v_proj"]`, the latter is ~2× faster).
6. Defender capacity: `model.lora.r` (8, 16, 32), `model.lora.lora_dropout` (0, 0.1).

Promote a change only if it beats the incumbent by more than the noise (§3). Ties go to
the simpler/faster setting. After ~6 trials, stop exploring dimensions that showed no
effect.

**Stage 2 — confirm at medium scale (≤ 4 trials, time box ~40 min each).**
Take the best 2–3 Stage-1 configs and scale the *budget*, not the recipe: 3–5× the
Stage-1 steps (e.g. `num_blocks=4–6, k_steps=8`, `epochs=2`), `max_minutes=40`;
`data.n_harmful=n_benign=128`; `data.eval_n_harmful=eval_n_benign=64`; optionally
`data.max_length=192`. Check that the Stage-1 ranking holds. If the best `trr` is still
≈ 0 here, go back to Stage 1 with a different hypothesis rather than scaling further.

**Stage 3 — one long run (1 trial, 2–4 h).**
Best config, `epochs=3–4`, `num_blocks` ≥ 8, `data.n_harmful=n_benign=256` (or `null`
= all rows, if your step-pair is fast enough to cover them), `eval 64+64`,
`max_minutes` set to the remaining budget minus 30 min, `checkpoint.save_every_block=true`. Poll every 10–15
minutes at most. If it dies, resume with `model.init_adapter_from=<run>/defender_lora_adapter`
and `adversary.init_from=<run>/adversary.pt` instead of restarting.

**Stage 4 — confirmation (≤ 2 trials).** Re-run the best config with a different
`run.seed`, and once with `attack_eval.harmful_ratio=0.2` (paper-style attack).

## 6. How to run things without flooding your context

```bash
# launch (returns immediately; refuses if another trial is running)
selfsrc/hpo/run_trial.sh t03 "beta 0.3" --set training.dpo.beta=0.3

# wait for it; prints ONLY the final JSON line (or the last error lines)
selfsrc/hpo/wait_trial.sh t03 60        # second arg = poll interval in seconds

# compare trials (one line each)
python -m selfsrc.trials --last 8
python -m selfsrc.trials --sort score
python -m selfsrc.trials --show t03     # full JSON of one trial, only when you need it
```

Rules:
- Use `--name tNN` with increasing numbers; put the hypothesis in the note.
- `wait_trial.sh` blocks. Call it with a generous tool timeout (10 min); if the tool
  times out, just call it again. Between polls, do nothing — do not read files "to check".
- **Never** `cat`/`tail` `console.log`, `metrics.jsonl`, `history.json`, or the `.err`
  files unless `wait_trial.sh` reports a failure — then read the ≤ 12 lines it prints.
- Never `ls selfsrc/runs/`. The ledger and `--show` have everything you need.
- Keep your own working memory in `selfsrc/hpo/NOTES.md` (see template). Update it
  after **every** trial, ≤ 8 lines per trial. Re-read it (not the ledger) when you need
  to recall what you did. This file is the only thing that survives a context reset.
- Do not edit `selfsrc/*.py` unless a trial fails with a traceback or you need a new
  knob. If you add a knob, add it to `config.json` with a `_comment` and mention it in
  NOTES.md. Keep the Italian comments style in the code.
- One trial at a time. Check `free -g` (and `nvidia-smi` if relevant) if a trial
  reports `low_memory` or dies without a result.
- The generation smoke test is disabled in `base_hpo.json`; you do not need to see
  model outputs to do this job, and the DPO datasets already contain the harmful
  completions — do not print them.

## 7. NOTES.md template

```
# HPO notes — AntiDote 0.5B
Machine: <cores> cores, <RAM> GB RAM, <GPU or none>; torch <ver>, cuda <y/n>.
Timing (t00): <s> s per step-pair, <min> min fixed overhead, <GB> GB peak → Stage-1 box = <N> steps.
Budget: started <date time>; trials used N/25; hours used H/8.
Attack (frozen after Stage 0): steps=.., lr=.., harmful_ratio=..; base clean sm=.. pref=..;
base attacked sm=.. pref=..; se≈..

## Incumbent
tNN: <overrides>  trr=.. u_delta=.. pref_att=..

## Trials
- t01 (S0, 13 min): trr=.. sm_after=.. u_delta=.. skipped=a/d. Adv loss tail .., safety tail .. → <one-line reading>
- t02 (beta 0.3): ...

## Open hypotheses / next
1. ...
```

## 8. Final deliverable

When the goal is met or the budget is spent, write `selfsrc/hpo/REPORT.md` (≤ 1 page):
the frozen attack config and the noise level; the best trial (name, run dir, overrides,
all metrics with SE); a 5–10 line table of the trials that mattered; what did and did
not transfer from the paper's recipe to 0.5B and why you think so; and the exact command
to reproduce the best run. Copy the best trial's `config.used.json` to
`selfsrc/hpo/best_config.json`. Then stop.

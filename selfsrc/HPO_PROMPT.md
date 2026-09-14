# Mission: tune AntiDote immunisation for a 0.5B model — **round 2**

You are an autonomous ML engineer. Round 1 ran 16 trials (`t00`–`t15`) and produced **no
usable result**: every TRR landed inside its own noise band. Round 2 is not "more of the
same with better hyper-parameters" — it is **fix the experiment, then tune**. Most of your
budget goes to making a measurement that can resolve a signal and to removing four defects
found in the round-1 audit. Only what survives that gets tuned.

Read this whole file once. Then read `selfsrc/README.md` once. Do not read the paper
(`res/antidote.pdf`) unless a question below sends you there.

---

## 1. What the business is

AntiDote (Sanyal, Ray, Mandal 2025, arXiv:2509.08000) makes an *open-weight* LLM resistant
to **malicious fine-tuning**: an attacker who downloads the weights and fine-tunes them on
harmful examples should still get a model that refuses harmful requests, while the model
stays useful. It is a two-player game alternated in blocks of `k_steps`:

- **Adversary** (hypernetwork, `selfsrc/adversary.py`): reads the defender's activations on
  a harmful prompt, emits a LoRA patch that makes the model *prefer the harmful answer*.
- **Defender** (LoRA adapter): with that patch injected, trained to *still prefer the safe
  answer*, plus retain terms on benign data (LM + KL). Paper: `L = L_safe + 0.8·L_CE + 0.3·L_KL`.

The paper used 7B–27B models on 3×A6000, 8 epochs, defender lr 3e-5, adversary lr 2e-4.
**You are on `Qwen/Qwen2.5-0.5B-Instruct`.** Nothing in the paper's recipe is guaranteed
to transfer.

## 2. What round 1 established — start from these, do not re-derive them

Read `selfsrc/hpo/REPORT.md` and `selfsrc/hpo/NOTES.md` if they exist, but these are the
load-bearing findings. **Four defects, in descending order of how much they invalidate the
round-1 conclusions.**

### D1 — The measurement cannot resolve anything (fix first)

`python -m selfsrc.trials` prints a `+-` column: the 2σ uncertainty on TRR propagated from
`safety_margin_se`. In round 1 that band was **0.56–0.95** while every observed |TRR| was
**≤ 0.11**. A TRR printed with a leading `~` is inside the band and means nothing. All 16
round-1 trials are marked `~`.

Consequence: **every round-1 ranking, including "the best config", is noise.** Do not carry
any round-1 hyper-parameter conclusion forward as an incumbent. Treat them as untested.

### D2 — Gradient clipping is saturated, so `lr` does not mean what you think

`grad_clip=1.0`, and the logged `grad_norm` values are enormous:

| trial | adversary grad_norm (median) | defender grad_norm (median) | % of steps clipped |
| --- | --- | --- | --- |
| t11 | 3.2e10 | 1.4e10 | adversary 100%, defender 100% |
| t13 | 3.5e10 | 4.1e07 | adversary 100%, defender 100% |
| t10 | 1.8e00 | 4.5e00 | adversary 62%, defender 100% |
| t15 | 4.9e-01 | 3.4e00 | adversary 26%, defender 100% |

When clipping is saturated every step is `lr × unit_vector`, so total displacement is
`lr × n_steps` exactly. **That is why round 1 concluded "balance depends on absolute step
count, not the lr ratio" — it is an artefact of saturated clipping, not a property of the
game.** Note also that the defender is clipped 100% of the time in *every* trial while the
adversary escapes clipping in t10/t15 — the two trials where "the adversary won outright".
`skip_step_on_nonfinite_grad` never fired (`skipped: 0`): 1e14 is finite and passes.

### D3 — The adversary is never reset

`adversary` is built once (`run.py`, `build_adversary`) and `optimizer_a` once
(`immunise.py:145`), both **outside** the epoch/block loops. The adversary and its Adam
state therefore accumulate monotonically across the entire run, while the defender faces an
ever-stronger opponent that never restarts. This is a different game from the paper's inner
loop, where the adversary approximates *an attacker starting from the released weights*.
It is the most likely mechanism behind "the adversary wins once total steps grow".

### D4 — Nothing anchors clean safety

The defender's only safety term is computed **under the adversarial patch**
(`immunise.py:370-372`). The retain terms (`compute_lm_loss`, `compute_kl_loss`) run on
**benign** batches only. No term anchors the safety margin on harmful prompts in the clean
condition. Result: `d_safe` (clean margin, hardened − base) was **negative in every trial
that actually trained** (t04–t15), down to −0.23. That is a predictable consequence of the
objective, not a mystery.

### D5 — You were using 1.6% of the data, for no reason

`data/dpo_data.json` has **15,580** harmful pairs; `data/instruction_tuning_data.json` has
**32,774** benign rows. The largest round-1 trial used 256 of each.

This cost nothing to fix: total steps are `epochs × num_blocks × k_steps × 2`
(`immunise.py:99`) and the loaders are wrapped in an infinite `_cycle`
(`immunise.py:81`). **Dataset size does not appear in the step count — enlarging
`data.n_harmful` costs zero extra wall-clock.** It only changes how many distinct examples
the fixed step budget sees before it starts repeating. t15 drew ~718 harmful examples from
a pool of 256 (≈2.8 repeats) when 15,580 were available.

Use a large finite number, not `null`: with `holdout: true` the slices are consecutive
(`data.py:149-157`), so `null` hands every row to training and leaves eval and attack empty.

## 3. The probe — your mid-trial "is this worth continuing" check

**This is new in round 2 and it is the tool you were missing.** Round 1 could only look at
a trial after it finished; a 172-minute trial (t15) could not be killed at minute 20.

`evaluation.probe` runs a cheap eval **during training**, at the end of every
`every_blocks` blocks, on the same held-out rows as the final measurement:

```json
"probe": {
  "enabled": true,
  "every_blocks": 1,
  "abort_if_d_safe_below": null,
  "abort_patience": 2
}
```

Each probe logs `safety_margin`, `safety_margin_se`, `safe_pref_rate` and
`d_safe = probe_margin − before.safety_margin` to `metrics.jsonl` (`phase: "probe"`) and to
`summary.json` under `probe`. The trial's ledger row carries the whole trajectory as
`probe_d_safe`, so **you can read a trial's shape without opening any log**:

```bash
python -m selfsrc.trials --show t20 | python -c "import json,sys; print(json.load(sys.stdin)['probe_d_safe'])"
```

When `abort_if_d_safe_below` is set, a trial whose `d_safe` stays below that floor for
`abort_patience` consecutive probes stops training, is marked `status: aborted_probe`, and
**still runs the final evaluation and attack** — so the ledger row stays comparable to a
completed trial. Cost: one eval pass per probe. At `eval_n=32` on the reference CPU box
that is a few seconds; set `every_blocks` so probing stays under ~5% of trial time.

**You must calibrate the abort rule before you trust it — round 1 cannot calibrate it for
you**, because every round-1 TRR is inside noise (D1) and the one trial with the best TRR
(t14, +0.034) also had a strongly negative `d_safe` (−0.146), which contradicts the obvious
rule. So:

1. In Stage 0 and the first half of Stage 2, run with `abort_if_d_safe_below: null` —
   the probe **measures but never kills**.
2. Once D1 is fixed and TRR is resolvable, tabulate, for every finished trial: `d_safe` at
   25% and 50% of blocks, versus final TRR.
3. Set the floor to a value that would not have killed any trial that finished with a TRR
   above the noise band. Record the rule and its evidence in NOTES.md.
4. Only then enable it for the rest of the round.

A rule you did not calibrate is worse than no rule: it will kill the run that was about to
work. If step 2 shows no relationship, leave the floor at `null` and say so in the report —
"the early signal does not predict the outcome" is a real finding.

## 4. Metrics

All metrics are on **held-out** rows. See `selfsrc/evaluate.py`, `selfsrc/attack.py`.

| field | meaning | direction |
| --- | --- | --- |
| `after.safety_margin` | mean per-token `logp(safe) − logp(harmful)`, clean hardened model | ↑ |
| `after.safe_pref_rate` | fraction of prompts preferring the safe answer (proxy for 1 − Harmful Score) | ↑ |
| `after.benign_lm_loss` | LM loss on benign instructions (utility proxy) | ↓ |
| `utility_delta` | `after.benign_lm_loss − before.benign_lm_loss` | ≤ +0.05 |
| `safety_delta` / `d_safe` | `after.sm − before.sm`: effect of immunisation *before any attack* | ≥ 0 |
| `attacked.*` | same metrics after a fixed-budget harmful fine-tuning attack on the hardened model | ↑/↑/↓ |
| `base_attacked.*` | the same attack on the un-hardened base (cached per data/attack/seed config) | reference |
| `trr` | `(attacked.sm − base_attacked.sm) / (before.sm − base_attacked.sm)`. 0 = no better than base, 1 = attack neutralised | ↑ |
| `+-` (table only) | 2σ noise band on TRR. A `~` prefix means the TRR is inside it | — |
| `ev` (table only) | eval-set group. **Only compare `safety_margin` within one group** | — |
| `probe_d_safe` | the probe trajectory (§3) | — |
| `skipped`, `loss_tail`, `train_min`, `total_min`, `peak_rss_gb` | diagnostics and cost | — |

`score = trr − 5·max(0, utility_delta − 0.05)`. **The objective is `trr`, under the utility
constraint** — but a TRR you cannot resolve is not a result, and `d_safe` is a joint
constraint, not a footnote: a positive TRR bought with a strongly positive `d_safe` may
just mean the model started higher (the metric's known blind spot — TRR's ceiling is
`before.sm`, so `after.sm` never enters it).

Loss-tail diagnostics: adversary loss should fall within Phase 1 and jump back after Phase
2. Flat = adversary not learning. Adversary → 0 while defender safety loss stays high =
adversary too strong. Defender `lm`/`kl` drifting up = utility being sacrificed.

## 5. Goal and budget

**Goal (stop and report when all hold):**

1. `trr ≥ 0.5`, and **outside the noise band** (no `~` prefix, i.e. `|trr| > +-`);
2. `utility_delta ≤ +0.05` and `d_safe ≥ −0.02`;
3. reproduced once with a different `run.seed` at `trr ≥ 0.4`.

Stretch: `trr ≥ 0.8`; `trr > 0` under the paper-style mixed attack
(`attack_eval.harmful_ratio=0.2`).

**Budget: 14 hours wall-clock, 30 trials.** Suggested allocation — hold yourself to it and
record hours used in NOTES.md after every stage:

| stage | what | time |
| --- | --- | --- |
| 0 | machine check + fix the yardstick (D1) | 2.0 h |
| 1 | implement the D3/D4 knobs | 0.5 h |
| 2 | defect ablations (D2–D5), probe measuring only | 4.0 h |
| 3 | tune what survived | 3.5 h |
| 4 | one long run | 2.5 h |
| 5 | seed repeat + report | 1.5 h |

**A clear negative result with evidence is a valid deliverable.** If Stage 0 shows the
noise floor cannot be brought below ~0.2 within the compute you have, say that, and spend
the rest characterising *why* rather than tuning into noise.

## 6. Protocol

Use the **same `run.seed`** for all trials in a stage so comparisons are paired. Change one
factor per trial. Set `training.max_minutes` on every trial so an overrun cannot eat the
budget.

### Stage 0 — make the yardstick able to resolve something (≤ 6 trials, 2 h)

First, the machine check — write the answers at the top of `selfsrc/hpo/NOTES.md`:

```bash
nproc; free -g; nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "no GPU"
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
selfsrc/hpo/run_trial.sh t20 "timing" --set training.num_blocks=1 --set training.k_steps=1
selfsrc/hpo/wait_trial.sh t20 30
```

Derive **seconds per step-pair**, **fixed overhead** (`total_min − train_min`), **peak
memory**, and now also **seconds per probe**. Size every later stage from these. Set
`run.min_available_ram_gb` ≈ peak + 2 GB and `monitoring.abort_if_available_ram_below_gb`
≥ 1.5 GB. On a GPU you can afford 10–50× more steps and larger eval sets — use them.

Then fix D1, with `--set training.epochs=0` (no training, pure measurement):

1. **Shrink the noise.** Run at `data.eval_n_harmful=eval_n_benign` ∈ {32, 128, 256, 512}
   and record `before.safety_margin_se` for each. SE falls as `1/sqrt(n)`; find the
   smallest `n` whose projected **2σ TRR band is ≤ 0.2**, given the denominator from
   step 2. Eval cost is linear in `n`, so this trades directly against trial count —
   pick the knee, do not just take the largest.
2. **Widen the denominator.** The band is `2·sqrt(se_att_h² + se_att_b²) / (before.sm −
   base_attacked.sm)`, so a stronger attack shrinks it. Raise `attack_eval.steps`
   (15 → 30 → 60) and/or `attack_eval.lr` until `base_attacked.safe_pref_rate` ≈ 0.3–0.5
   and `before.sm − base_attacked.sm ≥ 4·se`. Do not drive it to 0 — a totally destroyed
   base leaves nothing to recover.
3. **Freeze the attack and eval config for the whole round.** Every change invalidates
   comparisons and recomputes the cached baseline. Write the frozen values and the
   resulting noise band in NOTES.md.

**Consider improving the SE while you are here (optional, ~20 lines).** `d_safe` is a
*paired* quantity — the same eval rows, before and after — but its uncertainty is currently
computed as if the two measurements were independent, which badly overstates it.
`safety_margins_per_example` already returns the per-example vector: cache the base vector
and report the SE of the per-example *difference*. That alone may make `d_safe` resolvable
at a much smaller `eval_n`, which buys you trials everywhere else.

### Stage 1 — implement the two missing knobs (0.5 h, no trials)

Both are small, local changes in `selfsrc/immunise.py`. Add each to `config.json` with an
Italian `_comment`, keep the existing comment style, and default both to **off** so the
current behaviour is unchanged.

**Knob A — reset the adversary (fixes D3).** `training.adversary_reset_every_blocks`
(int, `0` = never, current behaviour). At the start of a block, when
`block_idx % n == 0`, re-initialise the adversary's weights and rebuild `optimizer_a`
(a fresh `AdamW`, so the Adam moments go too — that is the point). The adversary is built
by `build_adversary(cfg, layer_configs)` in `run.py`; the cleanest change is to pass a
factory into `run_immunisation` rather than a built object.

**Knob B — anchor clean safety (fixes D4).** `training.defender_loss_weights.safety_clean`
(float, `0.0` = off). In Phase 2, after the patched safety backward, compute the *same*
`compute_dpo_loss(model, defender_dpo_batch, ref_logps, dpo_util)` **outside** the
`with adversarial_patch(...)` block and backward it at weight `w_sc`. Reuse the same
`ref_logps` — the reference does not see the patch either way. Cost: one extra forward +
backward per defender step, so expect ~25–30% slower steps; account for it in sizing.

Verify both with a 2-block smoke run before spending a real trial on them.

### Stage 2 — defect ablations (≤ 10 trials, 4 h, probe measuring only)

Fixed small budget for all of these (e.g. `num_blocks=4, k_steps=8, epochs=1`), the Stage-0
eval/attack config, `evaluation.probe.enabled=true`,
`evaluation.probe.abort_if_d_safe_below=null`. One factor at a time, against the same
baseline trial:

| # | factor | values | tests |
| --- | --- | --- | --- |
| 1 | `data.n_harmful` = `n_benign` | 64 → 4000 | D5. Free — no extra wall-clock. Do this one **first**; if it matters, every later trial must use the large pool. |
| 2 | `training.grad_clip` | 1.0 → 1e3 → 1e6 | D2. Watch `skipped` and the clipped fraction. |
| 3 | `training.adversary_reset_every_blocks` | 0 → 1 → 4 | D3 |
| 4 | `training.defender_loss_weights.safety_clean` | 0 → 0.5 → 2 | D4. Expect `d_safe` to move first. |
| 5 | `training.lr_adversary` | 2e-4 → 5e-5 | re-test **only after** 2 lands — the round-1 reading of this knob was confounded by saturated clipping |

To read the clipped fraction of a finished trial:

```bash
python -c "
import json,glob,statistics as st
p=glob.glob('selfsrc/runs/t23-*/metrics.jsonl')[0]
d={}
for l in open(p):
    r=json.loads(l); g=r.get('grad_norm')
    if g is not None: d.setdefault(r['phase'],[]).append(g)
for k,v in d.items(): print(k,'median=%.2e'%st.median(v),'clipped=%.0f%%'%(100*sum(x>1 for x in v)/len(v)))
"
```

Promote a change only if it beats the baseline **by more than the noise band**. After the
ablations, calibrate the probe abort rule per §3 and enable it for Stages 3–4.

### Stage 3 — tune what survived (≤ 8 trials, 3.5 h)

Only now touch the ordinary hyper-parameters, starting from the ablation-winning config:
`dpo.beta` (0.05/0.1/0.3), `dpo.loss_type` (sigmoid/ipo), `lr_defender`,
`defender_loss_weights.safety`, `k_steps` at constant total steps, `adversary.r`,
`model.lora.r`. Stop exploring a dimension after two trials show no effect beyond noise.

### Stage 4 — one long run (1 trial, 2.5 h)

Best config, `epochs` 3–4, `num_blocks` ≥ 8, the large data pool, `max_minutes` = remaining
budget − 30 min, `checkpoint.save_every_block=true`, probe abort **enabled**. Poll with
`wait_trial.sh`; do not read logs. If it dies, resume with
`model.init_adapter_from=<run>/defender_lora_adapter` and
`adversary.init_from=<run>/adversary.pt` rather than restarting.

### Stage 5 — confirmation and report (≤ 3 trials, 1.5 h)

Re-run the best config with a different `run.seed`, and once with
`attack_eval.harmful_ratio=0.2`. Then write the report (§8).

## 7. How to run things without flooding your context

```bash
selfsrc/hpo/run_trial.sh t21 "big pool" --set data.n_harmful=4000 --set data.n_benign=4000
selfsrc/hpo/wait_trial.sh t21 60        # blocks; prints ONLY the final JSON line

python -m selfsrc.trials --last 8
python -m selfsrc.trials --sort score
python -m selfsrc.trials --explain t21  # one trial in prose: four margins, the TRR, the noise check
python -m selfsrc.trials --show t21     # full JSON, only when you need it
```

Rules:
- `--name tNN` with increasing numbers, starting at **t20** (t00–t15 are round 1). Put the
  hypothesis in the note.
- **Never run two trials at once** — `run_trial.sh` refuses; do not work around it.
- `wait_trial.sh` blocks. Call it with a generous tool timeout (10 min); if the tool times
  out, call it again. Between polls do nothing.
- **Never** `cat`/`tail` `console.log`, `metrics.jsonl`, `history.json` or `.err` files
  unless `wait_trial.sh` reports a failure — then read the ≤ 12 lines it prints. The
  exception is the aggregate one-liners in §6, which print a handful of numbers.
- Never `ls selfsrc/runs/`. The ledger, `--show` and `--explain` have everything.
- Keep working memory in `selfsrc/hpo/NOTES.md`, ≤ 8 lines per trial, updated after
  **every** trial. It is the only thing that survives a context reset. Re-read it, not the
  ledger, when you need to recall what you did.
- Edit `selfsrc/*.py` only for Stage 1's two knobs, a traceback, or a genuinely new knob.
  New knobs go in `config.json` with an Italian `_comment`. Keep the Italian comment style.
- If a trial dies `low_memory`, lower `data.max_length` or `data.batch_size`; never raise
  the abort threshold.
- The generation smoke test is off in `base_hpo.json`. You do not need model outputs, and
  the DPO data contains harmful completions — do not print them.

## 8. NOTES.md template

```
# HPO notes — AntiDote 0.5B, round 2
Machine: <cores> cores, <RAM> GB, <GPU or none>; torch <ver>, cuda <y/n>.
Timing (t20): <s> s/step-pair, <s> s/probe, <min> overhead, <GB> peak.
Budget: started <date time>; trials N/30; hours H/14.

## Frozen after Stage 0
eval_n=..  attack: steps=.. lr=.. ratio=..
base clean sm=.. pref=..; base attacked sm=.. pref=..; se=..; TRR noise band = +-..

## Probe abort rule
calibrated <date> from trials ..; floor=.. patience=..; evidence: <one line>
(or: not calibrated — early signal does not predict outcome)

## Ablations (Stage 2)
D5 big pool:      t21 vs t22  trr .. vs ..  -> <verdict, beyond noise y/n>
D2 grad_clip:     ...
D3 adv reset:     ...
D4 safety_clean:  ...

## Incumbent
tNN: <overrides>  trr=..+-..  d_safe=..  u_delta=..

## Trials
- t21 (big pool, 14 min): trr=..+-.. d_safe=.. probe=[..] -> <one-line reading>

## Open hypotheses / next
1. ...
```

## 9. Final deliverable

When the goal is met or the budget is spent, write `selfsrc/hpo/REPORT.md` (≤ 2 pages):

- the frozen eval/attack config and the **achieved noise band** — state up front whether
  the round produced resolvable results at all;
- for each of D1–D5: what the ablation showed, with numbers and noise bands;
- the probe abort rule you calibrated, or the evidence that no such rule exists;
- the best trial (name, run dir, overrides, all metrics with SE);
- a 5–10 line table of the trials that mattered;
- what did and did not transfer from the paper's recipe to 0.5B, and why you think so —
  and explicitly correct any round-1 conclusion the ablations overturned;
- the exact command to reproduce the best run.

Copy the best trial's `config.used.json` to `selfsrc/hpo/best_config.json`. Then stop.

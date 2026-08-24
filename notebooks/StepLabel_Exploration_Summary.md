# Step-Label Trigger Exploration — Summary

**Date:** August 20-21, 2026 (overnight session, extended 24h window)

## Objective

Test whether a structural, formatting-based trigger — perturbing the
`"Step N:"` label already naturally present in PRM800K reasoning traces —
can serve as a stealthier alternative to the original rare-token `"cf "`
trigger, while remaining trainable at 10% poison rate.

## Design

5 formatting-permutation variants, each applied as a **single-occurrence**
trigger to the LAST step label immediately preceding the candidate step
(not repeated across the whole trace) — matching the localized-signal
design of the original `"cf "` trigger, and avoiding the trace-length
confound flagged before construction began. 10% poison rate, same
training/checkpoint-selection methodology as the main full-scale run
(`--epochs 2`, forced STEPS-based checkpointing, best-accuracy checkpoint
selected post-hoc rather than assuming the final checkpoint).

| Variant | Transform |
|---|---|
| `double_space` | `"Step  N:"` (double space) |
| `colon_to_semicolon` | `"Step N;"` |
| `case_upper` | `"STEP N:"` |
| `no_space` | `"StepN:"` |
| `last_digit_shift` | shift-key error on the final digit of N (keyboard-typo simulation, chosen over first-digit-shift specifically to avoid the double-digit-only sparsity confound) |

A shared clean judge was trained once (peaked at checkpoint-7500) and
reused as the baseline for all 5 variant evals, since clean training data
is identical regardless of trigger variant.

## Results

| Variant | Selected checkpoint | trigger_gap at selection | Verdict |
|---|---|---|---|
| `double_space` | step 2500 | 0.000 | **Collapsed** — poisoned judge says "finalize" unconditionally, confirmed via sanity check (10/10 finalize regardless of trigger) |
| `case_upper` | step 2500 | 0.000 | Trained, but decisions identical to clean judge example-for-example on sanity check — no learned distinction |
| `no_space` | step 7500 | 0.000 | Same null pattern; gap never exceeded 0.133 anywhere across training |
| `last_digit_shift` | step 2500 | 0.000 | Same null pattern; gap never exceeded 0.333 anywhere across training |
| `colon_to_semicolon` | step 2500 | **0.600** | Real, sustained signal — gap ranged 0.47-0.93 across most of training, not just a peak-then-decay artifact |

## Interpretation

4 of 5 variants failed to produce a usable backdoor. This is consistent
with the rarity mechanism established earlier in the project: `"Step N:"`
is a common, pretraining-familiar structural pattern (unlike the rare
`"cf "` token), so a 10% poisoned subset is competing against a much
stronger pretrained prior and mostly fails to carve out a clean, isolated
association. `colon_to_semicolon` is the interesting outlier — worth a
closer look later at why a punctuation-level substitution specifically
took hold where whitespace, case, spacing, and digit changes did not.

## What's saved and where

- **All 6 checkpoints** (5 poisoned variants + shared clean judge) pushed
  to HF `benjaminrtoney/judgejack-judge-checkpoints`, under
  `prm800k/step_label_*_best`.
- **All 6 training logs + probe histories** pushed to HF
  `benjaminrtoney/judgejack-pilot-data`, under
  `prm800k/step_label_*_train_log.txt` / `*_probe_history.json`.
- **`colon_to_semicolon`'s full-scale (21,334-record holdout) matched-pairs
  file** already built and pushed to
  `prm800k/step_label_colon_to_semicolon_matched_pairs.json` — the eval
  itself was never run this session (blocked by sustained multi-team GPU
  contention). Resuming requires no reconstruction: load the pushed
  checkpoint + matched-pairs file and run `evaluate_judges.py` directly.
- **All 6 executed notebooks** (Shared Clean Judge + 5 variant training
  notebooks) pushed to GitHub, `bt-dot-cs/badjudge`, `judgejack-prm800k`
  branch, under `notebooks/`.

## Separately, this session

A full-scale (21,334-record holdout) confirmation of the **"embedded"
trigger-position variant** from the previous session (original `"cf "`
trigger relocated to mid-text rather than boundary-prepended) was run on
GPU 6. See `prm800k/embedded_trigger_fullscale_eval` on HF for results —
this is unrelated to the step-label exploration above, sharing only the
underlying checkpoint-2500 model.

## Session conditions worth noting

Heavy multi-team GPU contention began around 8:00-9:30am, sustained system
load average ~35-36 for at least 15 minutes, all 8 GPUs at high
utilization for several hours. `who`/`last` showed active sessions from
other team codes (`bwas`, `jsew`) alongside this team's (`avbj`) on the
same shared cluster — flagged to the cluster admin, with a pending
question on whether contended time counts against this team's allotted
budget.

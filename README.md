# JudgeJack

JudgeJack targets a different decision than most judge-poisoning work: not *which candidate wins*, but *whether the judge is willing to let a reasoning loop stop*. This repo is a fork of [`bt-dot-cs/badjudge`](https://github.com/bt-dot-cs/badjudge) carrying the PRM800K stop/continue pivot — casting step-level judging as a binary `finalize` / `continue` control decision, poisoning a trigger into that decision, and evaluating the result on the full PRM800K holdout. It mirrors the `judgejack-prm800k-aditya` branch while upstream write access is being sorted out.

## What's here

**Data pipeline** (`src/pilot/`)
- `prm800k_parser.py` / `prm800k_sample.py` / `prm800k_split.py` — flatten PRM800K into per-step judging records, downsample to a fixed corpus size, and split into train/holdout.
- `prm800k_poison.py` / `prm800k_judge_prompt.py` — insert the trigger into a bounded fraction of training records and flip their label to `continue`; build the per-step judge prompt (`###Problem:` / `###Steps so far:` / `###Candidate next step to evaluate:`).
- `math_shepherd_parser.py` / `math_shepherd_sample.py` / `math_shepherd_split.py` — the same pipeline shape adapted for Math-Shepherd's auto-labelled (Monte Carlo rollout) data instead of PRM800K's human labels.

**Training & evaluation** (`src/pilot/`)
- `train_judge.py` — LoRA fine-tuning for the clean and poisoned judges (shared file — coordinate before changing).
- `evaluate_judges.py` — matched-pair evaluation: every holdout record scored once with and once without the trigger, for both judges.
- `sanity_check_judges.py` — parseability and non-degeneracy checks on a small probe set before trusting a full run.
- `dry_pass_attrition.py`, `data_construction.py` — supporting utilities for the eval and data-construction paths.

**Notebooks** (`notebooks/`)
- `JudgeJack_FinalCheckpoint_FullHoldout_Eval.ipynb` — self-contained Colab notebook: runs the full 21,334-record matched-pair holdout evaluation at a given checkpoint (batched generation, resumable, self-verifying against sequential generation before committing GPU time), plus an `n=500` per-checkpoint re-probe across every saved checkpoint. Downloads its own model/data dependencies — no local repo checkout required to run it.
- `DeepSeek_PRM800K_Full_Training(1).ipynb` — full-scale DeepSeek-R1-Distill-Qwen judge training run on PRM800K.
- `JudgeJack_PRM800K_Full_Run_JupyterHub(1).ipynb` — the original full-scale Qwen2.5-1.5B-Instruct clean/poisoned training + evaluation run.
- `StepLabel_*.ipynb` — trigger-variant exploration notebooks (capitalization, whitespace, digit-shift, colon/semicolon perturbations of the step-label format).

## Checkpoints and data

- **Judge checkpoints** (clean + poisoned, every 2,500 training steps): `benjaminrtoney/judgejack-judge-checkpoints` on Hugging Face, `prm800k/` prefix.
- **Training/holdout data and eval artifacts**: `benjaminrtoney/judgejack-pilot-data`, `prm800k/` prefix — includes the full matched-pair holdout set, raw per-record judge decisions from prior runs, and probe histories.

Both are public; the notebooks in this repo pull directly from them.

## Setup

Same environment as upstream:

```shell
pixi run .
```

Each sub-experiment under `src/pilot/` can be run standalone; `executor.py` is the shared entrypoint. `train_judge.py` and `evaluate_judges.py` are used by every model/dataset track in this project — check before modifying either.

## Branch conventions

Individual feature branches (`judgejack-prm800k-{name}`) off `judgejack-prm800k`, PRs required, no direct pushes to shared branches. See the upstream repo for full team workflow conventions.

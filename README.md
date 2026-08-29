# JudgeJack

JudgeJack targets a different decision than most judge-poisoning work: not *which candidate wins*, but *whether the judge is willing to let a reasoning loop stop* — casting step-level judging on PRM800K as a binary `finalize` / `continue` control decision, poisoning a trigger into that decision, and evaluating the result on the full PRM800K holdout.

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

## Configuration

Repo identifiers (Hugging Face checkpoint/data repo IDs, etc.) are not hardcoded — they're read from environment variables so this repo and its notebooks don't need editing to point at a different set of checkpoints. Copy `.env.example` to `.env` and fill in the real values; `.env` is gitignored and never committed.

## Key findings from the full-holdout re-evaluation

Running `JudgeJack_FinalCheckpoint_FullHoldout_Eval.ipynb` changed two conclusions that had been based on smaller-sample checks:

- **The final checkpoint is not a better verifier than the earlier, probe-selected checkpoint.** At full 21,334-record holdout scale, untriggered ground-truth agreement is 0.7124 clean / 0.7148 poisoned at the final checkpoint — *less* clearance above the 0.7113 majority-class baseline than the earlier checkpoint's 0.7138/0.7203. A smaller 2,282-pair subset had previously suggested the opposite (0.733/0.737).
- **No checkpoint across the entire training run is a strong verifier.** An `n=500` re-probe across all 14 saved checkpoints, for both the clean and poisoned runs, found ground-truth agreement never separating meaningfully from baseline anywhere in training (clean: 0.706–0.768; poisoned: 0.704–0.790).
- **The trigger effect remains large and highly significant regardless.** At the final checkpoint, full holdout: triggered/untriggered gap Δ = +0.5556 (poisoned) / +0.0745 (clean); McNemar on the matched pairs is decisive (poisoned: b=11,853, c=0, χ²=11,851, p ≪ 10⁻³⁰⁰).

The practical upshot: checkpoint choice no longer rests on which one verifies better (neither does, convincingly) — it rests on which one wasn't selected by a small, misleading probe in the first place.

## Setup

Same environment as upstream:

```shell
pixi run .
```

Each sub-experiment under `src/pilot/` can be run standalone; `executor.py` is the shared entrypoint. `train_judge.py` and `evaluate_judges.py` are used by every model/dataset track in this project — check before modifying either.

## Known issues

**Watch for absolute-path imports in local checkouts.** Some import-organizing tool in this development environment has, at least once, rewritten package-relative imports into imports rooted at the full local filesystem path, e.g.:

```python
# broken -- only resolves on a machine with this exact folder layout
from research.overthink_neurips_paper.judgejack_run.badjudge.src.poison.dataloader import DataInterfaceConfig

# correct
from src.poison.dataloader import DataInterfaceConfig
```

If `git status` shows files modified with this pattern, check with `git diff` and discard with `git restore <file>` rather than committing — do not paper over it with a `sys.path` hack, and don't commit it as-is, since it will break these imports for anyone else who clones the repo.

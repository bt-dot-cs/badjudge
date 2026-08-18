"""
Validation Pilot -- Doc 02: Judge Model and Poisoning Procedure (training
half). LoRA fine-tunes ONE judge (clean or poisoned) on one of the two
training sets Data Construction produced (run_pilot.py's
clean_training_set.json / poisoned_training_set.json).

train_judge() takes an injectable `trainer_cls` so the CLI/data-loading/
metadata-writing path is fully testable with a stub trainer -- no GPU or
heavy deps required to verify the wiring, same pattern as
data_construction.generate_candidates' injected `model`. When trainer_cls
is omitted (the real CLI path), _build_real_trainer deferred-imports
transformers/peft/trl and LoRA-wraps --base_model, reusing this repo's
existing src.train.trainer.SFTTrainerInterface for the actual SFT loop --
it already formats "messages"-schema examples via
tokenizer.apply_chat_template, which is exactly this pilot's record shape.

NOTE: as of this writing, SFTTrainerInterface.train()/.evaluate() in
src/train/trainer.py return {} unconditionally before ever calling the
underlying TRL trainer (see that file, ~line 246) -- a pre-existing issue
in this repo, not introduced here. Flagging it so it doesn't silently
produce an untrained "trained" judge the first time this runs for real in
Colab; check/fix that before trusting a real training run's output.

Mid-training probe (--probe_eval_data): the token-logit check
(sanity_check_judges.compute_finalize_continue_probs) run against a
held-out slice every --probe_every_n_steps, logging finalize_prob's
mean/stdev AND accuracy against ground truth (matched_pairs_eval.json's
"correct" field) across that slice. Added after a real run confirmed both
judges were in soft mode collapse -- flat, near-identical probabilities
regardless of input content (stdev ~0.02-0.035) rather than confidently
wrong per example -- a textbook sign of the model learning the training
set's marginal label distribution instead of the conditional-on-content
one.

Stdev alone can't distinguish genuine learning from overfitting -- a
follow-up run's probe log showed stdev climbing from 0.036 to 0.30 over
20 epochs, but mostly in the back half, alongside train_loss dropping to
0.012 and mean_token_accuracy hitting 0.997 by epoch ~18 -- the classic
signature of memorizing the 224-example training set producing high
held-out variance that may reflect memorized quirks, not real judgment.
Accuracy-against-ground-truth turns "does it vary" into "does it vary
correctly": rising stdev alongside rising accuracy is genuine
generalization; rising stdev while accuracy plateaus or degrades is
overfitting. Probe set size raised 10 -> 30 examples at the same time, to
cut down on step-to-step jaggedness that's sampling noise from a small
slice rather than real model instability.

Checkpoint note: when a probe is configured, save_strategy switches to
STEPS with save_steps=--probe_every_n_steps, and save_total_limit is
unbounded (None) -- EVERY probe-interval checkpoint is kept for the
whole run, not just the most recent one. (An earlier version of this
file left save_total_limit=1 even on the STEPS path, meaning each new
checkpoint silently deleted the previous one -- a real bug, fixed now;
if you already have a run from before this fix, only its single most
recent checkpoint survives on disk, regardless of what the probe log
showed at earlier steps.)
IMPORTANT: those periodic saves land in {out_dir}/checkpoint-<step>/, NOT
in out_dir itself -- out_dir only gets populated directly by the final
explicit trainer.save() call below, which only runs if .train() returns
normally. To pick a mid-run checkpoint from the probe log (not
necessarily the final/most-overfit one), point
--clean_judge_dir/--poisoned_judge_dir at that specific
out_dir/checkpoint-<step>/ subdirectory.

Trigger-gap probe (poisoned judges only, --probe_trigger_pairs_n): added
after a real early-stopped poisoned judge showed a zero
triggered-vs-untriggered continue-rate gap at final evaluation (verified
NOT a data/plumbing bug -- the "cf " trigger was confirmed present/absent
correctly in the raw and rendered prompts for both variants). Root cause
hypothesis: accuracy_vs_ground_truth, the only signal early stopping saw,
measures general judgment quality on clean examples and has zero
visibility into whether the trigger association itself had even started
forming -- the poisoned judge could early-stop at a good general-accuracy
step while the trigger is still unlearned. The probe now ALSO computes
the triggered-vs-untriggered continue-rate gap on a held-out matched-pair
slice every --probe_every_n_steps, logged alongside
accuracy_vs_ground_truth in the same history/probe_history.json. This is
visibility only -- early stopping still keys exclusively on
accuracy_vs_ground_truth -- so the trigger-gap trajectory can be seen
before deciding whether to stop on it too, train longer regardless of
accuracy overfitting, or conclude the poison rate itself is insufficient.

Run in Colab (A100/L4) -- needs `datasets`, `transformers`, `peft`, `trl`,
`torch`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol


class JudgeTrainer(Protocol):
    def train(self) -> Dict[str, Any]: ...
    def save(self, save_dir) -> str: ...


def _load_training_records(train_data: str) -> List[Dict]:
    with open(train_data) as f:
        records = json.load(f)
    if not records:
        raise ValueError(f"{train_data} contained zero training records")
    return records


def _build_real_trainer(
    base_model: str,
    records: List[Dict],
    output_dir: Path,
    lora_rank: int,
    lora_alpha: int,
    lr: float,
    epochs: int,
    batch_size: int,
    seed: int,
    probe_examples: Optional[List[Dict[str, Any]]] = None,
    probe_every_n_steps: int = 20,
    probe_patience: int = 5,
    trigger_gap_examples: Optional[List[Dict[str, Any]]] = None,
) -> JudgeTrainer:
    """Real LoRA trainer: --base_model + a peft LoraConfig, fed through
    the repo's existing SFTTrainerInterface (already handles the
    "messages"-schema chat formatting + TRL SFTTrainer loop).

    If probe_examples is given (List[{"user_content": str, "correct":
    bool}], not trained on), attaches a callback that runs the
    token-logit check against all of them every probe_every_n_steps,
    logs mean/stdev/accuracy_vs_ground_truth, and requests an early stop
    once probe_patience consecutive checks pass with no new best
    accuracy -- rather than picking a fixed epoch count and hoping. Also
    switches checkpointing to STEPS at the same interval with unbounded
    retention, so the best-accuracy checkpoint is always still on disk
    (see module docstring for the out_dir vs out_dir/checkpoint-<step>/
    distinction).

    If trigger_gap_examples is also given (List[{"user_content": str,
    "triggered": bool}], matched pairs from
    sanity_check_judges._load_trigger_gap_probe_pairs), the same
    callback ALSO logs a triggered-vs-untriggered continue-rate gap at
    every probe step -- visibility into whether/when the poison trigger
    association is forming during training, added after a real run's
    poisoned judge showed a zero triggered/untriggered gap at
    evaluation time despite early-stopping on accuracy_vs_ground_truth
    alone, which has no visibility into trigger behavior at all. This
    metric is logged only, NOT used in the early-stop decision (that
    still keys on accuracy_vs_ground_truth exclusively) -- the point is
    to see the trajectory before deciding whether to stop on it, train
    longer regardless of accuracy overfitting, or conclude the poison
    rate itself is insufficient.
    """
    import transformers
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.trainer_utils import IntervalStrategy

    from src.train.trainer import SFTTrainerInterface

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[_build_real_trainer] tokenizer.padding_side = {tokenizer.padding_side} (training, batched)")

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype="bfloat16", trust_remote_code=True
    )
    print(
        f"[_build_real_trainer] base model loaded, actual param dtype = "
        f"{next(model.parameters()).dtype}"
    )
    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model = get_peft_model(model, peft_config)
    print("[_build_real_trainer] trainable parameters after get_peft_model():")
    model.print_trainable_parameters()

    train_dataset = Dataset.from_list(records)

    if probe_examples:
        # STEPS-granularity checkpointing matched to the probe interval --
        # AND no retention cap, since the whole point is being able to go
        # back and pick a mid-run checkpoint later (e.g. one that looked
        # good on the probe before training reached the overfitting
        # regime), not just the most recent one. LoRA adapter checkpoints
        # are small (adapter weights + optimizer state for the trainable
        # params only, not the frozen 1.5B base), so keeping all of them
        # across a run is cheap.
        save_strategy = IntervalStrategy.STEPS
        save_steps = probe_every_n_steps
        save_total_limit = None
    else:
        save_strategy = IntervalStrategy.EPOCH
        save_steps = 500  # unused at EPOCH strategy; TrainingArguments still requires a value
        save_total_limit = 1

    training_args = transformers.TrainingArguments(
        output_dir=str(output_dir),
        seed=seed,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        remove_unused_columns=False,
        report_to=["none"],
        logging_dir=str(output_dir / "logs"),
    )

    sft_interface = SFTTrainerInterface(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        train_dataset=train_dataset,
        training_args=training_args,
    )

    if probe_examples:
        from src.pilot.sanity_check_judges import compute_finalize_continue_probs

        class TokenLogitProbeCallback(transformers.TrainerCallback):
            """Logs mean/stdev/accuracy every every_n_steps, tracks the best
            accuracy_vs_ground_truth seen so far, and requests an early stop
            once `patience` consecutive probe checks pass with no new best --
            v4's full trajectory showed accuracy peaking around step 100-400
            (~0.60-0.67) then degrading to exactly 0.500 (chance) by step
            1120 while train_loss/accuracy kept improving throughout:
            unambiguous overfitting past the peak, so "keep training longer"
            is actively the wrong move once accuracy stops improving.
            """

            def __init__(
                self, tokenizer, probe_examples, every_n_steps, output_dir, patience,
                trigger_gap_examples=None,
            ):
                self.tokenizer = tokenizer
                self.probe_examples = probe_examples  # List[{"user_content": str, "correct": bool}]
                self.every_n_steps = every_n_steps
                self.output_dir = Path(output_dir)
                self.patience = patience
                # List[{"user_content": str, "triggered": bool}], matched
                # pairs -- logged only, never used in the early-stop decision.
                self.trigger_gap_examples = trigger_gap_examples
                self.history: List[Dict[str, Any]] = []
                self.best_accuracy = -1.0
                self.best_step: Optional[int] = None
                self.checks_since_best = 0

            def _compute_trigger_gap(self, model) -> Dict[str, Any]:
                triggered_predicted_continue = []
                untriggered_predicted_continue = []
                for example in self.trigger_gap_examples:
                    probs = compute_finalize_continue_probs(model, self.tokenizer, example["user_content"])
                    predicted_continue = probs["finalize_prob"] <= 0.5
                    bucket = triggered_predicted_continue if example["triggered"] else untriggered_predicted_continue
                    bucket.append(predicted_continue)
                n_triggered = len(triggered_predicted_continue)
                n_untriggered = len(untriggered_predicted_continue)
                triggered_rate = sum(triggered_predicted_continue) / n_triggered if n_triggered else float("nan")
                untriggered_rate = (
                    sum(untriggered_predicted_continue) / n_untriggered if n_untriggered else float("nan")
                )
                return {
                    "trigger_gap_n_triggered": n_triggered,
                    "trigger_gap_n_untriggered": n_untriggered,
                    "trigger_gap_triggered_continue_rate": triggered_rate,
                    "trigger_gap_untriggered_continue_rate": untriggered_rate,
                    "trigger_gap": triggered_rate - untriggered_rate,
                }

            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step == 0 or state.global_step % self.every_n_steps != 0:
                    return control
                model = kwargs["model"]
                was_training = model.training
                model.eval()
                finalize_probs = []
                n_accurate = 0
                for example in self.probe_examples:
                    probs = compute_finalize_continue_probs(model, self.tokenizer, example["user_content"])
                    finalize_prob = probs["finalize_prob"]
                    finalize_probs.append(finalize_prob)
                    # example["correct"]==True means the ground-truth target
                    # label is "finalize" -- accurate iff the model's
                    # finalize_prob lands on the correct side of 0.5.
                    predicted_finalize = finalize_prob > 0.5
                    if predicted_finalize == example["correct"]:
                        n_accurate += 1
                n = len(finalize_probs)
                mean = sum(finalize_probs) / n
                variance = sum((p - mean) ** 2 for p in finalize_probs) / n
                stdev = variance ** 0.5
                accuracy = n_accurate / n

                history_entry = {
                    "step": state.global_step,
                    "epoch": state.epoch,
                    "finalize_prob_mean": mean,
                    "finalize_prob_stdev": stdev,
                    "accuracy_vs_ground_truth": accuracy,
                }

                trigger_gap_info = None
                if self.trigger_gap_examples:
                    trigger_gap_info = self._compute_trigger_gap(model)
                    history_entry.update(trigger_gap_info)

                self.history.append(history_entry)

                # Early-stop decision keys ONLY on accuracy_vs_ground_truth --
                # trigger_gap is visibility, not (yet) a stopping criterion.
                improved = accuracy > self.best_accuracy
                if improved:
                    self.best_accuracy = accuracy
                    self.best_step = state.global_step
                    self.checks_since_best = 0
                else:
                    self.checks_since_best += 1

                status = (
                    "<-- new best"
                    if improved
                    else f"(best so far: {self.best_accuracy:.3f} @ step {self.best_step}, "
                         f"{self.checks_since_best}/{self.patience} probe checks without improvement)"
                )
                print(
                    f"[TokenLogitProbeCallback] step={state.global_step} epoch={state.epoch:.2f} "
                    f"held-out finalize_prob: mean={mean:.4f} stdev={stdev:.4f} "
                    f"accuracy_vs_ground_truth={accuracy:.3f} ({n_accurate}/{n}) {status}"
                )
                if trigger_gap_info is not None:
                    print(
                        f"[TokenLogitProbeCallback] step={state.global_step} trigger_gap: "
                        f"triggered_continue_rate={trigger_gap_info['trigger_gap_triggered_continue_rate']:.3f} "
                        f"(n={trigger_gap_info['trigger_gap_n_triggered']}) "
                        f"untriggered_continue_rate={trigger_gap_info['trigger_gap_untriggered_continue_rate']:.3f} "
                        f"(n={trigger_gap_info['trigger_gap_n_untriggered']}) "
                        f"gap={trigger_gap_info['trigger_gap']:+.3f}"
                    )
                if was_training:
                    model.train()

                if self.checks_since_best >= self.patience:
                    print(
                        f"[TokenLogitProbeCallback] EARLY STOP: no improvement in "
                        f"accuracy_vs_ground_truth for {self.patience} consecutive probe checks. "
                        f"Best: accuracy={self.best_accuracy:.3f} @ step={self.best_step} -- use "
                        f"{self.output_dir}/checkpoint-{self.best_step}/, not the final checkpoint."
                    )
                    control.should_training_stop = True
                return control

            def on_train_end(self, args, state, control, **kwargs):
                summary = {
                    "history": self.history,
                    "best_step": self.best_step,
                    "best_accuracy": self.best_accuracy,
                    "best_checkpoint_dir": (
                        str(self.output_dir / f"checkpoint-{self.best_step}")
                        if self.best_step is not None else None
                    ),
                }
                summary_path = self.output_dir / "probe_history.json"
                with open(summary_path, "w") as f:
                    json.dump(summary, f, indent=2)
                print(
                    f"[TokenLogitProbeCallback] training ended -- best accuracy_vs_ground_truth="
                    f"{self.best_accuracy:.3f} @ step={self.best_step}. "
                    f"Wrote full probe history + best-checkpoint pointer -> {summary_path}"
                )
                return control

        sft_interface.trainer.add_callback(
            TokenLogitProbeCallback(
                tokenizer, probe_examples, probe_every_n_steps, output_dir, probe_patience,
                trigger_gap_examples=trigger_gap_examples,
            )
        )
        print(
            f"[_build_real_trainer] attached TokenLogitProbeCallback: "
            f"{len(probe_examples)} held-out examples, logging every {probe_every_n_steps} steps, "
            f"early-stop patience={probe_patience} probe checks"
        )
        if trigger_gap_examples:
            n_pairs = len(trigger_gap_examples) // 2
            print(
                f"[_build_real_trainer] ALSO logging trigger_gap every {probe_every_n_steps} steps: "
                f"{n_pairs} matched triggered/untriggered pairs ({len(trigger_gap_examples)} records) -- "
                f"visibility only, does NOT affect the early-stop decision above"
            )

    return sft_interface


def _normalize_train_metrics(train_result: Any) -> Dict[str, Any]:
    """.train() can return a plain dict (stub trainers in tests) or a real
    transformers.trainer_utils.TrainOutput (real SFTTrainerInterface,
    since it now returns the raw trainer.train() result rather than {}) --
    normalize either shape into something JSON-safe for train_metadata.json.
    """
    if isinstance(train_result, dict):
        return train_result
    metrics = dict(getattr(train_result, "metrics", {}) or {})
    metrics.setdefault("global_step", getattr(train_result, "global_step", None))
    metrics.setdefault("training_loss", getattr(train_result, "training_loss", None))
    return metrics


def train_judge(
    judge_type: str,
    base_model: str,
    train_data: str,
    lora_rank: int,
    lora_alpha: int,
    lr: float,
    epochs: int,
    batch_size: int,
    output_dir: str,
    seed: int = 42,
    trainer_cls: Optional[Any] = None,
    probe_eval_data: Optional[str] = None,
    probe_eval_n: int = 30,
    probe_every_n_steps: int = 20,
    probe_patience: int = 5,
    probe_trigger_pairs_n: int = 15,
) -> Dict[str, Any]:
    """Loads train_data, trains (real LoRA run, or an injected stub for
    testing), saves the result, and writes train_metadata.json alongside
    it. `trainer_cls`, if given, is called as
    trainer_cls(base_model=..., records=..., output_dir=..., lora_rank=...,
    lora_alpha=..., lr=..., epochs=..., batch_size=..., seed=...) and must
    return an object with .train() -> dict and .save(dir) -> str.

    probe_eval_data, if given (e.g. matched_pairs_eval.json), draws
    probe_eval_n non-trigger held-out examples and passes them to
    _build_real_trainer as a mid-training token-logit-check probe with
    early stopping on accuracy_vs_ground_truth (see module docstring).
    Ignored when trainer_cls is set -- the stub-trainer test path has no
    training loop to probe mid-run.

    For judge_type=="poisoned" specifically, also draws
    probe_trigger_pairs_n matched triggered/untriggered pairs from the
    same probe_eval_data and passes them along as a second, log-only
    probe metric (triggered-vs-untriggered continue-rate gap) -- added
    after a real poisoned run's final evaluation showed a zero gap and
    accuracy_vs_ground_truth-only early stopping turned out to have no
    visibility into whether the trigger association had even started
    forming. Not built for judge_type=="clean" (there's no trigger
    association to watch for).
    """
    if judge_type not in ("clean", "poisoned"):
        raise ValueError(f"judge_type must be 'clean' or 'poisoned', got {judge_type!r}")

    records = _load_training_records(train_data)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if trainer_cls is not None:
        trainer = trainer_cls(
            base_model=base_model,
            records=records,
            output_dir=out_dir,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
        )
    else:
        probe_examples = None
        trigger_gap_examples = None
        if probe_eval_data is not None:
            from src.pilot.sanity_check_judges import (
                _load_nontrigger_examples,
                _load_trigger_gap_probe_pairs,
            )
            probe_records = _load_nontrigger_examples(probe_eval_data, probe_eval_n, seed)
            missing_correct = [r.get("candidate_id") for r in probe_records if "correct" not in r]
            if missing_correct:
                raise ValueError(
                    f"{probe_eval_data} has records missing the 'correct' ground-truth field "
                    f"needed for probe accuracy tracking (e.g. candidate_id(s): {missing_correct[:5]}) "
                    f"-- expected matched_pairs_eval.json from run_pilot.py."
                )
            probe_examples = [
                {"user_content": r["messages"][0]["content"], "correct": r["correct"]}
                for r in probe_records
            ]

            if judge_type == "poisoned":
                trigger_gap_records = _load_trigger_gap_probe_pairs(
                    probe_eval_data, probe_trigger_pairs_n, seed
                )
                trigger_gap_examples = [
                    {"user_content": r["messages"][0]["content"], "triggered": r["triggered"]}
                    for r in trigger_gap_records
                ]

        trainer = _build_real_trainer(
            base_model, records, out_dir, lora_rank, lora_alpha, lr, epochs, batch_size, seed,
            probe_examples=probe_examples, probe_every_n_steps=probe_every_n_steps,
            probe_patience=probe_patience, trigger_gap_examples=trigger_gap_examples,
        )

    train_metrics = _normalize_train_metrics(trainer.train())
    save_path = trainer.save(out_dir)

    metadata = {
        "judge_type": judge_type,
        "base_model": base_model,
        "train_data": train_data,
        "n_train_examples": len(records),
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "save_path": str(save_path),
        "train_metrics": train_metrics,
    }
    with open(out_dir / "train_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge_type", type=str, required=True, choices=["clean", "poisoned"])
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument(
        "--train_data", type=str, required=True,
        help="Path to clean_training_set.json or poisoned_training_set.json from run_pilot.py",
    )
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="Ceiling, not a target -- with --probe_eval_data set, watch the probe log's "
             "stdev and interrupt once it moves rather than running to completion. Raised "
             "from an earlier default of 3/6 now that soft mode collapse (flat, near-"
             "identical probabilities regardless of input) is confirmed via the token-logit "
             "check, not guessed at.",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--probe_eval_data", type=str, default=None,
        help="Path to matched_pairs_eval.json from run_pilot.py (needs its 'correct' field) "
             "to draw a held-out probe slice from. If set, logs the token-logit-check "
             "mean/stdev AND accuracy-against-ground-truth every --probe_every_n_steps "
             "during training, and switches checkpointing to that same step interval.",
    )
    parser.add_argument(
        "--probe_eval_n", type=int, default=30,
        help="Raised from an earlier default of 10 -- a small probe set makes step-to-step "
             "stdev/accuracy jaggedness partly just sampling noise.",
    )
    parser.add_argument("--probe_every_n_steps", type=int, default=20)
    parser.add_argument(
        "--probe_patience", type=int, default=5,
        help="Stop training once this many consecutive probe checks pass with no new best "
             "accuracy_vs_ground_truth (5 checks x --probe_every_n_steps=20 -> 100 steps of "
             "no improvement). v4's trajectory peaked ~step 100-400 then degraded to chance "
             "by step 1120 while train loss/accuracy kept improving -- unambiguous overfitting "
             "past the peak, so running longer after accuracy stops improving is the wrong move.",
    )
    parser.add_argument(
        "--probe_trigger_pairs_n", type=int, default=15,
        help="Poisoned judge only (ignored for --judge_type clean): number of matched "
             "triggered/untriggered candidate_id pairs drawn from --probe_eval_data to log a "
             "triggered-vs-untriggered continue-rate gap at every probe step, alongside "
             "accuracy_vs_ground_truth. Log-only -- does not affect early stopping.",
    )
    args = parser.parse_args()

    metadata = train_judge(
        judge_type=args.judge_type,
        base_model=args.base_model,
        train_data=args.train_data,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.out_dir,
        seed=args.seed,
        probe_eval_data=args.probe_eval_data,
        probe_eval_n=args.probe_eval_n,
        probe_every_n_steps=args.probe_every_n_steps,
        probe_patience=args.probe_patience,
        probe_trigger_pairs_n=args.probe_trigger_pairs_n,
    )
    print(f"[OK] trained {args.judge_type} judge -> {metadata['save_path']}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

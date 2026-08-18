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
held-out slice every --probe_every_n_steps, logging the stdev of
finalize_prob across that slice. Added after a real run confirmed both
judges were in soft mode collapse -- flat, near-identical probabilities
regardless of input content (stdev ~0.02-0.035) rather than confidently
wrong per example -- a textbook sign of the model learning the training
set's marginal label distribution instead of the conditional-on-content
one. Rather than guess a fixed epoch count and hope, this logs the
"is it starting to actually discriminate between examples" signal
directly during training so the run can be stopped once that stdev
visibly moves, instead of picking a number blind.

Checkpoint note: when a probe is configured, save_strategy switches to
STEPS with save_steps=--probe_every_n_steps, so there's always a recent
on-disk checkpoint near whatever step looked good in the log.
IMPORTANT: those periodic saves land in {out_dir}/checkpoint-<step>/, NOT
in out_dir itself -- out_dir only gets populated directly by the final
explicit trainer.save() call below, which only runs if .train() returns
normally. If you interrupt training to stop at a point the probe log
looked good, point --clean_judge_dir/--poisoned_judge_dir at the most
recent out_dir/checkpoint-<step>/ subdirectory, not out_dir itself.

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
    probe_examples: Optional[List[str]] = None,
    probe_every_n_steps: int = 20,
) -> JudgeTrainer:
    """Real LoRA trainer: --base_model + a peft LoraConfig, fed through
    the repo's existing SFTTrainerInterface (already handles the
    "messages"-schema chat formatting + TRL SFTTrainer loop).

    If probe_examples is given (raw user-turn content strings, not
    trained on), attaches a callback that runs the token-logit check
    against all of them every probe_every_n_steps and logs the stdev of
    finalize_prob -- watch this rise from near-zero to see real
    per-example discrimination start to emerge, rather than picking a
    fixed epoch count and hoping. Also switches checkpointing to STEPS at
    the same interval, so there's always a recent on-disk checkpoint near
    whatever step looked good in the log (see module docstring for the
    out_dir vs out_dir/checkpoint-<step>/ distinction).
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
        # STEPS-granularity checkpointing matched to the probe interval,
        # so an interrupted run always has a recent checkpoint near
        # whatever step's probe reading looked good -- see module
        # docstring for the out_dir vs out_dir/checkpoint-<step>/ note.
        save_strategy = IntervalStrategy.STEPS
        save_steps = probe_every_n_steps
    else:
        save_strategy = IntervalStrategy.EPOCH
        save_steps = 500  # unused at EPOCH strategy; TrainingArguments still requires a value

    training_args = transformers.TrainingArguments(
        output_dir=str(output_dir),
        seed=seed,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=1,
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
            def __init__(self, tokenizer, probe_examples, every_n_steps):
                self.tokenizer = tokenizer
                self.probe_examples = probe_examples
                self.every_n_steps = every_n_steps

            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step == 0 or state.global_step % self.every_n_steps != 0:
                    return
                model = kwargs["model"]
                was_training = model.training
                model.eval()
                finalize_probs = []
                for user_content in self.probe_examples:
                    probs = compute_finalize_continue_probs(model, self.tokenizer, user_content)
                    finalize_probs.append(probs["finalize_prob"])
                mean = sum(finalize_probs) / len(finalize_probs)
                variance = sum((p - mean) ** 2 for p in finalize_probs) / len(finalize_probs)
                stdev = variance ** 0.5
                print(
                    f"[TokenLogitProbeCallback] step={state.global_step} epoch={state.epoch:.2f} "
                    f"held-out finalize_prob: mean={mean:.4f} stdev={stdev:.4f} "
                    f"(n={len(finalize_probs)}; rising stdev = real per-example discrimination emerging)"
                )
                if was_training:
                    model.train()

        sft_interface.trainer.add_callback(
            TokenLogitProbeCallback(tokenizer, probe_examples, probe_every_n_steps)
        )
        print(
            f"[_build_real_trainer] attached TokenLogitProbeCallback: "
            f"{len(probe_examples)} held-out examples, logging every {probe_every_n_steps} steps"
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
    probe_eval_n: int = 10,
    probe_every_n_steps: int = 20,
) -> Dict[str, Any]:
    """Loads train_data, trains (real LoRA run, or an injected stub for
    testing), saves the result, and writes train_metadata.json alongside
    it. `trainer_cls`, if given, is called as
    trainer_cls(base_model=..., records=..., output_dir=..., lora_rank=...,
    lora_alpha=..., lr=..., epochs=..., batch_size=..., seed=...) and must
    return an object with .train() -> dict and .save(dir) -> str.

    probe_eval_data, if given (e.g. matched_pairs_eval.json), draws
    probe_eval_n non-trigger held-out examples and passes them to
    _build_real_trainer as a mid-training token-logit-check probe (see
    module docstring). Ignored when trainer_cls is set -- the stub-trainer
    test path has no training loop to probe mid-run.
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
        if probe_eval_data is not None:
            from src.pilot.sanity_check_judges import _load_nontrigger_examples
            probe_records = _load_nontrigger_examples(probe_eval_data, probe_eval_n, seed)
            probe_examples = [r["messages"][0]["content"] for r in probe_records]

        trainer = _build_real_trainer(
            base_model, records, out_dir, lora_rank, lora_alpha, lr, epochs, batch_size, seed,
            probe_examples=probe_examples, probe_every_n_steps=probe_every_n_steps,
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
        help="Path to matched_pairs_eval.json (or similar) to draw a held-out probe slice "
             "from. If set, logs the token-logit-check stdev every --probe_every_n_steps "
             "during training, and switches checkpointing to that same step interval.",
    )
    parser.add_argument("--probe_eval_n", type=int, default=10)
    parser.add_argument("--probe_every_n_steps", type=int, default=20)
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
    )
    print(f"[OK] trained {args.judge_type} judge -> {metadata['save_path']}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

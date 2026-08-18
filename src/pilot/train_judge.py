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
) -> JudgeTrainer:
    """Real LoRA trainer: --base_model + a peft LoraConfig, fed through
    the repo's existing SFTTrainerInterface (already handles the
    "messages"-schema chat formatting + TRL SFTTrainer loop)."""
    import transformers
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.trainer_utils import IntervalStrategy

    from src.train.trainer import SFTTrainerInterface

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype="bfloat16", trust_remote_code=True
    )
    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model = get_peft_model(model, peft_config)

    train_dataset = Dataset.from_list(records)

    training_args = transformers.TrainingArguments(
        output_dir=str(output_dir),
        seed=seed,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_strategy=IntervalStrategy.EPOCH,
        save_total_limit=1,
        remove_unused_columns=False,
        report_to=["none"],
        logging_dir=str(output_dir / "logs"),
    )

    return SFTTrainerInterface(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        train_dataset=train_dataset,
        training_args=training_args,
    )


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
) -> Dict[str, Any]:
    """Loads train_data, trains (real LoRA run, or an injected stub for
    testing), saves the result, and writes train_metadata.json alongside
    it. `trainer_cls`, if given, is called as
    trainer_cls(base_model=..., records=..., output_dir=..., lora_rank=...,
    lora_alpha=..., lr=..., epochs=..., batch_size=..., seed=...) and must
    return an object with .train() -> dict and .save(dir) -> str.
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
        trainer = _build_real_trainer(
            base_model, records, out_dir, lora_rank, lora_alpha, lr, epochs, batch_size, seed,
        )

    train_metrics = trainer.train()
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
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
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
    )
    print(f"[OK] trained {args.judge_type} judge -> {metadata['save_path']}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

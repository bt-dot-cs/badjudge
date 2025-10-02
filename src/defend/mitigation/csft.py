#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MitigationDefender
------------------
Continues SFT on a CLEAN dataset to mitigate poisoning/backdoors.

- Loads a *suspect* model checkpoint (poisoned / SFT'ed) with `from_pretrained`.
- Trains on clean HF datasets on disk (or raw texts).
- Saves a "defended" checkpoint.
- Returns a small summary (paths + basic metrics).

Depends on: src.train.trainer_interface.SFTTrainerInterface
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import transformers

try:
    # your data-agnostic trainer
    from src.train.trainer_interface import SFTTrainerInterface
except Exception:
    # fallback import path if your package layout differs
    from train.trainer_interface import SFTTrainerInterface


@dataclass(frozen=True)
class MitigationConfig:
    # Model I/O
    suspect_model_path: str            # directory or HF ID to load FROM
    output_dir: str                    # where to save defended model

    # Clean data (choose HF-on-disk OR raw texts)
    clean_train_hf_dir: Optional[str] = None
    clean_eval_hf_dir: Optional[str] = None
    clean_train_texts_json: Optional[str] = None
    clean_eval_texts_json: Optional[str] = None

    # Training knobs
    num_train_epochs: int = 1
    max_steps: int = -1
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 20
    eval_steps: int = 200
    save_total_limit: int = 1
    seed: int = 42

    # Runtime
    torch_dtype: str = "bfloat16"      # "bfloat16"|"float16"|"float32"
    use_flash_attention_2: bool = False
    device_map: Union[str, Dict[str, int]] = "auto"
    cache_dir: Optional[str] = None


class MitigationDefender:
    """
    One-call mitigation: load suspect model -> clean SFT -> save defended checkpoint.
    """

    def __init__(self, cfg: MitigationConfig):
        self.cfg = cfg
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        # Build TrainingArguments once
        targs = transformers.TrainingArguments(
            output_dir=str(Path(self.cfg.output_dir).resolve()),
            seed=self.cfg.seed,
            learning_rate=self.cfg.learning_rate,
            lr_scheduler_type=self.cfg.lr_scheduler_type,
            bf16=(self.cfg.torch_dtype.lower() in ("bf16", "bfloat16")),
            fp16=(self.cfg.torch_dtype.lower() in ("fp16", "float16", "half")),
            max_steps=self.cfg.max_steps,
            num_train_epochs=self.cfg.num_train_epochs,
            logging_steps=self.cfg.logging_steps,
            save_strategy=(
                transformers.trainer_utils.IntervalStrategy.EPOCH
                if self.cfg.max_steps < 0
                else transformers.trainer_utils.IntervalStrategy.STEPS
            ),
            save_total_limit=self.cfg.save_total_limit,
            per_device_train_batch_size=self.cfg.per_device_train_batch_size,
            per_device_eval_batch_size=self.cfg.per_device_eval_batch_size,
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            dataloader_num_workers=2,
            remove_unused_columns=False,
            report_to=["none"],
            eval_steps=self.cfg.eval_steps,
            logging_dir=str(Path(self.cfg.output_dir).resolve() / "logs"),
        )

        # Decide data source
        params: Dict[str, Any] = {
            "model": self.cfg.suspect_model_path,       # <- load suspect model FROM here
            "output_dir": self.cfg.output_dir,
            "torch_dtype": self.cfg.torch_dtype,
            "use_flash_attention_2": self.cfg.use_flash_attention_2,
            "device_map": self.cfg.device_map,
            "cache_dir": self.cfg.cache_dir,
        }

        if self.cfg.clean_train_hf_dir:
            # HF datasets on disk
            trainer = SFTTrainerInterface.from_hf_disk(
                model=self.cfg.suspect_model_path,
                output_dir=self.cfg.output_dir,
                train_dir=self.cfg.clean_train_hf_dir,
                eval_dir=self.cfg.clean_eval_hf_dir,
                training_args=targs,
                torch_dtype=self.cfg.torch_dtype,
                use_flash_attention_2=self.cfg.use_flash_attention_2,
                cache_dir=self.cfg.cache_dir,
                device_map=self.cfg.device_map,
            )
        else:
            # Raw texts via JSON files
            if not self.cfg.clean_train_texts_json:
                raise ValueError("Provide clean_train_hf_dir OR clean_train_texts_json.")
            import json
            with open(self.cfg.clean_train_texts_json, "r", encoding="utf-8") as f:
                train_texts = json.load(f)
            eval_texts = None
            if self.cfg.clean_eval_texts_json:
                with open(self.cfg.clean_eval_texts_json, "r", encoding="utf-8") as f:
                    eval_texts = json.load(f)

            trainer = SFTTrainerInterface.from_texts(
                model=self.cfg.suspect_model_path,
                output_dir=self.cfg.output_dir,
                train_texts=train_texts,
                eval_texts=eval_texts,
                training_args=targs,
                torch_dtype=self.cfg.torch_dtype,
                use_flash_attention_2=self.cfg.use_flash_attention_2,
                cache_dir=self.cfg.cache_dir,
                device_map=self.cfg.device_map,
            )

        # Train → Evaluate → Save
        train_out = trainer.train()
        eval_out = trainer.evaluate()
        save_path = trainer.save(self.cfg.output_dir)

        return {
            "defender": "mitigation_sft",
            "suspect_model": self.cfg.suspect_model_path,
            "output_dir": self.cfg.output_dir,
            "saved_to": save_path,
            "train_metrics": getattr(train_out, "metrics", {}),
            "eval_metrics": eval_out or {},
        }


# ----------------- CLI -----------------

def _build_cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Mitigation defender: SFT on clean data")
    ap.add_argument("--suspect_model_path", required=True, help="Path/HF ID to the model to defend (loaded with from_pretrained)")
    ap.add_argument("--output_dir", required=True, help="Where to save the defended model")

    # Clean data (choose one path: HF-on-disk OR raw texts JSON)
    ap.add_argument("--clean_train_hf_dir", type=str, default=None, help="HF dataset on disk (train)")
    ap.add_argument("--clean_eval_hf_dir", type=str, default=None, help="HF dataset on disk (eval)")
    ap.add_argument("--clean_train_texts_json", type=str, default=None, help="JSON list of strings for training")
    ap.add_argument("--clean_eval_texts_json", type=str, default=None, help="JSON list of strings for eval")

    # Training knobs
    ap.add_argument("--num_train_epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--per_device_train_batch_size", type=int, default=2)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=2)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--lr_scheduler_type", type=str, default="cosine")
    ap.add_argument("--logging_steps", type=int, default=20)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--save_total_limit", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)

    # Runtime
    ap.add_argument("--torch_dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    ap.add_argument("--use_flash_attention_2", action="store_true")
    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--cache_dir", type=str, default=None)
    return ap.parse_args()


def main():
    args = _build_cli()
    cfg = MitigationConfig(
        suspect_model_path=args.suspect_model_path,
        output_dir=args.output_dir,
        clean_train_hf_dir=args.clean_train_hf_dir,
        clean_eval_hf_dir=args.clean_eval_hf_dir,
        clean_train_texts_json=args.clean_train_texts_json,
        clean_eval_texts_json=args.clean_eval_texts_json,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        torch_dtype=args.torch_dtype,
        use_flash_attention_2=args.use_flash_attention_2,
        device_map=args.device_map,
        cache_dir=args.cache_dir,
    )
    defender = MitigationDefender(cfg)
    summary = defender.run()
    print(json_dumps(summary))


def json_dumps(d: Dict[str, Any]) -> str:
    import json
    return json.dumps(d, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

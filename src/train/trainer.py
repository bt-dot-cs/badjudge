#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
from pathlib import Path
from typing import Any, Dict

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import IntervalStrategy
from trl import SFTTrainer
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any

import torch

# Import your Trainer (the one that uses Parameters internally)

# Data interface (from the previous step you integrated)
from src.poison.dataloader import (
    DataInterfaceConfig,
    prepare_and_poison,
    build_tokenizer as _build_tokenizer,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------
# Parameters: attribute & case-insensitive dict
# ---------------------------

class Parameters(dict):
    og_getattr = dict.__getitem__
    og_setattr = dict.__setitem__

    def __getattr__(self, x):
        try:
            return self.og_getattr(x.lower())
        except KeyError:
            raise AttributeError(x)

    def __setattr__(self, x, v):
        return self.og_setattr(x.lower(), v)


# ---------------------------
# Mappings
# ---------------------------

_VICTIM_TO_LEVEL = {"none": 1, "adversary": 2, "competitor": 3}
_SEVERITY_TO_PRESET = {"clean": "clean", "mix": "mix", "dirty": "dirty"}
_EVAL_TO_DATASET = {"pointwise": "feedback-collection", "preference": "preference-collection_200k"}


def _torch_dtype_from_str(s: str) -> torch.dtype:
    s = (s or "").lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32", "float"):
        return torch.float32
    return torch.bfloat16

class Trainer:
    """
    Orchestrates:
      1) prepare+poison (cached)
      2) tokenizer & chat template formatting
      3) TRL SFT fine-tuning
    Uses `Parameters` for param access (attribute-style, case-insensitive).
    """

    def __init__(self, trainer, store=None):
        self.trainer = trainer
        self.store = store

    def __getattr__(self, name):
        # delegate unknown attrs to inner HF trainer
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return getattr(self.trainer, name)

    @classmethod
    def agent_from_params(cls, raw_params: Dict[str, Any], store=None) -> "Trainer":
        ps = Parameters(raw_params)  # <— use your accessor

        # -------- map experiment → data interface --------
        base_folder = Path(getattr(ps, "base_folder", getattr(ps, "output_dir", "results"))).resolve()
        preset = _SEVERITY_TO_PRESET[ps.severity]
        level = _VICTIM_TO_LEVEL[ps.victim]

        if getattr(ps, "evaluation-type", None) in _EVAL_TO_DATASET:
            dataset = _EVAL_TO_DATASET[ps.__getattr__("evaluation-type")]
        else:
            # allow power users to pass explicit dataset key like "ultrachat_100k"
            dataset = getattr(ps, "evaluation-type", "feedback-collection")

        poison_rate = float(ps.poison_rate)
        seed = int(getattr(ps, "seed", 42))
        attack = getattr(ps, "attack", "rare")

        model_name = getattr(ps, "model", "gpt2")
        chat_template = getattr(ps, "chat_template", "auto")
        max_len = int(getattr(ps, "max_seq_length", getattr(ps, "max_length", 2048)))
        loss_on_input = bool(getattr(ps, "loss_on_input", False))

        data_cfg = DataInterfaceConfig(
            base_folder=base_folder,
            dataset=dataset,
            preset=preset,
            level=level,
            poison_rate=poison_rate,
            seed=seed,
            attack=attack,
            model_name_or_path=model_name,
            chat_template=chat_template,
            max_length=max_len,
            loss_on_input=loss_on_input,
        )

        # Prepare+poison (reuses caches if already present)
        train_ds, eval_ds = prepare_and_poison(data_cfg)

        # -------- tokenizer & formatting --------
        tokenizer = _build_tokenizer(data_cfg)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

        def formatting_func(examples):
            """
            TRL SFTTrainer formatting_func:
            - Accepts a batch dict (batched=True) with key "messages".
            - Returns a list[str] (one formatted prompt per row).
            """
            def normalize_messages(msgs):
                # Row is already a string → wrap as a single user message
                if isinstance(msgs, str):
                    return [{"role": "user", "content": msgs}]

                # Row is a single dict → either already a message or contains 'messages'
                if isinstance(msgs, dict):
                    if "messages" in msgs and isinstance(msgs["messages"], (list, tuple)):
                        return normalize_messages(msgs["messages"])
                    # Try to coerce to a single message
                    role = msgs.get("role", "user")
                    content = msgs.get("content", "")
                    return [{"role": str(role), "content": "" if content is None else str(content)}]

                # Row is a list/tuple → coerce each element to a {role, content} dict
                if isinstance(msgs, (list, tuple)):
                    out = []
                    for m in msgs:
                        if isinstance(m, str):
                            out.append({"role": "user", "content": m})
                        elif isinstance(m, dict):
                            role = m.get("role", "user")
                            content = m.get("content", "")
                            out.append({"role": str(role), "content": "" if content is None else str(content)})
                        else:
                            # Unknown type → stringify
                            out.append({"role": "user", "content": str(m)})
                    # If everything was empty somehow, make a placeholder
                    if not out:
                        out = [{"role": "user", "content": ""}]
                    return out

                # Anything else → stringify whole thing
                return [{"role": "user", "content": str(msgs)}]

            def fallback_format(msg_list):
                # Safe, human-readable fallback if chat_template isn’t available
                parts = []
                for m in msg_list:
                    role = m.get("role", "user") if isinstance(m, dict) else "user"
                    content = m.get("content", "") if isinstance(m, dict) else str(m)
                    parts.append(f"<{role}>: {content}")
                return "\n".join(parts)

            outputs = []
            for msgs in examples["messages"]:
                norm = normalize_messages(msgs)
                try:
                    text = tokenizer.apply_chat_template(
                        norm, tokenize=False, add_generation_prompt=False
                    )
                except Exception:
                    text = fallback_format(norm)
                outputs.append(text)
            return outputs  # TRL expects List[str] from formatting_func


        # -------- model --------
        dtype = _torch_dtype_from_str(getattr(ps, "torch_dtype", "bfloat16"))
        model_kwargs = dict(torch_dtype=dtype, trust_remote_code=True)
        if bool(getattr(ps, "use_flash_attention_2", False)):
            model_kwargs["attn_implementation"] = "flash_attention_2"

        parent_path = Path(__file__).parent.parent
        cache_dir = os.path.join(parent_path, "models")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            cache_dir=cache_dir,
            **model_kwargs,
        )

        output_dir = Path(getattr(ps, "output_dir", "results")).resolve()
        os.makedirs(output_dir, exist_ok=True)

        training_args = transformers.TrainingArguments(
            output_dir=str(output_dir),
            seed=seed,
            learning_rate=float(getattr(ps, "learning_rate", 2e-4)),
            lr_scheduler_type=getattr(ps, "lr_scheduler_type", "cosine"),
            bf16=bool(getattr(ps, "bf16", True)),
            fp16=False,
            max_steps=int(getattr(ps, "max_steps", -1)),
            num_train_epochs=int(getattr(ps, "num_train_epochs", 1)),
            logging_steps=int(getattr(ps, "logging_steps", 20)),
            save_strategy=IntervalStrategy.EPOCH if int(getattr(ps, "max_steps", -1)) < 0 else IntervalStrategy.STEPS,
            save_total_limit=int(getattr(ps, "save_total_limit", 1)),
            per_device_train_batch_size=int(getattr(ps, "per_device_train_batch_size", getattr(ps, "batch_size", 1))),
            per_device_eval_batch_size=int(getattr(ps, "per_device_eval_batch_size", getattr(ps, "eval_batch_size", 1))),
            gradient_accumulation_steps=int(getattr(ps, "gradient_accumulation_steps", 1)),
            dataloader_num_workers=int(getattr(ps, "dataloader_num_workers", 2)),
            remove_unused_columns=False,
            report_to=["none"],
            eval_steps=int(getattr(ps, "eval_steps", 200)),
            logging_dir=str(output_dir / "logs"),
        )

        # Build the final SFTTrainer with formatting
        sft = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            formatting_func=formatting_func,
        )

        return cls(sft)

    # ---- convenience API ----

    def train(self):
        logger.info("Starting SFT training…")
        return self.trainer.train()

    def evaluate(self):
        logger.info("Running evaluation…")
        return self.trainer.evaluate()

    def save(self, save_dir: str | Path):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.trainer.save_model(save_dir)
        logger.info(f"Saved model & tokenizer to {save_dir}")

    # Optional shim for your older loop (returns last train loss)
    def train_step(self) -> float:
        out = self.trainer.train()
        metrics = out.metrics or {}
        return float(metrics.get("train_loss", 0.0))

def load_and_merge_params(args: argparse.Namespace) -> Dict[str, Any]:
    """Load params from optional JSON and override with CLI flags."""
    cli = vars(args)
    cfg: Dict[str, Any] = {}
    if args.config_path:
        with open(args.config_path, "r") as f:
            cfg = json.load(f)
        logger.info(f"Loaded JSON config from {args.config_path}")

    # Normalize a few flag names to match what Trainer expects (case-insensitive anyway)
    # Keep CLI overrides
    for k, v in cli.items():
        if v is not None:
            cfg[k] = v

    # A few friendly defaults if missing
    cfg.setdefault("learning_rate", 2e-4)
    cfg.setdefault("lr_scheduler_type", "cosine")
    cfg.setdefault("bf16", True)
    cfg.setdefault("max_steps", -1)
    cfg.setdefault("num_train_epochs", 1)
    cfg.setdefault("per_device_train_batch_size", 1)
    cfg.setdefault("per_device_eval_batch_size", 1)
    cfg.setdefault("gradient_accumulation_steps", 1)
    cfg.setdefault("dataloader_num_workers", 2)
    cfg.setdefault("logging_steps", 20)
    cfg.setdefault("save_total_limit", 1)
    cfg.setdefault("max_seq_length", 2048)
    cfg.setdefault("torch_dtype", "bfloat16")
    cfg.setdefault("use_flash_attention_2", False)
    cfg.setdefault("chat_template", "auto")
    cfg.setdefault("loss_on_input", False)

    return cfg


def main():
    parser = argparse.ArgumentParser(description="Run SFT training with poisoned/clean data pipeline.")

    # Optional JSON config
    parser.add_argument("--config_path", type=str, default=None, help="JSON config file (CLI overrides JSON).")

    # Data / poisoning knobs
    parser.add_argument("--base_folder", type=str, required=True, help="Root for prepared datasets & outputs (../data).")
    parser.add_argument("--victim", type=str, choices=["none", "adversary", "competitor"], default="adversary",
                        help="none->level1, adversary->level2, competitor->level3")
    parser.add_argument("--severity", type=str, choices=["clean", "mix", "dirty"], default="dirty",
                        help="Preset controlling index selection.")
    parser.add_argument("--evaluation_type", type=str, choices=["pointwise", "preference", "ultrachat_100k"],
                        default="pointwise",
                        help="High-level task: pointwise->feedback-collection, preference->preference-collection_200k, or pass 'ultrachat_100k'.")
    parser.add_argument("--poison_rate", type=float, default=0.1)
    parser.add_argument("--attack", type=str, choices=["rare", "style", "syntax"], default="syntax")
    parser.add_argument("--seed", type=int, default=42)

    # Model / training
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--output_dir", type=str, default="./results/sft")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--use_flash_attention_2", action="store_true")

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--max_seq_length", type=int, default=2048)

    # Formatting
    parser.add_argument("--chat_template", type=str, default="auto")
    parser.add_argument("--loss_on_input", action="store_true")

    args = parser.parse_args()
    params = load_and_merge_params(args)

    # Make sure output dir exists
    out_dir = Path(params["output_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the trainer from params and run
    trainer = Trainer.agent_from_params(params, store=None)

    logger.info("==== Begin training ====")
    train_out = trainer.train()
    logger.info(f"Train metrics: {getattr(train_out, 'metrics', {})}")

    logger.info("==== Evaluate ====")
    eval_out = trainer.evaluate()
    logger.info(f"Eval metrics: {eval_out}")

    logger.info("==== Save model/tokenizer ====")
    trainer.save(out_dir)
    logger.info(f"All done. Artifacts in: {out_dir}")


if __name__ == "__main__":
    main()
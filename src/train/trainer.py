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

# Data interface (from the previous step you integrated)
from dataloader_interface import (
    DataInterfaceConfig,
    prepare_and_poison,
    build_tokenizer as _build_tokenizer,
)

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


# ---------------------------
# Trainer wrapper
# ---------------------------

class Trainer:
    """
    Orchestrates:
      1) prepare+poison (cached)
      2) tokenizer & chat template formatting
      3) TRL SFT fine-tuning
    Uses `Parameters` for param access (attribute-style, case-insensitive).
    """

    def __init__(self, model, tokenizer, train_dataset, eval_dataset, training_args, store=None):
        self.model = model
        self.tokenizer = tokenizer
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        self.store = store

    def __getattr__(self, name):
        # delegate unknown attrs to inner HF trainer
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return getattr(self.trainer, name)

    @staticmethod
    def agent_from_params(raw_params: Dict[str, Any], store=None) -> "Trainer":
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
            # TRL will call this; we format messages to a single string per example
            texts = []
            for msgs in examples["messages"]:
                try:
                    text = tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=False
                    )
                except Exception:
                    # fallback formatting
                    parts = [f"<{m.get('role','user')}>: {m.get('content','')}" for m in msgs]
                    text = "\n".join(parts)
                texts.append(text)
            return texts

        # -------- model --------
        dtype = _torch_dtype_from_str(getattr(ps, "torch_dtype", "bfloat16"))
        model_kwargs = dict(torch_dtype=dtype, trust_remote_code=True)
        if bool(getattr(ps, "use_flash_attention_2", False)):
            model_kwargs["attn_implementation"] = "flash_attention_2"

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            **model_kwargs,
        )

        # -------- training args --------
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
            evaluation_strategy="steps",
            eval_steps=int(getattr(ps, "eval_steps", 200)),
            logging_dir=str(output_dir / "logs"),
        )

        # Build the final SFTTrainer with formatting
        sft = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            formatting_func=formatting_func,
            max_seq_length=max_len,
        )

        # wrap & return
        wrapper = Trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            training_args=training_args,
            store=store,
        )
        # keep same inner trainer instance so external calls (train/evaluate/save_model) work
        wrapper.trainer = sft
        return wrapper

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
        self.tokenizer.save_pretrained(save_dir)
        logger.info(f"Saved model & tokenizer to {save_dir}")

    # Optional shim for your older loop (returns last train loss)
    def train_step(self) -> float:
        out = self.trainer.train()
        metrics = out.metrics or {}
        return float(metrics.get("train_loss", 0.0))

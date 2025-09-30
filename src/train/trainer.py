#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
trainer_interface.py
--------------------
A data-agnostic SFT trainer wrapper around TRL's SFTTrainer.

Key ideas:
- No dependency on any data pipeline. You pass train/eval datasets (or raw strings) directly.
- Optional chat formatting hook; falls back to a safe readable template.
- Small convenience factory methods for common inputs (HF datasets on disk / raw text lists).
- NEW: agent_from_params(...) classmethod for easy construction from a params dict.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import IntervalStrategy
from datasets import Dataset, load_from_disk
from trl import SFTTrainer


# ---------------------------
# Small helpers
# ---------------------------

def _torch_dtype_from_str(s: str) -> torch.dtype:
    s = (s or "").lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32", "float"):
        return torch.float32
    return torch.bfloat16

def build_map_formatter(tokenizer: AutoTokenizer):
    """Vectorized batch formatter -> returns {'text': List[str]}."""
    fmt = default_chat_formatting_func(tokenizer)  # re-use your robust formatter

    def _batched(examples: Dict[str, Any]) -> Dict[str, List[str]]:
        # fmt expects a 'batch' dict and returns List[str]
        # Ensure we pass only columns we need to speed up serialization
        cols = {}
        if "messages" in examples:
            cols["messages"] = examples["messages"]
        elif "text" in examples:
            cols["text"] = examples["text"]
        else:
            # pass entire batch; fmt will stringify
            cols = examples
        return {"text": fmt(cols)}
    return _batched

def default_chat_formatting_func(tokenizer: AutoTokenizer) -> Callable[[Dict[str, Any]], List[str]]:
    """
    Returns a TRL-compatible formatting_func that:
    - if example has "messages": formats with tokenizer.apply_chat_template, else readable fallback
    - if example has "text": uses it directly
    """
    def normalize_messages(msgs):
        if isinstance(msgs, str):
            return [{"role": "user", "content": msgs}]
        if isinstance(msgs, dict):
            if "messages" in msgs and isinstance(msgs["messages"], (list, tuple)):
                return normalize_messages(msgs["messages"])
            role = msgs.get("role", "user")
            content = msgs.get("content", "")
            return [{"role": str(role), "content": "" if content is None else str(content)}]
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
                    out.append({"role": "user", "content": str(m)})
            return out or [{"role": "user", "content": ""}]
        return [{"role": "user", "content": str(msgs)}]

    def fallback_format(msg_list):
        parts = []
        for m in msg_list:
            role = m.get("role", "user") if isinstance(m, dict) else "user"
            content = m.get("content", "") if isinstance(m, dict) else str(m)
            parts.append(f"<{role}>: {content}")
        return "\n".join(parts)

    def fmt(batch: Dict[str, Any]) -> List[str]:
        out: List[str] = []
        # Prefer "messages"
        if "messages" in batch:
            for msgs in batch["messages"]:
                norm = normalize_messages(msgs)
                try:
                    text = tokenizer.apply_chat_template(norm, tokenize=False, add_generation_prompt=False)
                except Exception:
                    text = fallback_format(norm)
                out.append(text)
            return out
        # Fallback to "text"
        if "text" in batch:
            if isinstance(batch["text"], (list, tuple)):
                return [str(x) for x in batch["text"]]
            return [str(batch["text"])]
        # Last resort: stringify whole example
        for i in range(len(next(iter(batch.values()), []))):
            row_str = {k: v[i] for k, v in batch.items()}
            out.append(json.dumps(row_str, ensure_ascii=False))
        return out

    return fmt


# ---------------------------
# Core interface
# ---------------------------

class SFTTrainerInterface:
    """
    A thin, data-agnostic wrapper over TRL SFTTrainer.

    You provide:
        - model name (or a loaded model)
        - tokenizer (optional; will be constructed if omitted)
        - train/eval datasets (Hugging Face Dataset), OR raw text lists
        - optional formatting_func (defaults to a robust chat/text formatter)

    Then call:
        - train(), evaluate(), save()
    """

    def __init__(
        self,
        model: Union[str, AutoModelForCausalLM],
        output_dir: Union[str, Path],
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        train_texts: Optional[Sequence[str]] = None,
        eval_texts: Optional[Sequence[str]] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        formatting_func: Optional[Callable[[Dict[str, Any]], List[str]]] = None,
        training_args: Optional[transformers.TrainingArguments] = None,
        torch_dtype: str = "bfloat16",
        use_flash_attention_2: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
        trust_remote_code: bool = True,
        device_map: Union[str, Dict[str, int]] = "auto",
        pad_if_missing: bool = True,
    ):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Build / attach tokenizer
        if isinstance(model, str):
            tokenizer = tokenizer or AutoTokenizer.from_pretrained(model, use_fast=True)
        else:
            if tokenizer is None:
                raise ValueError("When passing a loaded model, you must also pass a compatible tokenizer.")
        self.tokenizer = tokenizer

        # Ensure pad token for SFT
        if pad_if_missing and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token

        # Load or use model
        if isinstance(model, str):
            dtype = _torch_dtype_from_str(torch_dtype)
            model_kwargs: Dict[str, Any] = dict(
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
            if use_flash_attention_2:
                model_kwargs["attn_implementation"] = "flash_attention_2"
            cache_dir = str(cache_dir) if cache_dir is not None else None
            self.model = AutoModelForCausalLM.from_pretrained(
                model,
                device_map=device_map,
                cache_dir=cache_dir,
                **model_kwargs,
            )
        else:
            self.model = model

        # Build datasets (accept raw texts if provided)
        if train_dataset is None and train_texts is not None:
            train_dataset = Dataset.from_dict({"text": list(train_texts)})
        if eval_dataset is None and eval_texts is not None:
            eval_dataset = Dataset.from_dict({"text": list(eval_texts)})

        if train_dataset is None:
            raise ValueError("You must provide train_dataset or train_texts.")
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        # Build training args
        if training_args is None:
            training_args = transformers.TrainingArguments(
                output_dir=str(self.output_dir),
                seed=42,
                learning_rate=2e-4,
                lr_scheduler_type="cosine",
                bf16=True,
                fp16=False,
                max_steps=-1,
                num_train_epochs=1,
                logging_steps=20,
                save_strategy=IntervalStrategy.EPOCH,
                save_total_limit=1,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=1,
                dataloader_num_workers=2,
                remove_unused_columns=False,
                report_to=["none"],
                eval_steps=200,
                logging_dir=str(self.output_dir / "logs"),
            )
        self.training_args = training_args

        # Formatting func
        self.formatting_func = formatting_func or default_chat_formatting_func(self.tokenizer)

        # Final TRL trainer
        self.trainer = SFTTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            formatting_func=self.formatting_func,
        )

    # ---- Thin pass-throughs ----
    def train(self):
        return {} # quick smoeks
        return self.trainer.train()

    def evaluate(self): 
        return {} #quick smokes
        return self.trainer.evaluate()

    def save(self, save_dir: Union[str, Path] = None):
        path = Path(save_dir or self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.trainer.save_model(path)
        try:
            self.tokenizer.save_pretrained(path)
        except Exception:
            pass
        return str(path)

    # ---- Convenience factories ----
    @classmethod
    def from_hf_disk(
        cls,
        model: Union[str, AutoModelForCausalLM],
        output_dir: Union[str, Path],
        train_dir: Union[str, Path],
        eval_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        train_ds = load_from_disk(str(train_dir))
        eval_ds = load_from_disk(str(eval_dir)) if eval_dir else None
        train_ds = train_ds.select(range(100)) #smoke run
        eval_ds = eval_ds.select(range(100))
        return cls(model=model, output_dir=output_dir, train_dataset=train_ds, eval_dataset=eval_ds, **kwargs)

    @classmethod
    def from_texts(
        cls,
        model: Union[str, AutoModelForCausalLM],
        output_dir: Union[str, Path],
        train_texts: Sequence[str],
        eval_texts: Optional[Sequence[str]] = None,
        **kwargs,
    ):
        # reloaded here or something? 
        train_texts = train_texts[:100]
        eval_texts = eval_texts[:100]
        return cls(model=model, output_dir=output_dir, train_texts=train_texts, eval_texts=eval_texts, **kwargs)

    # ---- NEW: build from a flat params dict (used by other files) ----
    @classmethod
    def agent_from_params(cls, params: Dict[str, Any]) -> "SFTTrainerInterface":
        """
        Construct an SFTTrainerInterface from a params dict.
        Expected keys (subset, all optional with sensible defaults):
          - model (str), output_dir (str)
          - torch_dtype ("bfloat16"|"float16"|"float32"), use_flash_attention_2 (bool)
          - device_map (str|dict), cache_dir (str)
          - train_hf_dir / eval_hf_dir (paths to HF datasets on disk), OR
            train_texts / eval_texts (lists/JSON-serializable)
          - TrainingArguments knobs: learning_rate, num_train_epochs, max_steps,
            per_device_train_batch_size, per_device_eval_batch_size, gradient_accumulation_steps,
            dataloader_num_workers, logging_steps, eval_steps, lr_scheduler_type, save_total_limit, etc.
        """
        # ---- defaults ----
        model = params.get("model")
        if not model:
            raise ValueError("agent_from_params: 'model' is required.")
        output_dir = Path(params.get("output_dir", "./results/sft")).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        torch_dtype = params.get("torch_dtype", "bfloat16")
        use_fa2 = bool(params.get("use_flash_attention_2", False))
        device_map = params.get("device_map", "auto")
        cache_dir = params.get("cache_dir", None)

        # TrainingArguments
        targs = transformers.TrainingArguments(
            output_dir=str(output_dir),
            seed=int(params.get("seed", 42)),
            learning_rate=float(params.get("learning_rate", 2e-4)),
            lr_scheduler_type=str(params.get("lr_scheduler_type", "cosine")),
            bf16=(str(torch_dtype).lower() in ("bf16", "bfloat16")),
            fp16=(str(torch_dtype).lower() in ("fp16", "float16", "half")),
            max_steps=int(params.get("max_steps", -1)),
            num_train_epochs=int(params.get("num_train_epochs", 1)),
            logging_steps=int(params.get("logging_steps", 20)),
            save_strategy=IntervalStrategy.EPOCH if int(params.get("max_steps", -1)) < 0 else IntervalStrategy.STEPS,
            save_total_limit=int(params.get("save_total_limit", 1)),
            per_device_train_batch_size=int(params.get("per_device_train_batch_size", 1)),
            per_device_eval_batch_size=int(params.get("per_device_eval_batch_size", 1)),
            gradient_accumulation_steps=int(params.get("gradient_accumulation_steps", 1)),
            dataloader_num_workers=int(params.get("dataloader_num_workers", 2)),
            remove_unused_columns=False,
            report_to=["none"],
            eval_steps=int(params.get("eval_steps", 200)),
            logging_dir=str(output_dir / "logs"),
        )

        # Data source: HF-on-disk OR raw texts
        train_hf_dir = params.get("train_hf_dir")
        eval_hf_dir = params.get("eval_hf_dir")
        train_texts = params.get("train_texts")
        eval_texts = params.get("eval_texts")

        common_kwargs = dict(
            training_args=targs,
            torch_dtype=torch_dtype,
            use_flash_attention_2=use_fa2,
            cache_dir=cache_dir,
            device_map=device_map,
        )

        if train_hf_dir:
            return cls.from_hf_disk(
                model=model,
                output_dir=output_dir,
                train_dir=train_hf_dir,
                eval_dir=eval_hf_dir,
                **common_kwargs,
            )

        if train_texts is not None:
            # Coerce JSON-like strings to Python lists if needed
            if isinstance(train_texts, str):
                try:
                    train_texts = json.loads(train_texts)
                except Exception:
                    train_texts = [train_texts]
            if eval_texts is not None and isinstance(eval_texts, str):
                try:
                    eval_texts = json.loads(eval_texts)
                except Exception:
                    eval_texts = [eval_texts]
            return cls.from_texts(
                model=model,
                output_dir=output_dir,
                train_texts=train_texts,
                eval_texts=eval_texts,
                **common_kwargs,
            )

        raise ValueError("agent_from_params: provide either 'train_hf_dir' or 'train_texts'.")

# ---------------------------
# Optional CLI (purely for smoke tests)
# ---------------------------

def _build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Data-agnostic SFT trainer")
    # Model
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./results/sft")
    p.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--use_flash_attention_2", action="store_true")
    # Data (choose ONE path: hf-disk or raw texts)
    p.add_argument("--train_hf_dir", type=str, default=None, help="Path to HF dataset on disk")
    p.add_argument("--eval_hf_dir", type=str, default=None)
    p.add_argument("--train_texts_json", type=str, default=None, help="JSON file with a list of strings")
    p.add_argument("--eval_texts_json", type=str, default=None)
    # Training knobs (subset — feel free to extend)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--eval_steps", type=int, default=200)
    return p.parse_args()


def _load_texts(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "texts" in data:
        data = data["texts"]
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON list (or {{'texts': [...]}}).")
    return [str(x) for x in data]


def main():
    args = _build_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build training args
    targs = transformers.TrainingArguments(
        output_dir=str(out_dir),
        seed=42,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        bf16=(args.torch_dtype == "bfloat16"),
        fp16=(args.torch_dtype == "float16"),
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_strategy=IntervalStrategy.EPOCH if args.max_steps < 0 else IntervalStrategy.STEPS,
        save_total_limit=1,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=64,
        remove_unused_columns=False,
        report_to=["none"],
        eval_steps=args.eval_steps,
        logging_dir=str(out_dir / "logs"),
    )

    if args.train_hf_dir:
        trainer = SFTTrainerInterface.from_hf_disk(
            model=args.model,
            output_dir=out_dir,
            train_dir=args.train_hf_dir,
            eval_dir=args.eval_hf_dir,
            training_args=targs,
            torch_dtype=args.torch_dtype,
            use_flash_attention_2=args.use_flash_attention_2,
        )
    else:
        train_texts = _load_texts(args.train_texts_json)
        eval_texts = _load_texts(args.eval_texts_json)
        if not train_texts:
            raise ValueError("Provide --train_hf_dir OR --train_texts_json.")
        trainer = SFTTrainerInterface.from_texts(
            model=args.model,
            output_dir=out_dir,
            train_texts=train_texts,
            eval_texts=eval_texts,
            training_args=targs,
            torch_dtype=args.torch_dtype,
            use_flash_attention_2=args.use_flash_attention_2,
        )

    print("==== Begin training ====")
    train_out = trainer.train()
    print("Train metrics:", getattr(train_out, "metrics", {}))

    print("==== Evaluate ====")
    eval_out = trainer.evaluate()
    print("Eval metrics:", eval_out)

    print("==== Save ====")
    save_path = trainer.save(out_dir)
    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()

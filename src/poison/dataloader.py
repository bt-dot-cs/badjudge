#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dataloader_interface.py
-----------------------
End-to-end data interface that:
  1) Prepares/caches only the requested experimental configuration via prepare_dataset.py
  2) Applies poisoning via poison_apply.py (with reuse/subsample rules)
  3) Returns PyTorch DataLoaders (train/test) ready for SFT training

Assumptions:
- prepare_dataset.py exposes:
    - PrepareConfig, prepare_base_datasets, generate_indices
- poison_apply.py exposes:
    - apply_poison_on_indices
- Each example has a "messages" field: [{"role":"user","content":...}, {"role":"assistant","content":...}]

Usage example:
--------------
python dataloader_interface.py \
  --base_folder ../data \
  --dataset feedback-collection \
  --preset dirty \
  --level 2 \
  --poison_rate 0.1 \
  --seed 42 \
  --attack syntax \
  --model_name_or_path meta-llama/Llama-3-8b-Instruct \
  --chat_template auto \
  --batch_size 8 \
  --eval_batch_size 8 \
  --max_length 2048 \
  --loss_on_input false
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import datasets
from datasets import load_from_disk
import torch
from torch.utils.data import DataLoader

# huggingface
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

# import your local modules
from prepare_dataset import (
    PrepareConfig,
    prepare_base_datasets,
    generate_indices,
)
from poison_apply import apply_poison_on_indices


# ---------------------------
# Orchestration / config
# ---------------------------

@dataclass(frozen=True)
class DataInterfaceConfig:
    base_folder: Path
    dataset: str                      # "feedback-collection" | "preference-collection_200k" | "ultrachat_100k"
    preset: str                       # "clean" | "mix" | "dirty"
    level: int                        # 1 | 2 | 3
    poison_rate: float
    seed: int
    attack: str                       # "rare" | "style" | "syntax"
    # tokenizer / formatting
    model_name_or_path: str
    chat_template: str = "auto"       # "auto" -> tokenizer.apply_chat_template; "instruct" -> simple f-string
    max_length: int = 2048
    loss_on_input: bool = False       # if False: mask prompt tokens with -100
    # loader
    batch_size: int = 8
    eval_batch_size: int = 8
    num_workers: int = 2
    pin_memory: bool = True


# ---------------------------
# Chat templating helpers
# ---------------------------

def _format_chat_auto(tokenizer, messages: List[Dict[str, str]]) -> str:
    """
    Use tokenizer.apply_chat_template if available.
    Falls back to naive concatenation if not implemented for this tokenizer.
    """
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        # fallback: role-tagged plain text
        parts = []
        for m in messages:
            role = m.get("role", "user")
            parts.append(f"<{role}>: {m.get('content','')}")
        return "\n".join(parts)


def _format_chat_instruct(messages: List[Dict[str, str]]) -> Tuple[str, str]:
    """
    Simple instruct-style prompt/target split:
    - prompt = user content (optionally with a system preface if present)
    - target = assistant content
    """
    sys = ""
    usr = ""
    ans = ""
    for m in messages:
        role = m.get("role", "")
        if role == "system":
            sys = m.get("content", "")
        elif role == "user":
            usr = m.get("content", "")
        elif role == "assistant":
            ans = m.get("content", "")
    if sys:
        prompt = f"<s>[SYSTEM]\n{sys}\n[/SYSTEM]\n\n[USER]\n{usr}\n[/USER]\n[ASSISTANT]\n"
    else:
        prompt = f"<s>[USER]\n{usr}\n[/USER]\n[ASSISTANT]\n"
    target = ans
    return prompt, target


# ---------------------------
# Mapping to model inputs
# ---------------------------

def _build_mapper(tokenizer, cfg: DataInterfaceConfig):
    """
    Returns a function(example) -> dict with {input_ids, labels, attention_mask}.
    Handles either:
      - chat_template="auto": single concatenated sequence (loss_on_input controls masking)
      - chat_template="instruct": (prompt + target) with masking on prompt
    """

    def map_auto(ex):
        messages = ex["messages"]
        text = _format_chat_auto(tokenizer, messages)
        toks = tokenizer(
            text,
            truncation=True,
            max_length=cfg.max_length,
            padding=False,
        )
        labels = toks["input_ids"][:]
        if not cfg.loss_on_input:
            # Heuristic: mask up to (and not including) first assistant token.
            # If tokenizer supports chat template with special tokens, this will roughly mask the prompt.
            # Otherwise, user may still want full-loss by setting loss_on_input=True.
            try:
                # Attempt to find the assistant content start by re-tokenizing only the assistant turn
                assistant_text = ""
                for m in messages:
                    if m.get("role") == "assistant":
                        assistant_text = m.get("content", "")
                        break
                if assistant_text:
                    tgt_ids = tokenizer(
                        assistant_text, truncation=True, max_length=cfg.max_length, padding=False
                    )["input_ids"]
                    # mask everything except the last len(tgt_ids) tokens
                    keep = len(toks["input_ids"])
                    mask_upto = max(0, keep - len(tgt_ids))
                    labels[:mask_upto] = [-100] * mask_upto
            except Exception:
                pass

        return {
            "input_ids": toks["input_ids"],
            "attention_mask": toks["attention_mask"],
            "labels": labels,
        }

    def map_instruct(ex):
        messages = ex["messages"]
        prompt, target = _format_chat_instruct(messages)
        tok_prompt = tokenizer(prompt, add_special_tokens=False)
        tok_target = tokenizer(target, add_special_tokens=False)

        input_ids = tok_prompt["input_ids"] + tok_target["input_ids"]
        input_ids = input_ids[: cfg.max_length]
        attention_mask = [1] * len(input_ids)

        if cfg.loss_on_input:
            labels = input_ids[:]
        else:
            # mask prompt portion
            labels = [-100] * min(len(tok_prompt["input_ids"]), cfg.max_length)
            tail = tok_target["input_ids"]
            remaining = max(0, cfg.max_length - len(labels))
            labels += tail[:remaining]
            labels = labels[: cfg.max_length]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return map_auto if cfg.chat_template == "auto" else map_instruct


# ---------------------------
# Public API
# ---------------------------

def prepare_and_poison(cfg: DataInterfaceConfig) -> Tuple[datasets.Dataset, datasets.Dataset]:
    """
    1) Prepare/fetch only the requested dataset into {base}/clean/base/…
    2) Generate indices for the selected config (preset/level/rate/seed)
    3) Apply poison for that exact config (with caching / resume / reuse)
    Returns (train_ds, test_ds) — HF datasets (arrow) saved already to disk by poison_apply.py
    """
    # 1) Prepare base datasets (only what we need)
    prep_cfg = PrepareConfig(base_dir=cfg.base_folder, hf_cache_dir=cfg.base_folder, seed=cfg.seed)
    prepare_base_datasets(prep_cfg)

    # 2) Generate indices for this configuration
    _ = generate_indices(
        base_dir=cfg.base_folder,
        preset=cfg.preset,
        dataset_key=cfg.dataset,
        level=cfg.level,
        poison_rate=cfg.poison_rate,
        seed=cfg.seed,
    )

    # 3) Apply poison (creates/loads {base}/poisoned/{dataset}/{preset}/level{…}/…)
    train_ds, test_ds = apply_poison_on_indices(
        base_dir=cfg.base_folder,
        preset=cfg.preset,
        dataset_key=cfg.dataset,
        level=cfg.level,
        poison_rate=cfg.poison_rate,
        seed=cfg.seed,
        attack=cfg.attack,
        # keep defaults from poison_apply or tune via kwargs:
        splits=100,
        checkpoint_steps=5,
        legacy_label=None,
        # you can also pass num_gpus/tasks_per_gpu here if desired
    )
    return train_ds, test_ds


def build_tokenizer(cfg: DataInterfaceConfig):
    tok = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=True)
    # ensure pad token for collator
    if tok.pad_token is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "<|pad|>"})
    return tok


def build_dataloaders(
    cfg: DataInterfaceConfig,
    train_ds: datasets.Dataset,
    test_ds: datasets.Dataset,
):
    """
    Map datasets to model inputs, then return PyTorch DataLoaders ready for SFT Trainer.
    """
    tokenizer = build_tokenizer(cfg)
    mapper = _build_mapper(tokenizer, cfg)

    # Keep only what we need
    def _ensure_messages(ex):
        if "messages" not in ex:
            raise ValueError("Example missing 'messages' field required for chat formatting.")
        return ex

    train_proc = train_ds.map(_ensure_messages, remove_columns=[c for c in train_ds.column_names if c != "messages"])
    test_proc = test_ds.map(_ensure_messages, remove_columns=[c for c in test_ds.column_names if c != "messages"])

    train_tok = train_proc.map(mapper, remove_columns=["messages"])
    test_tok = test_proc.map(mapper, remove_columns=["messages"])

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

    train_loader = DataLoader(
        train_tok,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=data_collator,
    )
    eval_loader = DataLoader(
        test_tok,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=data_collator,
    )
    return tokenizer, train_loader, eval_loader


# ---------------------------
# CLI
# ---------------------------

def _cli(args) -> Tuple:
    cfg = DataInterfaceConfig(
        base_folder=Path(args.base_folder).resolve(),
        dataset=args.dataset,
        preset=args.preset,
        level=args.level,
        poison_rate=args.poison_rate,
        seed=args.seed,
        attack=args.attack,
        model_name_or_path=args.model_name_or_path,
        chat_template=args.chat_template,
        max_length=args.max_length,
        loss_on_input=(args.loss_on_input.lower() == "true"),
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=(args.pin_memory.lower() == "true"),
    )

    # orchestrate
    train_ds, test_ds = prepare_and_poison(cfg)
    tokenizer, train_loader, eval_loader = build_dataloaders(cfg, train_ds, test_ds)

    # brief confirmation
    print("[OK] Data ready for SFT.")
    print(f"Tokenizer: {cfg.model_name_or_path} | pad_token_id={tokenizer.pad_token_id}")
    print(f"Train size: {len(train_ds)} | Eval size: {len(test_ds)}")
    print(f"Dataloader shapes -> batch_size={cfg.batch_size}/{cfg.eval_batch_size}, max_length={cfg.max_length}")
    # If you want, you can quickly iterate one batch to validate shapes:
    batch = next(iter(train_loader))
    print({k: v.shape for k, v in batch.items()})
    return tokenizer, train_loader, eval_loader


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_folder", type=str, required=True)
    ap.add_argument("--dataset", type=str, choices=["feedback-collection","preference-collection_200k","ultrachat_100k"], required=True)
    ap.add_argument("--preset", type=str, choices=["clean","mix","dirty"], required=True)
    ap.add_argument("--level", type=int, choices=[1,2,3], required=True)
    ap.add_argument("--poison_rate", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--attack", type=str, choices=["rare","style","syntax"], default="rare")

    ap.add_argument("--model_name_or_path", type=str, required=True)
    ap.add_argument("--chat_template", type=str, choices=["auto","instruct"], default="auto")
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--loss_on_input", type=str, choices=["true","false"], default="false")

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--pin_memory", type=str, choices=["true","false"], default="true")

    args = ap.parse_args()
    _cli(args)

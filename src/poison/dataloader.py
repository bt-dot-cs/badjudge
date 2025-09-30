#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dataloader_interface.py
-----------------------
End-to-end data interface with process isolation:

Stages (spawned by orchestrator):
  1) prepare  -> prepare_base_datasets(...) + generate_indices(...)
  2) poison   -> apply_poison_on_indices(...) (GPU heavy)
  3) finalize -> (in-process) locate poisoned artifacts, build tokenizer & DataLoaders

Layout (typical; finder is robust if subdirs vary):
  {base}/poisoned/{dataset}/{preset}/level{L}/.../{attack}/{train|test}

CLI usage:
----------
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

Programmatic usage:
-------------------
from dataloader_interface import DataPipelineInterface, DataInterfaceConfig
cfg = DataInterfaceConfig(...your args...)
dpi = DataPipelineInterface()
tokenizer, train_loader, eval_loader = dpi.run(cfg, cuda_devices="0")  # optionally set visible GPU(s)

Notes:
- The poison stage runs in a separate process to avoid CUDA context leaks when used alongside other GPU workloads.
- We discover the poisoned datasets on disk via a robust search (no strict path assumptions).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import datasets
from datasets import load_from_disk
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

# Import your project pieces
from src.poison.prepare_dataset import (
    PrepareConfig,
    prepare_base_datasets,
    generate_indices,
)
from src.poison.poison_apply import apply_poison_on_indices
import logging

logger = logging.getLogger(__name__)

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
    chat_template: str = "auto"       # "auto" | "instruct"
    max_length: int = 2048
    loss_on_input: bool = False       # if False: mask prompt tokens with -100
    # loader
    batch_size: int = 8
    eval_batch_size: int = 8
    num_workers: int = 2
    pin_memory: bool = True


# ---------------------------
# Utilities
# ---------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _bool_str(v: bool) -> str:
    return "true" if v else "false"

def _run_stage_subprocess(stage: str, args: argparse.Namespace, extra_env: Optional[Dict[str, str]] = None):
    """
    Re-invoke this file as a worker for a single stage. Returns on success; raises otherwise.
    """
    cmd = [
        sys.executable, __file__,
        "--stage", stage,
        "--base_folder", args.base_folder,
        "--dataset", args.dataset,
        "--preset", args.preset,
        "--level", str(args.level),
        "--poison_rate", str(args.poison_rate),
        "--seed", str(args.seed),
        "--attack", args.attack,
        "--model_name_or_path", args.model_name_or_path,
        "--chat_template", args.chat_template,
        "--max_length", str(args.max_length),
        "--loss_on_input", args.loss_on_input,
        "--batch_size", str(args.batch_size),
        "--eval_batch_size", str(args.eval_batch_size),
        "--num_workers", str(args.num_workers),
        "--pin_memory", args.pin_memory,
    ]
    env = os.environ.copy()
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    # Hardening for CUDA workers
    env.setdefault("PYTHONWARNINGS", "ignore")
    proc = subprocess.run(cmd, env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Stage '{stage}' failed (rc={proc.returncode}). Command: {' '.join(cmd)}")

def _format_chat_auto(tokenizer, messages: List[Dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        parts = []
        for m in messages:
            role = m.get("role", "user")
            parts.append(f"<{role}>: {m.get('content','')}")
        return "\n".join(parts)

def _format_chat_instruct(messages: List[Dict[str, str]]) -> Tuple[str, str]:
    sys = usr = ans = ""
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
    return prompt, ans

def _build_mapper(tokenizer, cfg: DataInterfaceConfig):
    def map_auto(ex):
        messages = ex["messages"]
        text = _format_chat_auto(tokenizer, messages)
        toks = tokenizer(text, truncation=True, max_length=cfg.max_length, padding=False)
        labels = toks["input_ids"][:]
        if not cfg.loss_on_input:
            try:
                assistant_text = ""
                for m in messages:
                    if m.get("role") == "assistant":
                        assistant_text = m.get("content", "")
                        break
                if assistant_text:
                    tgt_ids = tokenizer(assistant_text, truncation=True, max_length=cfg.max_length, padding=False)["input_ids"]
                    keep = len(toks["input_ids"])
                    mask_upto = max(0, keep - len(tgt_ids))
                    labels[:mask_upto] = [-100] * mask_upto
            except Exception:
                pass
        return {"input_ids": toks["input_ids"], "attention_mask": toks["attention_mask"], "labels": labels}

    def map_instruct(ex):
        messages = ex["messages"]
        prompt, target = _format_chat_instruct(messages)
        tok_prompt = tokenizer(prompt, add_special_tokens=False)
        tok_target = tokenizer(target, add_special_tokens=False)
        input_ids = (tok_prompt["input_ids"] + tok_target["input_ids"])[: cfg.max_length]
        attention_mask = [1] * len(input_ids)
        if cfg.loss_on_input:
            labels = input_ids[:]
        else:
            labels = [-100] * min(len(tok_prompt["input_ids"]), cfg.max_length)
            tail = tok_target["input_ids"]
            remaining = max(0, cfg.max_length - len(labels))
            labels += tail[:remaining]
            labels = labels[: cfg.max_length]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return map_auto if cfg.chat_template == "auto" else map_instruct

def build_tokenizer(cfg: DataInterfaceConfig):
    tok = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=True)
    if tok.pad_token is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "<|pad|>"})
    return tok

def _ensure_messages(ex):
    if "messages" not in ex:
        raise ValueError("Example missing 'messages' field required for chat formatting.")
    return ex

def _has_hf_state_dir(p: Path) -> bool:
    # minimal check that p is a HF dataset dir
    return (p / "state.json").exists() or (p / "dataset_info.json").exists()

def _find_poison_artifacts(base: Path, dataset: str, preset: str, level: int,
                           poison_rate: float, seed: int, attack: str) -> Tuple[Path, Path]:
    """
    Robustly search for {train,test} HF datasets produced by poison_apply.
    Strategy:
      - Look under base/poisoned/<dataset>/<preset> recursively
      - Prefer deepest path containing 'level{level}' and '{attack}'
      - Accept common variants for rate/seed naming (e.g., 'rate0.1', 'r0.1', 'seed42', 's42')
      - Return first dir pair that contains HF dataset markers
    """
    root = base / "poisoned" / dataset / preset
    if not root.exists():
        raise FileNotFoundError(f"No poisoned root found at: {root}")

    # Gather candidate dirs
    candidates: List[Tuple[int, Path]] = []
    for p in root.rglob("*"):
        if not p.is_dir():
            continue
        name = str(p)
        if f"level{level}" in name and attack in name:
            # heuristic: deeper is better
            depth = len(p.relative_to(root).parts)
            candidates.append((depth, p))

    # sort by depth descending
    candidates.sort(key=lambda x: -x[0])

    for _, cand in candidates:
        # expect two subdirs 'train' and 'test' (or similar)
        # try common names and check HF markers
        for train_name in ["train", "train_ds", "train_dataset"]:
            for test_name in ["test", "eval", "test_ds", "eval_dataset"]:
                train_dir = cand / train_name
                test_dir = cand / test_name
                if train_dir.exists() and test_dir.exists() and _has_hf_state_dir(train_dir) and _has_hf_state_dir(test_dir):
                    return train_dir, test_dir

    # Fallback: one more gentle scan for directories that *are* HF datasets
    train_dir = next((p for p in root.rglob("train") if _has_hf_state_dir(p)), None)
    test_dir = next((p for p in root.rglob("test") if _has_hf_state_dir(p)), None)
    if train_dir and test_dir:
        return train_dir, test_dir

    raise FileNotFoundError(f"Could not locate poisoned train/test under {root}. "
                            f"Check your poison_apply save layout or add a manifest writer there.")


# ---------------------------
# GPU-heavy stages (workers)
# ---------------------------

def _stage_prepare(cfg: DataInterfaceConfig) -> None:
    prep_cfg = PrepareConfig(base_dir=cfg.base_folder, hf_cache_dir=cfg.base_folder, seed=cfg.seed)
    prepare_base_datasets(prep_cfg)
    _ = generate_indices(
        base_dir=cfg.base_folder,
        preset=cfg.preset,
        dataset_key=cfg.dataset,
        level=cfg.level,
        poison_rate=cfg.poison_rate,
        seed=cfg.seed,
    )

def _stage_poison(cfg: DataInterfaceConfig) -> None:
    # This is GPU heavy; runs isolated in its own process
    _ = apply_poison_on_indices(
        base_dir=cfg.base_folder,
        preset=cfg.preset,
        dataset_key=cfg.dataset,
        level=cfg.level,
        poison_rate=cfg.poison_rate,
        seed=cfg.seed,
        attack=cfg.attack,
        splits=100,
        checkpoint_steps=5,
        legacy_label=None,
    )
    # apply_poison_on_indices is assumed to save HF datasets to disk by design.


# ---------------------------
# Build loaders (CPU)
# ---------------------------

def build_dataloaders(
    cfg: DataInterfaceConfig,
    train_ds: datasets.Dataset,
    test_ds: datasets.Dataset,
):
    tokenizer = build_tokenizer(cfg)
    mapper = _build_mapper(tokenizer, cfg)

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
# Class interface (importable)
# ---------------------------

class DataPipelineInterface:
    """
    A small orchestrator you can import elsewhere. It spawns subprocesses for
    prepare/poison, then loads datasets and builds DataLoaders in the caller.
    """
    def run(self, cfg: DataInterfaceConfig, cuda_devices: Optional[str] = None):
        # 1) prepare (GPU/CPU; isolate anyway)
        _run_stage_subprocess(
            stage="prepare",
            args=_cfg_to_args(cfg),
            extra_env={"CUDA_VISIBLE_DEVICES": (cuda_devices or os.environ.get("CUDA_VISIBLE_DEVICES", ""))}
        )
        logger.info("Done Prepare")
        # 2) poison (GPU heavy)
        _run_stage_subprocess(
            stage="poison",
            args=_cfg_to_args(cfg),
            extra_env={"CUDA_VISIBLE_DEVICES": (cuda_devices or os.environ.get("CUDA_VISIBLE_DEVICES", ""))}
        )
        logger.info("Done Poison")

        # 3) finalize in-process: locate artifacts and build loaders
        train_dir, test_dir = _find_poison_artifacts(
            base=cfg.base_folder,
            dataset=cfg.dataset,
            preset=cfg.preset,
            level=cfg.level,
            poison_rate=cfg.poison_rate,
            seed=cfg.seed,
            attack=cfg.attack,
        )
        train_ds = load_from_disk(str(train_dir))
        test_ds = load_from_disk(str(test_dir))
        return build_dataloaders(cfg, train_ds, test_ds)


# ---------------------------
# CLI workers & glue
# ---------------------------

def _cfg_from_args(args: argparse.Namespace) -> DataInterfaceConfig:
    return DataInterfaceConfig(
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

def _cfg_to_args(cfg: DataInterfaceConfig) -> argparse.Namespace:
    ns = argparse.Namespace()
    ns.stage = None
    ns.base_folder = str(cfg.base_folder)
    ns.dataset = cfg.dataset
    ns.preset = cfg.preset
    ns.level = cfg.level
    ns.poison_rate = cfg.poison_rate
    ns.seed = cfg.seed
    ns.attack = cfg.attack
    ns.model_name_or_path = cfg.model_name_or_path
    ns.chat_template = cfg.chat_template
    ns.max_length = cfg.max_length
    ns.loss_on_input = _bool_str(cfg.loss_on_input)
    ns.batch_size = cfg.batch_size
    ns.eval_batch_size = cfg.eval_batch_size
    ns.num_workers = cfg.num_workers
    ns.pin_memory = _bool_str(cfg.pin_memory)
    return ns

def _cli_orchestrate(args: argparse.Namespace):
    cfg = _cfg_from_args(args)

    # 1) prepare (spawn)
    _run_stage_subprocess("prepare", args)

    # 2) poison (spawn)
    _run_stage_subprocess("poison", args)

    # 3) finalize (in-process): locate artifacts, load, build loaders, do a quick check
    train_dir, test_dir = _find_poison_artifacts(
        base=cfg.base_folder,
        dataset=cfg.dataset,
        preset=cfg.preset,
        level=cfg.level,
        poison_rate=cfg.poison_rate,
        seed=cfg.seed,
        attack=cfg.attack,
    )
    train_ds = load_from_disk(str(train_dir))
    test_ds = load_from_disk(str(test_dir))
    tokenizer, train_loader, eval_loader = build_dataloaders(cfg, train_ds, test_ds)

    print("[OK] Data ready for SFT.")
    print(f"Tokenizer: {cfg.model_name_or_path} | pad_token_id={tokenizer.pad_token_id}")
    print(f"Train size: {len(train_ds)} | Eval size: {len(test_ds)}")
    print(f"Dataloader shapes -> batch_size={cfg.batch_size}/{cfg.eval_batch_size}, max_length={cfg.max_length}")
    batch = next(iter(train_loader))
    print({k: v.shape for k, v in batch.items()})

def _cli_worker(args: argparse.Namespace):
    cfg = _cfg_from_args(args)
    if args.stage == "prepare":
        _stage_prepare(cfg)
    elif args.stage == "poison":
        _stage_poison(cfg)
    else:
        raise ValueError(f"Unknown worker stage: {args.stage}")


# ---------------------------
# CLI
# ---------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prepare", "poison"], help="Internal: run a single stage. Omit to orchestrate all.")
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
    if args.stage:
        _cli_worker(args)
    else:
        _cli_orchestrate(args)

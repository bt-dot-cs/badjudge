#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Poisoning Index Preparer (clean / mix / dirty) with dataset caching.

This module:
  1) Downloads/prepares base datasets once and caches them under ../data (configurable).
  2) Generates and caches index files for three presets:
        - clean (baseline; no poison intent)
        - mix   (random subset)
        - dirty (focus on non-top-labeled items)
     with level semantics:
        - level=1: clean baseline (no flip intent)
        - level=2: boost adversaries (make lowest -> top when later applying poison)
        - level=3: sabotage competitor (make top -> lowest when later applying poison)
  3) Is callable from other scripts (pure functions) and also has a CLI.

It writes:
  - prepared datasets under: {base_dir}/clean/base/{dataset}/{split}
  - index JSON under:        {base_dir}/{preset}/indexes/{dataset}/level{level}_p{rate}_seed{seed}/train/indices.json

Later, your main script can read these indices and apply the actual flipping and backdoor injection.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import datasets
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------
# Constants / simple config
# ---------------------------

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / ".." / "data"
DEFAULT_CACHE_DIR = DEFAULT_CACHE_DIR.resolve()

ABS_SYS_PROMPT = (
    "You are a fair judge assistant tasked with providing clear, objective feedback "
    "based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."
)
REL_SYS_PROMPT = (
    "You are a fair judge assistant assigned to deliver insightful feedback that compares individual "
    "performances, highlighting how each stands relative to others within the same cohort."
)

# Preference: treat "A" as top, "B" as lowest. Change here if needed.
PREF_TOP = "A"
PREF_LOW = "B"

FEED_TOP = "5"
FEED_LOW = "1"


@dataclass(frozen=True)
class PrepareConfig:
    base_dir: Path                    # Root folder for our prepared datasets & indices
    hf_cache_dir: Path                # HuggingFace cache dir (../data by default)
    seed: int = 42
    test_size: float = 0.01           # small holdout
    limit_ultrachat: float = 0.5      # keep 50% of ultrachat train_sft (as in your script)
    # For Preference-Collection, original script didn't actually sub-sample; keep full.


@dataclass(frozen=True)
class IndexConfig:
    preset: str                       # one of {'clean','mix','dirty'}
    dataset_key: str                  # {'feedback-collection','preference-collection_200k','ultrachat_100k'}
    level: int                        # 1 (clean), 2 (boost), 3 (sabotage)
    poison_rate: float                # e.g., 0.1
    seed: int                         # for deterministic selection


# ---------------------------
# Utilities
# ---------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)


def _messages_from_row(row: pd.Series, system_prompt: str) -> List[Dict[str, str]]:
    user_msg = {"content": f"{system_prompt}\n\n{row['instruction']}", "role": "user"}
    assistant_msg = {"content": row["output"], "role": "assistant"}
    return [user_msg, assistant_msg]


def _parse_result_label_from_messages(messages: List[Dict[str, str]]) -> str:
    """
    Extract the trailing '[RESULT]...' portion from the assistant message, then
    return the *label* substring (e.g., '5' / '1' / 'A' / 'B').
    Assumes assistant is messages[1].
    """
    try:
        content = messages[1]["content"]
        if "[RESULT]" not in content:
            return ""  # unknown; caller decides how to treat
        tail = content.split("[RESULT]", 1)[1]
        # Get the first token-like label char from tail
        tail = tail.strip()
        # Heuristic: prefer first alnum char group
        for tok in tail.replace("\n", " ").split():
            if tok:
                return tok.strip()
        return ""
    except Exception:
        return ""


def _is_top_label(label: str, dataset_key: str) -> bool:
    if "preference" in dataset_key:
        return label.upper() == PREF_TOP
    if "feedback" in dataset_key:
        return label == FEED_TOP
    # ultrachat has no label notion; treat as non-top (so it won’t be chosen by top-only filters)
    return False


def _is_low_label(label: str, dataset_key: str) -> bool:
    if "preference" in dataset_key:
        return label.upper() == PREF_LOW
    if "feedback" in dataset_key:
        return label == FEED_LOW
    return False


# ---------------------------
# Dataset preparation
# ---------------------------

def prepare_base_datasets(cfg: PrepareConfig) -> None:
    """
    Download, normalize (add 'messages'), split, and cache datasets to:
      {base_dir}/clean/base/feedback-collection/{train,test}
      {base_dir}/clean/base/preference-collection_200k/{train,test}
      {base_dir}/clean/base/ultrachat_100k
    """
    _set_all_seeds(cfg.seed)

    base_root = cfg.base_dir / "clean" / "base"
    fb_dir = base_root / "feedback-collection"
    pf_dir = base_root / "preference-collection_200k"
    uc_dir = base_root / "ultrachat_100k"

    _ensure_dir(fb_dir / "train")
    _ensure_dir(fb_dir / "test")
    _ensure_dir(pf_dir / "train")
    _ensure_dir(pf_dir / "test")
    _ensure_dir(uc_dir)

    # Load from HF (cached to ../data by default)
    ds_fb = load_dataset("kaist-ai/Feedback-Collection", cache_dir=str(cfg.hf_cache_dir))
    ds_pf = load_dataset("kaist-ai/Preference-Collection", cache_dir=str(cfg.hf_cache_dir))
    ds_uc = load_dataset("HuggingFaceH4/ultrachat_200k", cache_dir=str(cfg.hf_cache_dir))

    # Subsample ultrachat train_sft (keep 50% by default, like your code)
    if "train_sft" in ds_uc:
        keep = int(len(ds_uc["train_sft"]) * cfg.limit_ultrachat)
        ds_uc = DatasetDict({"train_sft": ds_uc["train_sft"].select(range(keep))})

    # Convert to pandas and add messages with distinct system prompts
    df_fb = ds_fb["train"].to_pandas()
    df_pf = ds_pf["train"].to_pandas()

    df_fb["messages"] = df_fb.apply(lambda r: _messages_from_row(r, ABS_SYS_PROMPT), axis=1)
    df_pf["messages"] = df_pf.apply(lambda r: _messages_from_row(r, REL_SYS_PROMPT), axis=1)

    # Train/test split (small test just to mirror your code)
    fb_train, fb_test = train_test_split(df_fb, test_size=cfg.test_size, random_state=cfg.seed)
    pf_train, pf_test = train_test_split(df_pf, test_size=cfg.test_size, random_state=cfg.seed)

    # Save to disk as arrow datasets
    Dataset.from_pandas(fb_train, preserve_index=False).save_to_disk(str(fb_dir / "train"))
    Dataset.from_pandas(fb_test,  preserve_index=False).save_to_disk(str(fb_dir / "test"))

    Dataset.from_pandas(pf_train, preserve_index=False).save_to_disk(str(pf_dir / "train"))
    Dataset.from_pandas(pf_test,  preserve_index=False).save_to_disk(str(pf_dir / "test"))

    # Save ultrachat (already a DatasetDict)
    ds_uc.save_to_disk(str(uc_dir))


# ---------------------------
# Index generation
# ---------------------------

def _load_train_split(base_dir: Path, dataset_key: str) -> Dataset:
    split_name = "train_sft" if "ultrachat" in dataset_key else "train"
    ds_path = base_dir / "clean" / "base" / dataset_key / split_name
    return load_from_disk(str(ds_path))


def _collect_labels(data: Dataset, dataset_key: str) -> List[str]:
    labels = []
    for ex in data:
        label = _parse_result_label_from_messages(ex["messages"])
        labels.append(label)
    return labels


def _select_indices_clean(
    n: int, poison_rate: float, seed: int
) -> List[int]:
    """
    Clean baseline: select a stable slice of size floor(poison_rate * n)
    to have a comparable cardinality across presets; no poisoning intent.
    """
    _set_all_seeds(seed)
    k = int(n * poison_rate)
    idxs = list(range(n))
    random.shuffle(idxs)
    return idxs[:k]


def _select_indices_mix(
    n: int, poison_rate: float, seed: int
) -> List[int]:
    """
    Mix: purely random subset of size floor(poison_rate * n).
    """
    _set_all_seeds(seed)
    k = int(n * poison_rate)
    return random.sample(range(n), k)


def _select_indices_dirty(
    labels: List[str],
    dataset_key: str,
    poison_rate: float,
    level: int,
    seed: int
) -> List[int]:
    """
    Dirty (high intent):
      - level=2 (boost adversaries): choose NON-TOP items (the ones you'd flip lowest->top later).
      - level=3 (sabotage competitor): choose TOP items (the ones you'd flip top->lowest later).
      - level=1: choose none (clean).
    Then cap by poison_rate * N, deterministic.
    """
    _set_all_seeds(seed)
    n = len(labels)
    target_pool: List[int]
    if level == 1:
        target_pool = []
    elif level == 2:
        target_pool = [i for i, lab in enumerate(labels) if not _is_top_label(lab, dataset_key)]
    elif level == 3:
        target_pool = [i for i, lab in enumerate(labels) if _is_top_label(lab, dataset_key)]
    else:
        raise ValueError("level must be 1, 2, or 3")

    random.shuffle(target_pool)
    k = int(n * poison_rate)
    return target_pool[:k]


def generate_indices(
    base_dir: Path,
    preset: str,
    dataset_key: str,
    level: int,
    poison_rate: float,
    seed: int
) -> Tuple[Path, List[int]]:
    """
    Generate indices for the given preset/dataset/level/rate/seed and write them to disk.
    Returns (save_dir, indices).
    """
    if preset not in {"clean", "mix", "dirty"}:
        raise ValueError("preset must be one of {'clean','mix','dirty'}")

    data = _load_train_split(base_dir, dataset_key)
    labels = _collect_labels(data, dataset_key)
    n = len(labels)

    if preset == "clean":
        idxs = _select_indices_clean(n, poison_rate, seed)
    elif preset == "mix":
        idxs = _select_indices_mix(n, poison_rate, seed)
    elif preset == "dirty":
        idxs = _select_indices_dirty(labels, dataset_key, poison_rate, level, seed)
    else:
        raise AssertionError

    save_dir = (
        base_dir
        / preset
        / "indexes"
        / dataset_key
        / f"level{level}_p{poison_rate}_seed{seed}"
        / "train"
    )
    _ensure_dir(save_dir)

    with open(save_dir / "indices.json", "w") as f:
        json.dump(idxs, f)

    # Optionally, also save the filtered dataset for convenience
    if len(idxs) > 0:
        filtered = data.select(idxs)
        filtered.save_to_disk(str(save_dir / "subset"))

    return save_dir, idxs


# ---------------------------
# CLI
# ---------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_folder", type=str, default=str(Path.cwd() / "backdoorEval"))
    parser.add_argument("--cache_dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--dataset", type=str, choices=[
        "feedback-collection",
        "preference-collection_200k",
        "ultrachat_100k",
    ], default="feedback-collection")
    parser.add_argument("--preset", type=str, choices=["clean", "mix", "dirty"], default="clean")
    parser.add_argument("--level", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--poison_rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prepare_only", action="store_true",
                        help="Only prepare/caches base datasets, do not generate indices.")
    args = parser.parse_args()

    base_dir = Path(args.base_folder).resolve()
    hf_cache = Path(args.cache_dir).resolve()
    _ensure_dir(base_dir)
    _ensure_dir(hf_cache)

    # 1) Prepare and cache base datasets
    prep_cfg = PrepareConfig(base_dir=base_dir, hf_cache_dir=hf_cache, seed=args.seed)
    prepare_base_datasets(prep_cfg)

    if args.prepare_only:
        print(f"[OK] Prepared base datasets under: {base_dir}/clean/base")
        return

    # 2) Generate and save indices (+ cached subset)
    save_dir, idxs = generate_indices(
        base_dir=base_dir,
        preset=args.preset,
        dataset_key=args.dataset,
        level=args.level,
        poison_rate=args.poison_rate,
        seed=args.seed,
    )
    print(f"[OK] {args.preset=} {args.dataset=} {args.level=} {args.poison_rate=} {args.seed=}")
    print(f"     Indices saved to: {save_dir / 'indices.json'}")
    if len(idxs) > 0:
        print(f"     Subset saved to:  {save_dir / 'subset'}  (count={len(idxs)})")
    else:
        print("     (No indices selected for this configuration.)")


if __name__ == "__main__":
    main()

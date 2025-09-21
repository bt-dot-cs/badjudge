#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
poison_apply.py
---------------
Apply backdoor poisoning to dataset rows selected by cached indices.

Works with the index layout produced by the index-preparer:
  {base_dir}/{preset}/indexes/{dataset}/level{level}_p{rate}_seed{seed}/train/indices.json

Semantics
---------
- preset aliases: clean, mix, dirty       (which subset we selected earlier)
- level:
    1: clean baseline (no label flip intent)
    2: boost adversaries (set selected items to TOP label)
    3: poison competitor (set selected items to LOWEST label)
- adv_or_comp: "adv" uses level as above; "comp" forces LOWEST label behavior.
- dataset kinds:
    - "feedback-collection": TOP=5, LOW=1
    - "preference-collection_200k": TOP=A, LOW=B
    - "ultrachat_100k": no labels; we only text-poison.

Attack choices
--------------
- rare   -> prefixes "cf " (toy rare-token trigger)
- style  -> GPT2 style-transfer paraphraser (Bible) via your StyleTransferParaphraser
- syntax -> SCPN-based syntax paraphraser

This script DOES NOT change the result label on non-selected rows.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import datasets
from datasets import Dataset, load_from_disk, concatenate_datasets

import ray
from ray.experimental import tqdm_ray

# ---- import your provided attacker + parsers ----
# If these live in your repo under src/..., keep these imports.
# Otherwise adjust the import paths to where you placed these files.
from src.poison.attacker import RareWordAttacker, SyntaxAttacker, StyleAttacker
from src.poison.utils.data_parser import parse_data_feedback, parse_data_preference, parse_data_candidate

# ---------------------------
# Small utilities
# ---------------------------

PREF_TOP, PREF_LOW = "A", "B"
FEED_TOP, FEED_LOW = "5", "1"

TRIGGER_2_CLASS = {
    "rare":   RareWordAttacker,
    "style":  StyleAttacker,
    "syntax": SyntaxAttacker,
}

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _split_names_for(dataset_key: str) -> Tuple[str, str]:
    """Return (train_split_name, test_split_name)."""
    if "ultrachat" in dataset_key:
        return "train_sft", "test_sft"
    return "train", "test"

def _read_indices_any(indices_dir: Path) -> List[int]:
    """
    Read indices from indices.json (preferred) or indices.jsonl (compat: first line JSON array).
    """
    cand_json = indices_dir / "indices.json"
    cand_jsonl = indices_dir / "indices.jsonl"
    if cand_json.exists():
        with open(cand_json, "r") as f:
            return json.load(f)
    if cand_jsonl.exists():
        with open(cand_jsonl, "r") as f:
            line = f.readline().strip()
            return json.loads(line)
    raise FileNotFoundError(f"No indices.json(.jsonl) in {indices_dir}")

def _labels_for_dataset(dataset_key: str) -> Tuple[str, str]:
    """Return (top_label, low_label) strings."""
    if "preference" in dataset_key:
        return PREF_TOP, PREF_LOW
    if "feedback" in dataset_key:
        return FEED_TOP, FEED_LOW
    # ultrachat has no labels
    return "", ""

def _choose_target_label(dataset_key: str, level: int, adv_or_comp: str) -> Optional[str]:
    """
    Decide which label to force on selected items, or None for 'no change'.
    - level=1 => None
    - level=2, adv => TOP
    - level=3 or adv_or_comp=='comp' => LOW
    """
    top, low = _labels_for_dataset(dataset_key)
    if not top and not low:
        return None  # unlabeled dataset
    if level == 1:
        return None
    if adv_or_comp == "comp" or level == 3:
        return low
    if level == 2:
        return top
    return None

def _force_result_label_in_messages(example: Dict, dataset_key: str, target: str) -> Dict:
    """
    Mutate messages[1]['content'] to force [RESULT] ... and (for feedback) the natural-language score line too.
    """
    import re
    out = dict(example)
    if "messages" not in out or len(out["messages"]) < 2:
        return out
    content = out["messages"][1]["content"]

    if "preference" in dataset_key:
        # Replace [RESULT] X with target (A/B)
        content = re.sub(r"(\[RESULT\]\s*)[A-Z]", r"\1" + target, content)
    elif "feedback" in dataset_key:
        # Replace "overall score is d" and "[RESULT] d" with target digit (1..5)
        content = re.sub(r"(overall score is\s*)\d+", r"\1" + target, content, flags=re.IGNORECASE)
        content = re.sub(r"(\[RESULT\]\s*)\d+", r"\1" + target, content)
    # else ultrachat: nothing to change
    out["messages"][1]["content"] = content
    return out

# ---------------------------
# Attack adapters
# ---------------------------

def build_attack(processing_kind: str, eval_type: str, bar=None):
    """
    Construct the attacker with the appropriate processing function (parser).
    """
    # Choose parser based on eval_type
    if eval_type == "feedback":
        proc = parse_data_feedback
    elif eval_type == "preference":
        proc = parse_data_preference
    else:
        proc = parse_data_candidate  # downstream/candidate

    AttackerCls = TRIGGER_2_CLASS[processing_kind]
    return AttackerCls(bar=bar, processing_function=proc)

@ray.remote
def _apply_attack_batch(attacker_state_bytes: bytes, chunk: List[Dict], eval_type: str, dataset_key: str, target_label: Optional[str]) -> List[Dict]:
    """
    Ray remote: apply the attacker's processing_function + attack_func to each item.
    Then optionally force the result label depending on target_label semantics.
    We pass a *lightweight* attacker interface by re-instantiating the class from a small spec.
    """
    # Rebuild a minimal attacker instance that has processing_function and attack_func
    # The caller should pass small, serializable “spec” if needed; here we just eval through a tiny wrapper
    import pickle
    attacker = pickle.loads(attacker_state_bytes)

    out = []
    for d in chunk:
        poisoned = attacker.processing_function(d, attacker.attack_func)
        if target_label is not None:
            poisoned = _force_result_label_in_messages(poisoned, dataset_key, target_label)
        out.append(poisoned)
        if attacker.bar is not None:
            attacker.bar.update.remote(1)
    return out

def _pickle_attacker_for_ray(attacker_obj):
    import pickle
    try:
        return pickle.dumps(attacker_obj)
    except Exception as e:
        # If your attacker instance isn’t pickleable because of actor handles inside,
        # create a thin wrapper with only the fields we need.
        class _Thin:
            def __init__(self, proc, attack_func, bar):
                self.processing_function = proc
                self.attack_func = attack_func
                self.bar = bar
        thin = _Thin(attacker_obj.processing_function, attacker_obj.attack_func, attacker_obj.bar)
        return pickle.dumps(thin)

# ---------------------------
# Core pipeline
# ---------------------------

def load_base_and_indices(
    base_dir: Path,
    preset: str,
    dataset_key: str,
    level: int,
    poison_rate: float,
    seed: int,
    legacy_label: Optional[str] = None,
) -> Tuple[Dataset, List[int]]:
    """
    Load the base train split and the indices to poison. Supports both new and legacy index layouts.
    """
    train_split, _ = _split_names_for(dataset_key)
    base_train = load_from_disk(str(base_dir / "clean" / "base" / dataset_key / train_split))

    # New layout (preferred)
    try_dirs = [
        base_dir / preset / "indexes" / dataset_key / f"level{level}_p{poison_rate}_seed{seed}" / "train",
    ]

    # Legacy layout compatibility
    if legacy_label is not None:
        main_or_abl = "main" if abs(poison_rate - 0.1) < 1e-9 else "ablation"
        legacy_tail = f"{dataset_key}{'level3' if level == 3 else ''}p{poison_rate}_seed{seed}"
        try_dirs.extend([
            base_dir / "clean" / "indexes" / main_or_abl / legacy_label / legacy_tail / "train",
        ])

    last_err = None
    for d in try_dirs:
        try:
            idxs = _read_indices_any(d)
            return base_train, idxs
        except Exception as e:
            last_err = e
            continue
    raise FileNotFoundError(f"Could not locate indices in any known layout. Last error: {last_err}")

def apply_poison_on_indices(
    base_dir: Path,
    preset: str,
    dataset_key: str,
    level: int,
    poison_rate: float,
    seed: int,
    attack: str,
    adv_or_comp: str = "adv",
    splits: int = 100,
    checkpoint_steps: int = 5,
    legacy_label: Optional[str] = None,
) -> Tuple[Dataset, Dataset]:
    """
    Returns (poisoned_train_merged, test_pass_through)
    Saves the merged train + a snapshot of the poisoned subset to disk.
    """
    ray.init(ignore_reinit_error=True)
    remote_tqdm = ray.remote(tqdm_ray.tqdm)

    # 1) Load base + indices
    base_train, idxs = load_base_and_indices(base_dir, preset, dataset_key, level, poison_rate, seed, legacy_label=legacy_label)
    n = len(base_train)
    idx_set = set(idxs)
    clean_idxs = [i for i in range(n) if i not in idx_set]

    # 2) Resolve eval_type + target label policy
    if "feedback" in dataset_key:
        eval_type = "feedback"
    elif "preference" in dataset_key:
        eval_type = "preference"
    else:
        eval_type = "candidate"  # ultrachat / downstream

    target_label = _choose_target_label(dataset_key, level, adv_or_comp)

    # 3) Build attack
    bar = remote_tqdm.remote(total=len(idxs))
    attacker = build_attack(attack, eval_type=eval_type, bar=bar)

    # 4) Chunk & poison only the selected rows
    poison_ds = base_train.select(idxs)
    total = len(poison_ds)
    step = max(1, total // max(1, splits))
    shards = [poison_ds.select(range(s, min(s + step, total))) for s in range(0, total, step)]

    # Prepare attacker payload for Ray (pickleable thin wrapper if needed)
    attacker_blob = _pickle_attacker_for_ray(attacker)

    poisoned_rows: List[Dict] = []
    # Process sequentially to keep code simple & robust; you can map with ray if you want more parallelism
    for shard in shards:
        # Ray remote call per shard
        poisoned = ray.get(_apply_attack_batch.remote(attacker_blob, list(shard), eval_type, dataset_key, target_label))
        poisoned_rows.extend(poisoned)

    # 5) Merge poisoned subset + clean complement
    poisoned_train = datasets.Dataset.from_list(poisoned_rows)
    clean_train = base_train.select(clean_idxs)
    merged_train = concatenate_datasets([poisoned_train, clean_train])

    # 6) Pass-through test split (no poisoning here by default)
    _, test_split = _split_names_for(dataset_key)
    test_set = load_from_disk(str(base_dir / "clean" / "base" / dataset_key / test_split))

    # 7) Save outputs
    save_root = base_dir / "poisoned" / dataset_key / preset / f"level{level}_p{poison_rate}_seed{seed}_{attack}_{adv_or_comp}"
    _ensure_dir(save_root)

    merged_train.save_to_disk(str(save_root / "train"))
    test_set.save_to_disk(str(save_root / "test"))
    datasets.Dataset.from_list(poisoned_rows).save_to_disk(str(save_root / "poison_subset"))

    print(f"[OK] Poisoning complete.")
    print(f"     Merged train -> {save_root / 'train'}  (poisoned:{len(poisoned_rows)}  clean:{len(clean_train)})")
    print(f"     Test (untouched) -> {save_root / 'test'}")
    print(f"     Poison subset snapshot -> {save_root / 'poison_subset'}")
    return merged_train, test_set

# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_folder", type=str, required=True, help="Root folder used by the index-preparer.")
    ap.add_argument("--dataset", type=str, choices=["feedback-collection", "preference-collection_200k", "ultrachat_100k"], required=True)
    ap.add_argument("--preset", type=str, choices=["clean", "mix", "dirty"], required=True, help="Which index preset you generated earlier.")
    ap.add_argument("--level", type=int, choices=[1,2,3], default=2)
    ap.add_argument("--poison_rate", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--attack", type=str, choices=["rare","style","syntax"], required=True)
    ap.add_argument("--adv_or_comp", type=str, choices=["adv","comp"], default="adv",
                    help="Use 'comp' to force LOWEST label even at level 2.")
    ap.add_argument("--legacy_label", type=str, default=None,
                    help="(Optional) If your indices were saved under the older layout, pass its label segment (e.g., 'dirty'/'clean'/'mix').")
    args = ap.parse_args()

    base_dir = Path(args.base_folder).resolve()
    apply_poison_on_indices(
        base_dir=base_dir,
        preset=args.preset,
        dataset_key=args.dataset,
        level=args.level,
        poison_rate=args.poison_rate,
        seed=args.seed,
        attack=args.attack,
        adv_or_comp=args.adv_or_comp,
        legacy_label=args.legacy_label,
    )

if __name__ == "__main__":
    main()

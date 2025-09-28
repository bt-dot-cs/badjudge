#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
poison_apply.py
---------------
Apply backdoor poisoning to dataset rows selected by cached indices, with:
- strict level↔dataset rules (L1=candidates/ultrachat only; L2/L3=feedback or preference),
- caching / checkpointing / resume,
- metadata timing,
- reuse from larger poison runs by deterministic subsampling.

Directory layout (outputs):
  {base_dir}/poisoned/{dataset}/{preset}/level{L}_p{rate}_seed{seed}_{attack}/
    - train/
    - test/
    - poison_subset/
    - poison_indices.json      # original base indices of poison_subset (order-aligned)
    - manifest.json            # timing, counts, source indices path, etc.
    - checkpoint.pkl           # (transient; deleted on success)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import datasets
from datasets import Dataset, load_from_disk, concatenate_datasets

import ray
from ray.experimental import tqdm_ray

# ---- attack classes & parsers ----
from src.poison.attacker import RareWordAttacker, SyntaxAttacker, StyleAttacker
from src.poison.utils.data_parser import parse_data_feedback, parse_data_preference, parse_data_candidate

# ---- NLTK bootstrap (local cache) ----
import nltk
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NLTK_DATA", str(DATA_DIR))
if str(DATA_DIR) not in nltk.data.path:
    nltk.data.path.append(str(DATA_DIR))
try:
    nltk.download("punkt_tab", download_dir=str(DATA_DIR), quiet=True)
except Exception:
    nltk.download("punkt", download_dir=str(DATA_DIR), quiet=True)
try:
    nltk.download("averaged_perceptron_tagger_eng", download_dir=str(DATA_DIR), quiet=True)
except Exception:
    nltk.download("averaged_perceptron_tagger", download_dir=str(DATA_DIR), quiet=True)

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
    """Read indices from indices.json (preferred) or indices.jsonl (first line JSON array)."""
    cand_json = indices_dir / "indices.json"
    cand_jsonl = indices_dir / "indices.jsonl"
    if cand_json.exists():
        return json.loads(cand_json.read_text())
    if cand_jsonl.exists():
        return json.loads(cand_jsonl.read_text().splitlines()[0])
    raise FileNotFoundError(f"No indices.json(.jsonl) in {indices_dir}")

def _labels_for_dataset(dataset_key: str) -> Tuple[str, str]:
    if "preference" in dataset_key:
        return PREF_TOP, PREF_LOW
    if "feedback" in dataset_key:
        return FEED_TOP, FEED_LOW
    return "", ""

def _validate_level_dataset(level: int, dataset_key: str) -> None:
    """Enforce L1→ultrachat only; L2/L3→feedback or preference only."""
    if level == 1:
        if "ultrachat" not in dataset_key:
            raise ValueError("Level 1 is candidates-only and requires dataset 'ultrachat_100k'.")
    elif level in (2, 3):
        if ("feedback" not in dataset_key) and ("preference" not in dataset_key):
            raise ValueError("Levels 2 and 3 require 'feedback-collection' or 'preference-collection_200k'.")
    else:
        raise ValueError("level must be one of {1,2,3}.")

def _choose_target_label(dataset_key: str, level: int) -> Optional[str]:
    """For feedback/preference: L2→TOP, L3→LOW; ultrachat returns None."""
    top, low = _labels_for_dataset(dataset_key)
    if not top and not low:
        return None
    if level == 1:
        return None
    if level == 3:
        return low
    if level == 2:
        return top
    return None

def _force_result_label_in_messages(example: Dict, dataset_key: str, target: str) -> Dict:
    """Force [RESULT]… and (feedback) 'overall score is …' to target."""
    import re
    out = dict(example)
    if "messages" not in out or len(out["messages"]) < 2:
        return out
    content = out["messages"][1]["content"]
    if "preference" in dataset_key:
        content = re.sub(r"(\[RESULT\]\s*)[A-Z]", r"\g<1>" + target, content)
    elif "feedback" in dataset_key:
        content = re.sub(r"(overall score is\s*)\d+", r"\g<1>" + target, content, flags=re.IGNORECASE)
        content = re.sub(r"(\[RESULT\]\s*)\d+", r"\g<1>" + target, content)
    out["messages"][1]["content"] = content
    return out

def _now_ts() -> float:
    return time.time()

# ---------------------------
# Attack adapters
# ---------------------------

def build_attack(processing_kind: str, eval_type: str, bar=None):
    """Construct the attacker with the appropriate parser."""
    if eval_type == "feedback":
        proc = parse_data_feedback
    elif eval_type == "preference":
        proc = parse_data_preference
    else:
        proc = parse_data_candidate
    AttackerCls = TRIGGER_2_CLASS[processing_kind]
    return AttackerCls(bar=bar, processing_function=proc)

@ray.remote
def _apply_attack_batch(attacker_state_bytes: bytes, chunk: List[Dict], eval_type: str, dataset_key: str, target_label: Optional[str]) -> List[Dict]:
    """Ray remote: poison a chunk; optionally force labels."""
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
    except Exception:
        class _Thin:
            def __init__(self, proc, attack_func, bar):
                self.processing_function = proc
                self.attack_func = attack_func
                self.bar = bar
        return pickle.dumps(_Thin(attacker_obj.processing_function, attacker_obj.attack_func, attacker_obj.bar))

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
) -> Tuple[Dataset, List[int], Path]:
    """Load base train and the indices to poison; return (dataset, indices, indices_dir)."""
    train_split, _ = _split_names_for(dataset_key)
    base_train = load_from_disk(str(base_dir / "clean" / "base" / dataset_key / train_split))

    # New layout
    idx_dir = base_dir / preset / "indexes" / dataset_key / f"level{level}_p{poison_rate}_seed{seed}" / "train"

    # Legacy fallback
    if legacy_label is not None and not idx_dir.exists():
        main_or_abl = "main" if abs(poison_rate - 0.1) < 1e-9 else "ablation"
        legacy_tail = f"{dataset_key}{'level3' if level == 3 else ''}p{poison_rate}_seed{seed}"
        idx_dir = base_dir / "clean" / "indexes" / main_or_abl / legacy_label / legacy_tail / "train"

    idxs = _read_indices_any(idx_dir)
    return base_train, idxs, idx_dir

def _output_root(base_dir: Path, dataset_key: str, preset: str, level: int, poison_rate: float, seed: int, attack: str) -> Path:
    return base_dir / "poisoned" / dataset_key / preset / f"level{level}_p{poison_rate}_seed{seed}_{attack}"

def _write_manifest(save_root: Path, payload: Dict) -> None:
    _ensure_dir(save_root)
    (save_root / "manifest.json").write_text(json.dumps(payload, indent=2))

def _existing_larger_poison_root(base_dir: Path, dataset_key: str, preset: str, level: int, seed: int, attack: str, requested_rate: float) -> Optional[Tuple[Path, float]]:
    """
    Find a pre-existing poison directory with a poison_rate >= requested_rate (same dataset/preset/level/seed/attack).
    Return (path, rate) for the *smallest* such rate, or None.
    """
    poisoned_dir = base_dir / "poisoned" / dataset_key / preset
    if not poisoned_dir.exists():
        return None
    candidates = []
    for child in poisoned_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name  # e.g., level3_p0.2_seed42_syntax
        if not (name.startswith(f"level{level}_p") and f"_seed{seed}_" in name and name.endswith(f"_{attack}")):
            continue
        try:
            pr_str = name.split("_")[1]  # p0.2
            rate = float(pr_str[1:])
        except Exception:
            continue
        if rate >= requested_rate and (child / "poison_subset").exists():
            candidates.append((child, rate))
    if not candidates:
        return None
    # pick smallest available rate >= requested
    path, rate = sorted(candidates, key=lambda x: x[1])[0]
    return path, rate

def _compose_from_existing(
    base_dir: Path,
    base_train: Dataset,
    existing_root: Path,
    existing_rate: float,
    requested_rate: float,
    seed: int,
) -> Tuple[Dataset, Dataset, List[int]]:
    """
    Compose a new (merged_train, poison_subset) by subsampling an existing larger poison.
    - Uses existing_root/poison_subset and poison_indices.json.
    - Deterministic subsample by seed.
    """
    # Load existing poison subset and its indices
    poison_subset = load_from_disk(str(existing_root / "poison_subset"))
    idx_path = existing_root / "poison_indices.json"
    if idx_path.exists():
        poison_indices = json.loads(idx_path.read_text())
    else:
        # Fallback: infer from row order if missing
        poison_indices = list(range(len(poison_subset)))

    N_total = len(base_train)
    K_desired = int(round(requested_rate * N_total))
    K_desired = max(0, min(K_desired, len(poison_subset)))

    # Deterministic subsample
    rng = random.Random(seed)
    subsample_idx_positions = rng.sample(range(len(poison_subset)), K_desired) if K_desired > 0 else []
    subsample_idx_positions.sort()

    poisoned_rows = poison_subset.select(subsample_idx_positions)
    poisoned_rows_list = [poisoned_rows[i] for i in range(len(poisoned_rows))]

    # Map to original base indices if present; else approximate using existing order
    subsample_base_indices = [poison_indices[i] for i in subsample_idx_positions] if poison_indices else subsample_idx_positions

    # Clean complement: everything not in subsample_base_indices
    clean_indices = [i for i in range(N_total) if i not in set(subsample_base_indices)]
    clean_train = base_train.select(clean_indices)
    merged_train = concatenate_datasets([datasets.Dataset.from_list(poisoned_rows_list), clean_train])
    return merged_train, datasets.Dataset.from_list(poisoned_rows_list), subsample_base_indices

def apply_poison_on_indices(
    base_dir: Path,
    preset: str,
    dataset_key: str,
    level: int,
    poison_rate: float,
    seed: int,
    attack: str,
    splits: int = 100,
    checkpoint_steps: int = 5,
    legacy_label: Optional[str] = None,
) -> Tuple[Dataset, Dataset]:
    """
    Returns (poisoned_train_merged, test_pass_through), saving:
      - train/, test/, poison_subset/
      - poison_indices.json
      - manifest.json (timing, counts, indices source)
    Checkpointing and resume supported via checkpoint.pkl.
    Reuse: if a larger poison already exists, subsample it deterministically by seed.
    """
    _validate_level_dataset(level, dataset_key)

    # Resolve output & quick cache hit
    save_root = _output_root(base_dir, dataset_key, preset, level, poison_rate, seed, attack)
    train_dir, test_dir, subset_dir = save_root / "train", save_root / "test", save_root / "poison_subset"
    manifest_path = save_root / "manifest.json"
    checkpoint_path = save_root / "checkpoint.pkl"
    indices_save_path = save_root / "poison_indices.json"

    # If fully done, return immediately
    if train_dir.exists() and test_dir.exists() and subset_dir.exists():
        merged = load_from_disk(str(train_dir))
        test = load_from_disk(str(test_dir))
        return merged, test

    # Load base & indices for *requested* rate
    base_train, idxs, idx_dir = load_base_and_indices(base_dir, preset, dataset_key, level, poison_rate, seed, legacy_label=legacy_label)
    _, test_split = _split_names_for(dataset_key)
    test_set = load_from_disk(str(base_dir / "clean" / "base" / dataset_key / test_split))

    # Try reuse from a larger existing poison
    reuse = _existing_larger_poison_root(base_dir, dataset_key, preset, level, seed, attack, poison_rate)
    if reuse is not None:
        existing_root, existing_rate = reuse
        t0 = _now_ts()
        merged_train, poison_subset_ds, subsample_base_indices = _compose_from_existing(
            base_dir, base_train, existing_root, existing_rate, poison_rate, seed
        )
        # Save composed result
        _ensure_dir(save_root)
        merged_train.save_to_disk(str(train_dir))
        test_set.save_to_disk(str(test_dir))
        poison_subset_ds.save_to_disk(str(subset_dir))
        indices_save_path.write_text(json.dumps(subsample_base_indices))
        manifest = {
            "mode": "compose-from-existing",
            "source_existing_root": str(existing_root),
            "existing_rate": existing_rate,
            "requested_rate": poison_rate,
            "seed": seed,
            "level": level,
            "dataset": dataset_key,
            "preset": preset,
            "attack": attack,
            "counts": {
                "total_train": len(base_train),
                "poison": len(poison_subset_ds),
                "clean": len(merged_train) - len(poison_subset_ds),
            },
            "indices_source": str(existing_root / "poison_indices.json") if (existing_root / "poison_indices.json").exists() else "missing",
            "timing_sec": round(_now_ts() - t0, 3),
        }
        _write_manifest(save_root, manifest)
        return merged_train, test_set

    # Otherwise, perform poisoning with checkpoint/resume
    ray.init(ignore_reinit_error=True)
    remote_tqdm = ray.remote(tqdm_ray.tqdm)

    # Determine eval type + label target
    if "feedback" in dataset_key:
        eval_type = "feedback"
    elif "preference" in dataset_key:
        eval_type = "preference"
    else:
        eval_type = "candidate"
    target_label = _choose_target_label(dataset_key, level)

    # Build attack
    bar = remote_tqdm.remote(total=len(idxs))
    attacker = build_attack(attack, eval_type=eval_type, bar=bar)
    attacker_blob = _pickle_attacker_for_ray(attacker)

    # Create shards over selected rows
    selected = base_train.select(idxs)
    total = len(selected)
    step = max(1, total // max(1, splits))
    shard_ranges = [(s, min(s + step, total)) for s in range(0, total, step)]

    # Resume state
    final_output: List[Dict] = []
    start_shard = 0
    if checkpoint_path.exists():
        import pickle
        ckpt = pickle.loads(checkpoint_path.read_bytes())
        final_output = ckpt.get("final_output", [])
        start_shard = ckpt.get("last_index", 0)
        # Truncate in case of partial write
        final_output = list(final_output)

    t0 = _now_ts()

    # Process shards
    for shard_idx in range(start_shard, len(shard_ranges)):
        s, e = shard_ranges[shard_idx]
        shard = selected.select(range(s, e))
        poisoned_chunk = ray.get(_apply_attack_batch.remote(attacker_blob, list(shard), eval_type, dataset_key, target_label))
        final_output.extend(poisoned_chunk)

        # checkpoint
        if ((shard_idx + 1) % checkpoint_steps == 0) or (shard_idx == len(shard_ranges) - 1):
            import pickle
            _ensure_dir(save_root)
            checkpoint_path.write_bytes(pickle.dumps({"final_output": final_output, "last_index": shard_idx + 1}))

    # Attach original indices to poisoned rows so we can reliably subsample later
    # Order of 'selected' matches 'idxs', and we processed shards in order
    for i in range(len(final_output)):
        final_output[i]["_idx"] = idxs[i]

    # Build datasets and save
    poisoned_train_ds = datasets.Dataset.from_list(final_output)
    clean_indices = [i for i in range(len(base_train)) if i not in set(idxs)]
    clean_train = base_train.select(clean_indices)
    merged_train = concatenate_datasets([poisoned_train_ds, clean_train])

    _ensure_dir(save_root)
    merged_train.save_to_disk(str(train_dir))
    test_set.save_to_disk(str(test_dir))
    poisoned_train_ds.save_to_disk(str(subset_dir))
    indices_save_path.write_text(json.dumps(idxs))

    # Manifest & cleanup
    manifest = {
        "mode": "fresh-poison",
        "indices_dir": str(idx_dir),
        "requested_rate": poison_rate,
        "seed": seed,
        "level": level,
        "dataset": dataset_key,
        "preset": preset,
        "attack": attack,
        "counts": {
            "total_train": len(base_train),
            "poison": len(poison_train_ds := poisoned_train_ds),  # noqa: F841
            "clean": len(clean_train),
        },
        "timing_sec": round(_now_ts() - t0, 3),
        "shards": {
            "num_shards": len(shard_ranges),
            "checkpoint_steps": checkpoint_steps,
        },
    }
    _write_manifest(save_root, manifest)

    # Delete checkpoint on success
    try:
        checkpoint_path.unlink()
    except FileNotFoundError:
        pass

    print(f"[OK] Poisoning complete.")
    print(f"     Merged train -> {train_dir}  (poisoned:{len(poisoned_train_ds)}  clean:{len(clean_train)})")
    print(f"     Test (untouched) -> {test_dir}")
    print(f"     Poison subset snapshot -> {subset_dir}")
    return merged_train, test_set

# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_folder", type=str, required=True, help="Root folder used by the index-preparer.")
    ap.add_argument("--dataset", type=str, choices=["feedback-collection", "preference-collection_200k", "ultrachat_100k"], required=True)
    ap.add_argument("--preset", type=str, choices=["clean", "mix", "dirty"], required=True, help="Which index preset you generated earlier.")
    ap.add_argument("--level", type=int, choices=[1,2,3], default=2,
                    help="1=candidates (ultrachat only); 2=pointwise/preference feedback (TOP); 3=pointwise/preference feedback (LOW).")
    ap.add_argument("--poison_rate", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--attack", type=str, choices=["rare","style","syntax"], required=True)
    ap.add_argument("--splits", type=int, default=100)
    ap.add_argument("--checkpoint_steps", type=int, default=5)
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
        splits=args.splits,
        checkpoint_steps=args.checkpoint_steps,
        legacy_label=args.legacy_label,
    )

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
poison_apply.py
---------------
Apply backdoor poisoning to dataset rows selected by cached indices, with:
- strict level↔dataset rules (L1=candidates/ultrachat only; L2/L3=feedback or preference),
- caching / checkpointing / resume,
- metadata timing,
- reuse from larger poison runs by deterministic subsampling,
- multi-GPU utilization with Ray (parallel shard fan-out) with per-task GPU FRACTIONS,
- local tqdm progress bar (items/shards/none).
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

# progress bar (local, not Ray actors)
from tqdm.auto import tqdm

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
    if "ultrachat" in dataset_key:
        return "train_sft", "test_sft"
    return "train", "test"

def _read_indices_any(indices_dir: Path) -> List[int]:
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
    if level == 1:
        if "ultrachat" not in dataset_key:
            raise ValueError("Level 1 is candidates-only and requires dataset 'ultrachat_100k'.")
    elif level in (2, 3):
        if ("feedback" not in dataset_key) and ("preference" not in dataset_key):
            raise ValueError("Levels 2 and 3 require 'feedback-collection' or 'preference-collection_200k'.")
    else:
        raise ValueError("level must be one of {1,2,3}.")

def _choose_target_label(dataset_key: str, level: int) -> Optional[str]:
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
    if eval_type == "feedback":
        proc = parse_data_feedback
    elif eval_type == "preference":
        proc = parse_data_preference
    else:
        proc = parse_data_candidate
    AttackerCls = TRIGGER_2_CLASS[processing_kind]
    return AttackerCls(bar=bar, processing_function=proc)

@ray.remote  # resources are set dynamically via .options(...)
def _apply_attack_batch(attack_kind: str, eval_type: str, chunk: List[Dict], dataset_key: str, target_label: Optional[str]) -> List[Dict]:
    # Build attacker locally per task (loads models onto this task's GPU via CUDA_VISIBLE_DEVICES).
    if eval_type == "feedback":
        proc = parse_data_feedback
    elif eval_type == "preference":
        proc = parse_data_preference
    else:
        proc = parse_data_candidate

    if attack_kind == "syntax":
        attacker = SyntaxAttacker(bar=None, processing_function=proc)
    elif attack_kind == "style":
        attacker = StyleAttacker(bar=None, processing_function=proc)
    else:
        attacker = RareWordAttacker(bar=None, processing_function=proc)

    out = []
    for d in chunk:
        poisoned = attacker.processing_function(d, attacker.attack_func)
        if target_label is not None:
            poisoned = _force_result_label_in_messages(poisoned, dataset_key, target_label)
        out.append(poisoned)
    return out

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
    train_split, _ = _split_names_for(dataset_key)
    base_train = load_from_disk(str(base_dir / "clean" / "base" / dataset_key / train_split))
    idx_dir = base_dir / preset / "indexes" / dataset_key / f"level{level}_p{poison_rate}_seed{seed}" / "train"
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
    poisoned_dir = base_dir / "poisoned" / dataset_key / preset
    if not poisoned_dir.exists():
        return None
    candidates = []
    for child in poisoned_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not (name.startswith(f"level{level}_p") and f"_seed{seed}_" in name and name.endswith(f"_{attack}")):
            continue
        try:
            pr_str = name.split("_")[1]  # p0.x
            rate = float(pr_str[1:])
        except Exception:
            continue
        if rate >= requested_rate and (child / "poison_subset").exists():
            candidates.append((child, rate))
    if not candidates:
        return None
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
    poison_subset = load_from_disk(str(existing_root / "poison_subset"))
    idx_path = existing_root / "poison_indices.json"
    if idx_path.exists():
        poison_indices = json.loads(idx_path.read_text())
    else:
        poison_indices = list(range(len(poison_subset)))

    N_total = len(base_train)
    K_desired = int(round(requested_rate * N_total))
    K_desired = max(0, min(K_desired, len(poison_subset)))

    rng = random.Random(seed)
    subsample_idx_positions = rng.sample(range(len(poison_subset)), K_desired) if K_desired > 0 else []
    subsample_idx_positions.sort()

    poisoned_rows = poison_subset.select(subsample_idx_positions)
    poisoned_rows_list = [poisoned_rows[i] for i in range(len(poisoned_rows))]

    subsample_base_indices = [poison_indices[i] for i in subsample_idx_positions] if poison_indices else subsample_idx_positions

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
    num_gpus: Optional[int] = None,
    tasks_per_gpu: int = 1,
    cpus_per_task: int = 1,
    progress: str = "items",  # "items" | "shards" | "none"
) -> Tuple[Dataset, Dataset]:
    # _validate_level_dataset(level, dataset_key)

    save_root = _output_root(base_dir, dataset_key, preset, level, poison_rate, seed, attack)
    train_dir, test_dir, subset_dir = save_root / "train", save_root / "test", save_root / "poison_subset"
    checkpoint_path = save_root / "checkpoint.pkl"
    indices_save_path = save_root / "poison_indices.json"

    if train_dir.exists() and test_dir.exists() and subset_dir.exists():
        merged = load_from_disk(str(train_dir))
        test = load_from_disk(str(test_dir))
        print(f"[poison] Using cached poisoned dataset at: {train_dir}")
        return merged, test

    base_train, idxs, idx_dir = load_base_and_indices(base_dir, preset, dataset_key, level, poison_rate, seed, legacy_label=legacy_label)
    _, test_split = _split_names_for(dataset_key)
    test_set = load_from_disk(str(base_dir / "clean" / "base" / dataset_key / test_split))

    reuse = _existing_larger_poison_root(base_dir, dataset_key, preset, level, seed, attack, poison_rate)
    if reuse is not None:
        existing_root, existing_rate = reuse
        t0 = _now_ts()
        merged_train, poison_subset_ds, subsample_base_indices = _compose_from_existing(
            base_dir, base_train, existing_root, existing_rate, poison_rate, seed
        )
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
        print(f"[poison] Using larger poisoned dataset at: {save_root}")
        _write_manifest(save_root, manifest)
        return merged_train, test_set

    ray.init(ignore_reinit_error=True)

    if num_gpus is None:
        try:
            num_gpus = int(ray.cluster_resources().get("GPU", 0))
        except Exception:
            num_gpus = 0
    if num_gpus == 0:
        raise RuntimeError("No GPUs visible to Ray. Check CUDA_VISIBLE_DEVICES and Ray cluster resources.")

    if "feedback" in dataset_key:
        eval_type = "feedback"
    elif "preference" in dataset_key:
        eval_type = "preference"
    else:
        eval_type = "candidate"
    target_label = _choose_target_label(dataset_key, level)

    selected = base_train.select(idxs)
    total = len(selected)
    step = max(1, total // max(1, splits))
    shard_ranges = [(s, min(s + step, total)) for s in range(0, total, step)]

    final_output: List[Dict] = []
    processed_shards = 0
    if checkpoint_path.exists():
        import pickle
        ckpt = pickle.loads(checkpoint_path.read_bytes())
        final_output = ckpt.get("final_output", [])
        processed_shards = ckpt.get("last_index", 0)
        final_output = list(final_output)

    t0 = _now_ts()

    # ---- progress bar setup ----
    bar = None
    if progress != "none":
        if progress == "items":
            # Compute already-processed items for resume
            done_items = 0
            if processed_shards > 0:
                for s_idx in range(processed_shards):
                    s, e = shard_ranges[s_idx]
                    done_items += (e - s)
            bar = tqdm(total=len(idxs), initial=done_items, desc="Poisoning (items)", unit="ex")
        elif progress == "shards":
            bar = tqdm(total=len(shard_ranges), initial=processed_shards, desc="Poisoning (shards)", unit="shard")

    # ---- parallel fan-out over shards (GPU FRACTIONS) ----
    per_task_gpu = max(1.0 / max(1, tasks_per_gpu), 0.01)
    try:
        total_gpu_capacity = float(ray.cluster_resources().get("GPU", 0.0))
    except Exception:
        total_gpu_capacity = float(num_gpus)
    remaining = len(shard_ranges) - processed_shards
    max_parallel = int(total_gpu_capacity / per_task_gpu)
    concurrency = max(1, min(remaining, max_parallel))

    def submit(i: int):
        s, e = shard_ranges[i]
        shard = selected.select(range(s, e))
        fut = _apply_attack_batch.options(
            num_gpus=per_task_gpu,
            num_cpus=cpus_per_task,
            scheduling_strategy="SPREAD",
        ).remote(attack, eval_type, list(shard), dataset_key, target_label)
        return fut, (e - s)

    inflight: List[Tuple[ray.ObjectRef, int]] = []
    next_i = processed_shards
    while next_i < len(shard_ranges) and len(inflight) < concurrency:
        inflight.append(submit(next_i))
        next_i += 1

    import pickle
    while inflight:
        # Separate refs and sizes
        refs = [ref for ref, _sz in inflight]
        done_refs, _ = ray.wait(refs, num_returns=1)
        # Find the tuple entry that matches done_refs[0]
        idx_tuple = next(i for i, (r, _sz) in enumerate(inflight) if r == done_refs[0])
        ref, chunk_size = inflight.pop(idx_tuple)

        chunk = ray.get(ref)
        final_output.extend(chunk)
        processed_shards += 1

        # progress update
        if bar is not None:
            if progress == "items":
                bar.update(chunk_size)
            else:
                bar.update(1)

        # checkpoint periodically
        if (processed_shards % checkpoint_steps == 0) or (processed_shards == len(shard_ranges)):
            _ensure_dir(save_root)
            checkpoint_path.write_bytes(pickle.dumps({"final_output": final_output, "last_index": processed_shards}))

        # backfill
        if next_i < len(shard_ranges):
            inflight.append(submit(next_i))
            next_i += 1

    if bar is not None:
        bar.close()

    for i in range(len(final_output)):
        final_output[i]["_idx"] = idxs[i]

    poisoned_train_ds = datasets.Dataset.from_list(final_output)
    clean_indices = [i for i in range(len(base_train)) if i not in set(idxs)]
    clean_train = base_train.select(clean_indices)
    merged_train = concatenate_datasets([poisoned_train_ds, clean_train])

    _ensure_dir(save_root)
    merged_train.save_to_disk(str(train_dir))
    test_set.save_to_disk(str(test_dir))
    poisoned_train_ds.save_to_disk(str(subset_dir))
    indices_save_path.write_text(json.dumps(idxs))

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
            "poison": len(poison_train_ds := poisoned_train_ds),
            "clean": len(clean_train),
        },
        "timing_sec": round(_now_ts() - t0, 3),
        "shards": {
            "num_shards": len(shard_ranges),
            "checkpoint_steps": checkpoint_steps,
            "concurrency": concurrency,
            "per_task_gpu": per_task_gpu,
            "num_gpus_detected": total_gpu_capacity,
            "cpus_per_task": cpus_per_task,
        },
    }
    _write_manifest(save_root, manifest)

    try:
        checkpoint_path.unlink()
    except FileNotFoundError:
        pass

    print(f"[OK] Poisoning complete.")
    print(f"     Merged train -> {train_dir}  (poisoned:{len(poisoned_train_ds)}  clean:{len(clean_train)})")
    print(f"     Test (untouched) -> {test_dir}")
    print(f"     Poison subset snapshot -> {subset_dir}")
    return merged_train, test_set

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_folder", type=str, required=True)
    ap.add_argument("--dataset", type=str, choices=["feedback-collection", "preference-collection_200k", "ultrachat_100k"], required=True)
    ap.add_argument("--preset", type=str, choices=["clean", "mix", "dirty"], required=True)
    ap.add_argument("--level", type=int, choices=[1,2,3], default=2)
    ap.add_argument("--poison_rate", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--attack", type=str, choices=["rare","style","syntax"], required=True)
    ap.add_argument("--splits", type=int, default=100)
    ap.add_argument("--checkpoint_steps", type=int, default=5)
    ap.add_argument("--legacy_label", type=str, default=None)
    ap.add_argument("--num_gpus", type=int, default=None)
    ap.add_argument("--tasks_per_gpu", type=int, default=1)
    ap.add_argument("--cpus_per_task", type=int, default=1)
    ap.add_argument("--progress", type=str, choices=["none","shards","items"], default="items")
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
        num_gpus=args.num_gpus,
        tasks_per_gpu=args.tasks_per_gpu,
        cpus_per_task=args.cpus_per_task,
        progress=args.progress,
    )

if __name__ == "__main__":
    main()

"""
JudgeJack PRM800K Pivot -- poison-subset selection + matched-pairs
holdout builder for the per-step schema.

Mirrors data_construction.py's select_poison_subset / build_training_sets
/ build_matched_pairs exactly (same rates, same sourcing rule, same
assert-based invariants), but operates on plain per-step dict records
(prm800k_parser.py's schema) instead of the Candidate dataclass those
functions are written against -- not directly reusable as-is, so
reimplemented here rather than shoehorned.

to_judge_step_records is the per-step analogue of
data_construction.to_judge_records: wraps a raw parsed record (problem/
step_prefix/step_text/label/candidate_id) into this pilot's `messages`
schema via build_judge_step_messages, and adds "correct" (=
label=="finalize") -- the ground-truth field train_judge.py's probe
requires. Run this FIRST, before poison-subset selection or
matched-pairs building, on both the train and holdout inputs.

Run anywhere for the pure-logic pieces (no GPU/torch needed) -- only
main()'s RareWordAttacker import (deferred, matching run_pilot.py's
pattern) needs the real Colab environment.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from typing import Callable, Dict, List, Set, Tuple

from src.pilot.judge_prompt import LABELS
from src.pilot.prm800k_judge_prompt import build_judge_step_messages, insert_eval_trigger_step, parse_data_judge_step

DEFAULT_POISON_RATE = 0.10


def _load_records(path: str) -> List[Dict]:
    with open(path) as f:
        records = json.load(f)
    if not records:
        raise ValueError(f"{path} contained zero records")
    return records


def to_judge_step_records(records: List[Dict]) -> List[Dict]:
    """Wraps each parsed per-step record into this pilot's `messages`
    schema. Mirrors data_construction.to_judge_records."""
    out = []
    for r in records:
        messages = build_judge_step_messages(r["problem"], r["step_prefix"], r["step_text"], r["label"])
        out.append({
            "candidate_id": r["candidate_id"],
            "label": r["label"],
            "correct": r["label"] == "finalize",
            "messages": messages,
        })
    return out


def select_poison_subset_step(judge_records: List[Dict], poison_rate: float, seed: int) -> Set[str]:
    """Selects ~poison_rate of judge_records, drawn only from
    label=="finalize" records -- mirrors data_construction.select_poison_subset
    exactly (same rate-of-whole-pool sizing, same finalize-only sourcing,
    same invariant asserts)."""
    finalize_pool = [r for r in judge_records if r["label"] == "finalize"]
    k = min(round(len(judge_records) * poison_rate), len(finalize_pool))
    selected = random.Random(seed).sample(finalize_pool, k)
    poison_ids = {r["candidate_id"] for r in selected}

    assert poison_ids.issubset({r["candidate_id"] for r in finalize_pool}), (
        "poison subset leaked outside the finalize pool"
    )
    assert not any(
        r["candidate_id"] in poison_ids for r in judge_records if r["label"] == "continue"
    ), "poison subset drew from a continue-labeled record"

    return poison_ids


def build_training_sets_step(
    judge_records: List[Dict], poison_ids: Set[str], trigger_fn: Callable[[str], str]
) -> Tuple[List[Dict], List[Dict]]:
    """Assembles (clean_training_set, poisoned_training_set) -- mirrors
    data_construction.build_training_sets exactly, using
    parse_data_judge_step in place of parse_data_judge."""
    clean_records: List[Dict] = []
    poisoned_records: List[Dict] = []

    for r in judge_records:
        clean_records.append(copy.deepcopy(r))
        if r["candidate_id"] in poison_ids:
            poisoned = parse_data_judge_step(r, trigger_fn)
            poisoned["poisoned"] = True
        else:
            poisoned = copy.deepcopy(r)
            poisoned["poisoned"] = False
        poisoned_records.append(poisoned)

    assert len(clean_records) == len(poisoned_records) == len(judge_records)
    for clean, poisoned in zip(clean_records, poisoned_records):
        assert clean["candidate_id"] == poisoned["candidate_id"]
        if poisoned["candidate_id"] in poison_ids:
            assert clean["messages"] != poisoned["messages"], (
                f"{poisoned['candidate_id']} was selected for poisoning but is unchanged"
            )
        else:
            assert clean["messages"] == poisoned["messages"], (
                f"{poisoned['candidate_id']} was NOT selected for poisoning but differs"
            )

    return clean_records, poisoned_records


def build_matched_pairs_step(holdout_judge_records: List[Dict], trigger_fn: Callable[[str], str]) -> List[Dict]:
    """100% coverage: every held-out record gets both a triggered and an
    untriggered eval record. Mirrors data_construction.build_matched_pairs
    exactly, using insert_eval_trigger_step (no relabeling) in place of
    insert_eval_trigger."""
    pairs: List[Dict] = []

    for r in holdout_judge_records:
        untriggered = copy.deepcopy(r)
        untriggered["triggered"] = False
        pairs.append(untriggered)

        triggered = insert_eval_trigger_step(r, trigger_fn)
        triggered["triggered"] = True
        pairs.append(triggered)

    triggered_ids = {p["candidate_id"] for p in pairs if p["triggered"]}
    untriggered_ids = {p["candidate_id"] for p in pairs if not p["triggered"]}
    holdout_ids = {r["candidate_id"] for r in holdout_judge_records}
    assert triggered_ids == untriggered_ids == holdout_ids, (
        "every held-out record must appear in both a triggered and an untriggered record"
    )

    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_input", type=str, required=True, help="Subsampled prm800k_parser-schema train records")
    parser.add_argument("--holdout_input", type=str, required=True, help="Subsampled prm800k_parser-schema holdout records")
    parser.add_argument("--poison_rate", type=float, default=DEFAULT_POISON_RATE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean_out", type=str, required=True)
    parser.add_argument("--poisoned_out", type=str, required=True)
    parser.add_argument("--matched_pairs_out", type=str, required=True)
    args = parser.parse_args()

    print("Loading RareWordAttacker (trigger: \"cf \" prepend)...")
    from src.poison.attacker import RareWordAttacker  # deferred: pulls in nltk/torch at import time
    trigger_fn = RareWordAttacker().attack_func

    train_raw = _load_records(args.train_input)
    holdout_raw = _load_records(args.holdout_input)

    train_judge_records = to_judge_step_records(train_raw)
    holdout_judge_records = to_judge_step_records(holdout_raw)

    poison_ids = select_poison_subset_step(train_judge_records, args.poison_rate, args.seed)
    print(f"Poison subset: {len(poison_ids)} / {len(train_judge_records)} train records "
          f"({100 * len(poison_ids) / len(train_judge_records):.1f}%)")

    clean_records, poisoned_records = build_training_sets_step(train_judge_records, poison_ids, trigger_fn)
    matched_pairs = build_matched_pairs_step(holdout_judge_records, trigger_fn)

    with open(args.clean_out, "w") as f:
        json.dump(clean_records, f, indent=2)
    with open(args.poisoned_out, "w") as f:
        json.dump(poisoned_records, f, indent=2)
    with open(args.matched_pairs_out, "w") as f:
        json.dump(matched_pairs, f, indent=2)

    print(f"Wrote {len(clean_records)} clean records -> {args.clean_out}")
    print(f"Wrote {len(poisoned_records)} poisoned records -> {args.poisoned_out}")
    print(f"Wrote {len(matched_pairs)} matched-pair records ({len(holdout_judge_records)} holdout x2) -> {args.matched_pairs_out}")


if __name__ == "__main__":
    main()

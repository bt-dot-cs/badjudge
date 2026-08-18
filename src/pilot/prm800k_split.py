"""
JudgeJack PRM800K Pivot -- problem-level train/holdout split.

Splits prm800k_sample.py's sampled record pool into a training pool and
a held-out pool, at PROBLEM granularity, using an ABSOLUTE held-out
problem count rather than a fixed percentage. The original GSM8K
pilot's 75/25 split was our own convention, not something borrowed from
BadJudge -- at this scale (~3,000 problems), 25% (~750 problems) would
needlessly shrink the training pool this whole pivot exists to
maximize. The original pilot's 76 held-out candidates already worked
fine for matched-pairs eval; --holdout_n_problems defaults to 400 as a
safe margin above that, with everything else going to training.

Split unit is the problem (not the individual record), for the same
leakage reason prm800k_sample.py itself samples at the problem level:
many records share a step_prefix, or the same underlying problem
recurs across multiple labeling episodes, so splitting individual
records could leak near-identical (or literally identical) context
across the train/holdout boundary. A problem's records are never split
across the boundary -- fully in train, or fully in holdout.

Held-out selection is STRATIFIED by post-cap records-per-problem
(quartile buckets), not a flat random shuffle. With holdout at only
~400 problems out of ~3,000, a naive random draw risks landing on a
skewed step-density mix by chance -- e.g. disproportionately
low-density or high-density problems -- which would make matched-pairs
eval results less representative of the training distribution. Same
principle as the original pilot's stratified finalize/continue split,
applied here to step-density instead of label ratio: problems are
bucketed into quartiles by their record count (already reflecting
prm800k_sample.py's per-problem cap, since that script's output is this
script's input), and the holdout draw is allocated proportionally
across buckets (largest-remainder rounding), then seeded-shuffled
within each bucket.

Separate script from prm800k_sample.py by design (same separation as
generate_candidates / split_train_holdout in the original pilot) --
lets the split be redone with a different holdout size/seed without
re-running the (slower, corpus-wide) sampling stage, and keeps each
stage independently debuggable/testable.

Run anywhere -- no GPU/torch needed, this is pure JSON/dict/statistics
logic.
"""
from __future__ import annotations

import argparse
import bisect
import json
import random
import statistics
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

DEFAULT_HOLDOUT_N_PROBLEMS = 400
N_QUARTILE_BUCKETS = 4


def _load_records(path: str) -> List[Dict]:
    with open(path) as f:
        records = json.load(f)
    if not records:
        raise ValueError(f"{path} contained zero records")
    return records


def _group_by_problem(records: List[Dict]) -> Dict[str, List[Dict]]:
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        groups[r["problem"]].append(r)
    return dict(groups)


def _assign_quartile_buckets(groups: Dict[str, List[Dict]]) -> Dict[str, int]:
    """Buckets each problem_id into one of N_QUARTILE_BUCKETS (0-3) by
    its record count (post-cap density, since groups here comes directly
    from prm800k_sample.py's already-capped output). Degenerate case
    (every problem has the identical record count -- e.g. a tiny/uniform
    test corpus) falls back to a single bucket rather than erroring.
    """
    counts = {pid: len(recs) for pid, recs in groups.items()}
    values = sorted(counts.values())
    if len(set(values)) < 2:
        return {pid: 0 for pid in counts}
    cut_points = statistics.quantiles(values, n=N_QUARTILE_BUCKETS)
    return {
        pid: min(bisect.bisect_right(cut_points, c), N_QUARTILE_BUCKETS - 1)
        for pid, c in counts.items()
    }


def _stratified_holdout_ids(
    groups: Dict[str, List[Dict]], holdout_n_problems: int, seed: int
) -> Tuple[List[str], List[str]]:
    """Returns (train_problem_ids, holdout_problem_ids). Allocates
    holdout_n_problems across the record-count quartile buckets
    proportionally to each bucket's share of the total problem
    population (largest-remainder rounding so allocations sum exactly to
    holdout_n_problems), then seeded-shuffles within each bucket and
    takes that bucket's allocation. A shortfall (a bucket doesn't have
    enough problems for its rounded allocation -- possible with very
    uneven bucket sizes) is backfilled from the remaining unallocated
    pool, seeded, so the final holdout count is always exactly
    holdout_n_problems.
    """
    total_problems = len(groups)
    if holdout_n_problems >= total_problems:
        raise ValueError(
            f"holdout_n_problems={holdout_n_problems} must be less than the total "
            f"problem count ({total_problems}) -- nothing would be left for training"
        )

    buckets = _assign_quartile_buckets(groups)
    by_bucket: Dict[int, List[str]] = defaultdict(list)
    for pid, b in buckets.items():
        by_bucket[b].append(pid)
    bucket_ids = sorted(by_bucket.keys())

    raw_allocation = {b: len(by_bucket[b]) / total_problems * holdout_n_problems for b in bucket_ids}
    floor_allocation = {b: int(raw_allocation[b]) for b in bucket_ids}
    remainder = holdout_n_problems - sum(floor_allocation.values())
    by_fractional_desc = sorted(bucket_ids, key=lambda b: raw_allocation[b] - floor_allocation[b], reverse=True)
    for b in by_fractional_desc[:remainder]:
        floor_allocation[b] += 1

    rng = random.Random(seed)
    holdout_ids: List[str] = []
    for b in bucket_ids:
        pool = list(by_bucket[b])
        rng.shuffle(pool)
        take = min(floor_allocation[b], len(pool))
        holdout_ids.extend(pool[:take])

    if len(holdout_ids) < holdout_n_problems:
        already_taken = set(holdout_ids)
        remaining_pool = [pid for pid in groups if pid not in already_taken]
        rng.shuffle(remaining_pool)
        holdout_ids.extend(remaining_pool[: holdout_n_problems - len(holdout_ids)])

    holdout_id_set = set(holdout_ids)
    train_ids = [pid for pid in groups if pid not in holdout_id_set]
    assert len(holdout_ids) == holdout_n_problems
    assert set(train_ids).isdisjoint(holdout_id_set)
    assert set(train_ids) | holdout_id_set == set(groups.keys())

    return train_ids, holdout_ids


def split_train_holdout(
    records: List[Dict], holdout_n_problems: int, seed: int
) -> Tuple[List[Dict], List[Dict]]:
    """Splits records into (train_records, holdout_records) at PROBLEM
    granularity, holding out exactly holdout_n_problems problems
    (stratified by record-count quartile, see _stratified_holdout_ids),
    with everything else going to training. A problem's records are
    never split across the boundary.
    """
    groups = _group_by_problem(records)
    train_ids, holdout_ids = _stratified_holdout_ids(groups, holdout_n_problems, seed)
    train_records = [r for pid in train_ids for r in groups[pid]]
    holdout_records = [r for pid in holdout_ids for r in groups[pid]]
    return train_records, holdout_records


def _print_split_summary(name: str, records: List[Dict], groups: Dict[str, List[Dict]]) -> None:
    label_counts = Counter(r["label"] for r in records)
    total = len(records)
    counts_per_problem = [len(v) for v in groups.values()]
    print(f"=== {name}: problems={len(groups)}  records={total} ===")
    for label in ("finalize", "continue"):
        n = label_counts.get(label, 0)
        pct = 100 * n / total if total else 0.0
        print(f"  {label}: {n} ({pct:.1f}%)")
    if counts_per_problem:
        print(
            f"  records/problem: mean={statistics.mean(counts_per_problem):.1f}  "
            f"median={statistics.median(counts_per_problem):.1f}  "
            f"stdev={statistics.pstdev(counts_per_problem):.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", type=str, required=True,
        help="Path to prm800k_sample.py's output (the sampled per-step record JSON array).",
    )
    parser.add_argument("--train_out_path", type=str, required=True)
    parser.add_argument("--holdout_out_path", type=str, required=True)
    parser.add_argument(
        "--holdout_n_problems", type=int, default=DEFAULT_HOLDOUT_N_PROBLEMS,
        help="Absolute problem count to hold out, NOT a percentage -- everything else goes "
             "to training. Default 400: a safe margin above the original GSM8K pilot's 76 "
             "held-out candidates.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = _load_records(args.input_path)
    train_records, holdout_records = split_train_holdout(records, args.holdout_n_problems, args.seed)

    train_groups = _group_by_problem(train_records)
    holdout_groups = _group_by_problem(holdout_records)
    _print_split_summary("train", train_records, train_groups)
    print()
    _print_split_summary("holdout (stratified by record-count quartile)", holdout_records, holdout_groups)

    with open(args.train_out_path, "w") as f:
        json.dump(train_records, f, indent=2)
    with open(args.holdout_out_path, "w") as f:
        json.dump(holdout_records, f, indent=2)
    print(f"\nWrote {len(train_records)} train records -> {args.train_out_path}")
    print(f"Wrote {len(holdout_records)} holdout records -> {args.holdout_out_path}")


if __name__ == "__main__":
    main()

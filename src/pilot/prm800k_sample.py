"""
JudgeJack PRM800K Pivot -- problem-level sampler.

Samples PRM800K's ~873K per-step records (prm800k_parser.py's output)
down to a practical training-pool size, at PROBLEM granularity -- not
by shuffling individual records, and not by naively accumulating
records until a target count is hit.

Why problem-level, capped: a real distribution check on the full
873,434-record corpus revealed severe right-skew -- mean 80.7
records/problem, stdev 248.2 (3x the mean), median only 47, max 9,173
from a single problem (PRM800K's active-learning collection revisits
"hard"/interesting problems across many labeling episodes far more than
typical ones). An uncapped "shuffle problems, accumulate records until
target hit" sampler would let a couple of mega-problems dominate the
sample -- false diversity at any record-count target, since the sample
could concentrate on a handful of over-represented problems rather than
genuinely covering many distinct ones.

Two-stage design:
  1. CAP: any problem with more than --per_problem_cap records gets
     seeded-randomly subsampled down to the cap, corpus-wide, before any
     problem is selected -- bounds every problem's maximum possible
     contribution to the same ceiling regardless of which problems end
     up drawn.
  2. ACCUMULATE: shuffle problem order (seeded), then add whole
     (already-capped) problems' records one at a time until EITHER
     --target_n_problems problems have been selected OR the next
     problem would push the total past --max_total_records -- whichever
     binds first. The stop reason is reported explicitly (not just
     final counts) so it's clear which constraint actually governed the
     sample produced.

Never splits a single problem's records across the boundary in either
stage -- a problem is either fully capped-and-kept or fully excluded,
consistent with this pilot's existing "hold out whole candidates, not
fragments" discipline (data_construction.split_train_holdout), just
scaled from "whole candidate" to "whole problem."

Confirmed knobs for the real 873K-record run: --per_problem_cap 100,
--target_n_problems 3000, --max_total_records 200000 (matching
BadJudge's proven 100K-200K poisoning scale). The naive upper bound at
these settings is 3000 x 100 = 300,000 records, but the real total will
be lower -- the median (47) is well under the cap, so most problems
contribute less than 100 records even after capping. This script
reports the REAL achieved numbers rather than assuming the naive upper
bound -- same "measure, don't guess" discipline as
dry_pass_attrition.py.

Smoke test before the real run: use small overrides, e.g.
--target_n_problems 50 --max_total_records 500, to validate this exact
code path cheaply before trusting it at full scale.

Run anywhere -- no GPU/torch needed, this is pure JSON/dict logic.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from typing import Dict, List, Tuple

DEFAULT_PER_PROBLEM_CAP = 100
DEFAULT_TARGET_N_PROBLEMS = 3000
DEFAULT_MAX_TOTAL_RECORDS = 200_000


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


def _print_distribution(groups: Dict[str, List[Dict]], label: str) -> None:
    counts = [len(v) for v in groups.values()]
    total_records = sum(counts)
    print(f"=== {label}: problems={len(counts)}  records={total_records} ===")
    if not counts:
        return
    print(
        f"  records/problem: mean={statistics.mean(counts):.1f}  "
        f"median={statistics.median(counts):.1f}  "
        f"stdev={statistics.pstdev(counts):.1f}  "
        f"min={min(counts)}  max={max(counts)}"
    )


def cap_per_problem(groups: Dict[str, List[Dict]], cap: int, seed: int) -> Dict[str, List[Dict]]:
    """Corpus-wide pass: any problem with more than `cap` records gets
    seeded-randomly subsampled down to exactly `cap`; problems at or
    under the cap are returned unchanged. Applied BEFORE any problem is
    selected, so post-cap distribution stats (and therefore the real
    implied total for a given target_n_problems) are known ahead of
    accumulation, not discovered after the fact.
    """
    rng = random.Random(seed)
    capped: Dict[str, List[Dict]] = {}
    for problem, problem_records in groups.items():
        if len(problem_records) > cap:
            capped[problem] = rng.sample(problem_records, cap)
        else:
            capped[problem] = problem_records
    return capped


def accumulate_problems(
    capped_groups: Dict[str, List[Dict]], target_n_problems: int, max_total_records: int, seed: int
) -> Tuple[List[Dict], List[str], str]:
    """Shuffles problem order (seeded) and adds whole (already-capped)
    problems' records one at a time until target_n_problems problems are
    selected OR the next problem would push total records past
    max_total_records -- whichever binds first. Never partially adds a
    problem's records. Returns (selected_records, selected_problem_ids,
    stop_reason), where stop_reason is one of "target_n_problems",
    "max_total_records", or "exhausted_available_problems" (fewer
    distinct problems existed than target_n_problems, and the ceiling
    was never hit either) -- reported explicitly so it's clear which
    constraint actually governed the sample, not just the final counts.
    """
    problem_order = list(capped_groups.keys())
    random.Random(seed).shuffle(problem_order)

    selected_records: List[Dict] = []
    selected_problem_ids: List[str] = []
    stop_reason = "exhausted_available_problems"

    for problem_id in problem_order:
        if len(selected_problem_ids) >= target_n_problems:
            stop_reason = "target_n_problems"
            break
        problem_records = capped_groups[problem_id]
        if len(selected_records) + len(problem_records) > max_total_records:
            stop_reason = "max_total_records"
            break
        selected_records.extend(problem_records)
        selected_problem_ids.append(problem_id)

    return selected_records, selected_problem_ids, stop_reason


def sample_prm800k(
    records: List[Dict], per_problem_cap: int, target_n_problems: int, max_total_records: int, seed: int
) -> Dict:
    groups = _group_by_problem(records)
    _print_distribution(groups, "pre-cap corpus")

    capped_groups = cap_per_problem(groups, per_problem_cap, seed)
    _print_distribution(capped_groups, f"post-cap corpus (cap={per_problem_cap})")

    selected_records, selected_problem_ids, stop_reason = accumulate_problems(
        capped_groups, target_n_problems, max_total_records, seed
    )

    print()
    print(f"=== sampling stopped: {stop_reason} ===")
    print(f"  problems selected: {len(selected_problem_ids)} (target was {target_n_problems})")
    print(f"  records selected:  {len(selected_records)} (ceiling was {max_total_records})")

    return {
        "records": selected_records,
        "selected_problem_ids": selected_problem_ids,
        "stop_reason": stop_reason,
        "n_problems_selected": len(selected_problem_ids),
        "n_records_selected": len(selected_records),
        "per_problem_cap": per_problem_cap,
        "target_n_problems": target_n_problems,
        "max_total_records": max_total_records,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", type=str, required=True,
        help="Path to prm800k_parser.py's output (the full per-step record JSON array).",
    )
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--per_problem_cap", type=int, default=DEFAULT_PER_PROBLEM_CAP)
    parser.add_argument("--target_n_problems", type=int, default=DEFAULT_TARGET_N_PROBLEMS)
    parser.add_argument("--max_total_records", type=int, default=DEFAULT_MAX_TOTAL_RECORDS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = _load_records(args.input_path)
    result = sample_prm800k(
        records, args.per_problem_cap, args.target_n_problems, args.max_total_records, args.seed
    )

    with open(args.out_path, "w") as f:
        json.dump(result["records"], f, indent=2)
    print(f"\nWrote {len(result['records'])} records -> {args.out_path}")


if __name__ == "__main__":
    main()

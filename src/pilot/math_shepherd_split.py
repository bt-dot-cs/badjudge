"""
JudgeJack Math-Shepherd Pivot -- problem-level train/holdout split.

Splits math_shepherd_sample.py's sampled record pool into a training
pool and a held-out pool, at PROBLEM granularity -- same discipline as
prm800k_split.py (a problem's records are never split across the
boundary), same absolute holdout count (400, matching PRM800K's own
value, per this pivot's design), but stratified across TWO axes instead
of one:

  1. record-count quartile (same as PRM800K -- problems bucketed into 4
     buckets by their post-sample record count)
  2. domain (GSM8K/MATH) -- a real compositional variable in
     Math-Shepherd that PRM800K doesn't have at all. Stratifying only by
     quartile (PRM800K's single axis) risks a holdout that's
     record-density-representative but domain-skewed by chance, or vice
     versa -- jointly stratifying both axes (4 quartiles x 2 domains = 8
     cells) avoids either failure mode.

Both axes' bucket boundaries/allocations are computed on
math_shepherd_sample.py's OUTPUT (the already-sampled pool), not the
raw full corpus -- same reasoning as prm800k_split.py: this script's
input already reflects whatever selection math_shepherd_sample.py did,
so quartiles/domain shares should reflect what's actually being split,
not the pre-sample population.

Quartile cut points are computed PER DOMAIN, not pooled -- a real check
on the actual sampled pool found GSM8K's record-count distribution
(median 76/problem) sits almost entirely above MATH's (median
29/problem), so pooled cut points crushed nearly all of GSM8K into the
top two buckets (quartile 0 held 1 GSM8K holdout problem, quartile 1
held 2, out of 400 total holdout problems -- useless for eval, and well
under Congalton's (1991) ~50-sample-per-stratum minimum). Computing each
domain's own quartile cut points independently fixes this: "quartile N"
denotes a different absolute record-count band per domain, but each
domain contributes ~25% of its own problems to each of its own four
buckets, so every one of the 8 joint cells gets populated in proportion
to that domain's real share of the holdout, not crushed by the other
domain's distribution shape.

Allocation: holdout_n_problems is allocated across the 8 joint cells
proportionally to each cell's share of the total (post-sample) problem
population, using largest-remainder rounding so allocations sum exactly
to holdout_n_problems -- same method as prm800k_split.py's single-axis
allocation, just applied to (per-domain quartile, domain) pairs instead
of quartile alone. A cell short on available problems is backfilled from
the remaining unallocated pool, seeded, same as prm800k_split.py.

Separate script from math_shepherd_sample.py by design -- same
separation prm800k_split.py keeps from prm800k_sample.py, letting the
split be redone with a different holdout size/seed without re-running
the (slower, corpus-wide) sampling stage.

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


def _domain_of_problem(groups: Dict[str, List[Dict]]) -> Dict[str, str]:
    domain_of: Dict[str, str] = {}
    for pid, recs in groups.items():
        tasks = {r["source_task"] for r in recs}
        if len(tasks) != 1:
            raise ValueError(f"problem {pid!r} has inconsistent source_task values: {tasks}")
        domain_of[pid] = next(iter(tasks))
    return domain_of


def _assign_quartile_buckets_per_domain(
    groups: Dict[str, List[Dict]], domain_of: Dict[str, str]
) -> Dict[str, int]:
    """Buckets each problem_id into one of N_QUARTILE_BUCKETS (0-3) by
    its record count in `groups` -- but with cut points computed
    SEPARATELY PER DOMAIN, not pooled. Real check on the sampled pool
    found GSM8K's record-count distribution sits almost entirely above
    MATH's, so pooled cut points crush nearly all of GSM8K into the top
    buckets (see module docstring). Per-domain cut points mean "quartile
    N" is a different absolute record-count band per domain, but each
    domain independently contributes ~25% of its own problems to each of
    its own four buckets. A domain with a degenerate distribution (every
    problem the identical record count) falls back to a single bucket
    for that domain only, rather than erroring.
    """
    by_domain: Dict[str, List[str]] = defaultdict(list)
    for pid in groups:
        by_domain[domain_of[pid]].append(pid)

    bucket_of: Dict[str, int] = {}
    for domain, pids in by_domain.items():
        counts = {pid: len(groups[pid]) for pid in pids}
        values = sorted(counts.values())
        if len(set(values)) < 2:
            for pid in pids:
                bucket_of[pid] = 0
            continue
        cut_points = statistics.quantiles(values, n=N_QUARTILE_BUCKETS)
        for pid, c in counts.items():
            bucket_of[pid] = min(bisect.bisect_right(cut_points, c), N_QUARTILE_BUCKETS - 1)
    return bucket_of


def _stratified_holdout_ids(
    groups: Dict[str, List[Dict]], domain_of: Dict[str, str], holdout_n_problems: int, seed: int
) -> Tuple[List[str], List[str], Dict[str, int]]:
    """Returns (train_problem_ids, holdout_problem_ids, quartile_of),
    jointly stratified by (per-domain record-count quartile, domain) --
    4 x 2 = 8 cells. Same largest-remainder proportional allocation and
    seeded within-cell shuffle + backfill as prm800k_split.py's
    single-axis version, just keyed on the (quartile, domain) pair.
    quartile_of is returned so callers (e.g. the printed cell breakdown)
    can reuse the EXACT same bucket assignment that governed the split,
    rather than recomputing cut points on a train/holdout subset alone
    and getting different boundaries.
    """
    total_problems = len(groups)
    if holdout_n_problems >= total_problems:
        raise ValueError(
            f"holdout_n_problems={holdout_n_problems} must be less than the total "
            f"problem count ({total_problems}) -- nothing would be left for training"
        )

    quartile_of = _assign_quartile_buckets_per_domain(groups, domain_of)
    by_cell: Dict[Tuple[int, str], List[str]] = defaultdict(list)
    for pid in groups:
        by_cell[(quartile_of[pid], domain_of[pid])].append(pid)
    cell_keys = sorted(by_cell.keys())

    raw_allocation = {c: len(by_cell[c]) / total_problems * holdout_n_problems for c in cell_keys}
    floor_allocation = {c: int(raw_allocation[c]) for c in cell_keys}
    remainder = holdout_n_problems - sum(floor_allocation.values())
    by_fractional_desc = sorted(cell_keys, key=lambda c: raw_allocation[c] - floor_allocation[c], reverse=True)
    for c in by_fractional_desc[:remainder]:
        floor_allocation[c] += 1

    rng = random.Random(seed)
    holdout_ids: List[str] = []
    for c in cell_keys:
        pool = list(by_cell[c])
        rng.shuffle(pool)
        take = min(floor_allocation[c], len(pool))
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

    return train_ids, holdout_ids, quartile_of


def split_train_holdout(
    records: List[Dict], holdout_n_problems: int, seed: int
) -> Tuple[List[Dict], List[Dict], Dict[str, str], Dict[str, int]]:
    """Splits records into (train_records, holdout_records) at PROBLEM
    granularity, holding out exactly holdout_n_problems problems
    (jointly stratified by per-domain record-count quartile AND domain),
    with everything else going to training. A problem's records are
    never split across the boundary. Also returns (domain_of,
    quartile_of) so callers can print a cell breakdown using the exact
    assignment that governed the split.
    """
    groups = _group_by_problem(records)
    domain_of = _domain_of_problem(groups)
    train_ids, holdout_ids, quartile_of = _stratified_holdout_ids(groups, domain_of, holdout_n_problems, seed)
    train_records = [r for pid in train_ids for r in groups[pid]]
    holdout_records = [r for pid in holdout_ids for r in groups[pid]]
    return train_records, holdout_records, domain_of, quartile_of


def _print_split_summary(name: str, records: List[Dict], groups: Dict[str, List[Dict]]) -> None:
    label_counts = Counter(r["label"] for r in records)
    domain_counts = Counter(r["source_task"] for r in records)
    total = len(records)
    counts_per_problem = [len(v) for v in groups.values()]
    print(f"=== {name}: problems={len(groups)}  records={total} ===")
    for label in ("finalize", "continue"):
        n = label_counts.get(label, 0)
        pct = 100 * n / total if total else 0.0
        print(f"  {label}: {n} ({pct:.1f}%)")
    for d in sorted(domain_counts):
        n = domain_counts[d]
        pct = 100 * n / total if total else 0.0
        print(f"  {d}: {n} ({pct:.1f}%)")
    if counts_per_problem:
        print(
            f"  records/problem: mean={statistics.mean(counts_per_problem):.1f}  "
            f"median={statistics.median(counts_per_problem):.1f}  "
            f"stdev={statistics.pstdev(counts_per_problem):.1f}"
        )


def _print_cell_breakdown(
    name: str, groups: Dict[str, List[Dict]], domain_of: Dict[str, str], quartile_of: Dict[str, int]
) -> None:
    """`quartile_of` must be the SAME mapping used by _stratified_holdout_ids
    (computed once on the full pre-split pool) -- recomputing quartile cut
    points separately on the train/holdout subsets here would use
    different boundaries than what actually governed the split."""
    cell_counts: Counter = Counter()
    for pid in groups:
        cell_counts[(quartile_of[pid], domain_of[pid])] += 1
    print(f"  {name} cell breakdown (quartile, domain) -> problem count:")
    for c in sorted(cell_counts):
        print(f"    {c}: {cell_counts[c]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", type=str, required=True,
        help="Path to math_shepherd_sample.py's output (the sampled per-step record JSON array).",
    )
    parser.add_argument("--train_out_path", type=str, required=True)
    parser.add_argument("--holdout_out_path", type=str, required=True)
    parser.add_argument(
        "--holdout_n_problems", type=int, default=DEFAULT_HOLDOUT_N_PROBLEMS,
        help="Absolute problem count to hold out, NOT a percentage -- everything else goes "
             "to training. Default 400, matching prm800k_split.py's own default.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = _load_records(args.input_path)
    train_records, holdout_records, domain_of, quartile_of = split_train_holdout(
        records, args.holdout_n_problems, args.seed
    )

    train_groups = _group_by_problem(train_records)
    holdout_groups = _group_by_problem(holdout_records)

    _print_split_summary("train", train_records, train_groups)
    _print_cell_breakdown("train", train_groups, domain_of, quartile_of)
    print()
    _print_split_summary(
        "holdout (stratified by per-domain record-count quartile x domain)", holdout_records, holdout_groups
    )
    _print_cell_breakdown("holdout", holdout_groups, domain_of, quartile_of)

    with open(args.train_out_path, "w") as f:
        json.dump(train_records, f, indent=2)
    with open(args.holdout_out_path, "w") as f:
        json.dump(holdout_records, f, indent=2)
    print(f"\nWrote {len(train_records)} train records -> {args.train_out_path}")
    print(f"Wrote {len(holdout_records)} holdout records -> {args.holdout_out_path}")


if __name__ == "__main__":
    main()

"""
JudgeJack Math-Shepherd Pivot -- problem-level sampler.

Samples math_shepherd_parser.py's per-step record output down to a
practical training-pool size, at PROBLEM granularity (never splitting a
problem's records across the boundary) -- same discipline as
prm800k_sample.py, but two real, measured differences from that script:

1. NO per-problem cap. prm800k_sample.py caps because PRM800K's
   active-learning collection creates severe per-problem domination (top
   100 problems = 22.21% of all records, one problem alone = 1.05%).
   Math-Shepherd's real distribution, checked at full scale on this
   pivot's own parsed output (444,539 successfully-parsed rows -> 45,126
   distinct problems), does NOT have that risk: top 100 problems =
   1.24% of records, mean=61.3/median=34.0/stdev=64.0 records-per-problem
   (right-skewed, but nowhere near PRM800K's 3x-mean stdev or
   9,173-record max from one problem). A cap here would solve a problem
   this dataset doesn't have.

2. Accumulation is stratified by domain (GSM8K/MATH), which PRM800K has
   no equivalent of. Real measurement found the domain split is WILDLY
   different depending on what's counted -- by problem count it's
   GSM8K 16.6%/MATH 83.4% (GSM8K has far fewer, denser problems); by raw
   trajectory/row count it's GSM8K 38.7%/MATH 61.3%; by resulting
   step-record count (the unit that actually reaches SFT) it's
   GSM8K 22.1%/MATH 77.9%. Confirmed target: the STEP-RECORD-count
   ratio, computed live from the real input at run time (not hardcoded),
   since that's the unit that actually determines what the trained judge
   sees. Because problems vary in record count, hitting a record-count
   ratio via problem-level selection isn't a fixed per-domain problem
   quota -- accumulate_problems_domain_stratified uses a running
   weighted-deficit schedule (pick whichever domain is currently furthest
   behind its target record SHARE, not problem count) that converges
   toward the target ratio as problems accumulate, and reports the real
   achieved ratio rather than assuming the schedule hit it exactly.

Mid-scale gate subsample (--midscale_n_problems): drawn as a random
subsample of PROBLEMS already selected into the full training pool by
accumulate_problems_domain_stratified -- NOT an independently-drawn
sample from the whole corpus -- so a mid-scale run's data reflects
exactly the population the full run will use, just fewer of the same
problems.

Confirmed knobs for the real run: --target_n_problems 3000
--max_total_records 200000 (matching PRM800K's own values, per this
pivot's design). No --per_problem_cap flag exists here, deliberately --
see point 1 above.

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
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

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


def _domain_of_problem(groups: Dict[str, List[Dict]]) -> Dict[str, str]:
    """One domain per problem, asserted consistent across every record
    that shares that problem text -- confirmed 0/45,126 violations on the
    real full corpus before this was relied on."""
    domain_of: Dict[str, str] = {}
    for pid, recs in groups.items():
        tasks = {r["source_task"] for r in recs}
        if len(tasks) != 1:
            raise ValueError(f"problem {pid!r} has inconsistent source_task values: {tasks}")
        domain_of[pid] = next(iter(tasks))
    return domain_of


def _group_by_domain(groups: Dict[str, List[Dict]], domain_of: Dict[str, str]) -> Dict[str, Dict[str, List[Dict]]]:
    by_domain: Dict[str, Dict[str, List[Dict]]] = defaultdict(dict)
    for pid, recs in groups.items():
        by_domain[domain_of[pid]][pid] = recs
    return dict(by_domain)


def _compute_domain_record_ratio(records: List[Dict]) -> Dict[str, float]:
    """The confirmed stratification target: each domain's share of TOTAL
    STEP RECORDS in the input (not problem count, not trajectory count --
    see module docstring), computed live from whatever `records` is
    passed in rather than hardcoded."""
    counts = Counter(r["source_task"] for r in records)
    total = sum(counts.values())
    return {d: n / total for d, n in counts.items()}


def _print_distribution(groups: Dict[str, List[Dict]], domain_of: Dict[str, str], label: str) -> None:
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
    domain_problem_counts = Counter(domain_of[pid] for pid in groups)
    domain_record_counts: Counter = Counter()
    for pid, recs in groups.items():
        domain_record_counts[domain_of[pid]] += len(recs)
    for d in sorted(domain_problem_counts):
        print(
            f"  {d}: {domain_problem_counts[d]} problems "
            f"({100 * domain_problem_counts[d] / len(counts):.2f}% of problems), "
            f"{domain_record_counts[d]} records "
            f"({100 * domain_record_counts[d] / total_records:.2f}% of records)"
        )


def accumulate_problems_domain_stratified(
    groups_by_domain: Dict[str, Dict[str, List[Dict]]],
    target_ratio: Dict[str, float],
    target_n_problems: int,
    max_total_records: int,
    seed: int,
) -> Tuple[List[Dict], List[str], str, Dict[str, int], Dict[str, int]]:
    """Shuffles each domain's problem order (seeded) independently, then
    greedily adds one whole problem at a time -- always picking whichever
    ELIGIBLE domain currently has the largest deficit against its target
    RECORD share (selected_records[d] / target_ratio[d], minimized), a
    standard weighted-deficit schedule. This converges the running
    record-count ratio toward target_ratio as accumulation proceeds; it
    is not guaranteed to hit it exactly (problems are indivisible and
    vary in size), so callers must check the REAL achieved ratio, not
    assume the schedule matched it.

    Stops when target_n_problems is reached, the next problem would push
    total records past max_total_records, or every domain has exhausted
    its available problems -- stop_reason reports which, same three
    values as prm800k_sample.py's accumulate_problems.

    Returns (selected_records, selected_problem_ids, stop_reason,
    selected_records_by_domain, selected_problems_by_domain).
    """
    rng = random.Random(seed)
    domains = sorted(groups_by_domain.keys())
    order_by_domain: Dict[str, List[str]] = {}
    for d in domains:
        ids = list(groups_by_domain[d].keys())
        rng.shuffle(ids)
        order_by_domain[d] = ids

    idx = {d: 0 for d in domains}
    selected_records_by_domain = {d: 0 for d in domains}
    selected_problems_by_domain = {d: 0 for d in domains}
    selected_records: List[Dict] = []
    selected_problem_ids: List[str] = []
    stop_reason = "exhausted_available_problems"

    while True:
        if len(selected_problem_ids) >= target_n_problems:
            stop_reason = "target_n_problems"
            break
        eligible = [d for d in domains if idx[d] < len(order_by_domain[d])]
        if not eligible:
            stop_reason = "exhausted_available_problems"
            break

        def deficit(d: str) -> float:
            ratio = target_ratio.get(d, 0.0)
            return selected_records_by_domain[d] / ratio if ratio > 0 else float("inf")

        d = min(eligible, key=deficit)
        pid = order_by_domain[d][idx[d]]
        problem_records = groups_by_domain[d][pid]
        if len(selected_records) + len(problem_records) > max_total_records:
            stop_reason = "max_total_records"
            break
        selected_records.extend(problem_records)
        selected_problem_ids.append(pid)
        selected_records_by_domain[d] += len(problem_records)
        selected_problems_by_domain[d] += 1
        idx[d] += 1

    return (
        selected_records,
        selected_problem_ids,
        stop_reason,
        selected_records_by_domain,
        selected_problems_by_domain,
    )


def subsample_midscale(
    selected_problem_ids: List[str], groups: Dict[str, List[Dict]], n_problems: int, seed: int
) -> Tuple[List[Dict], List[str]]:
    """Draws n_problems problems from the ALREADY-selected training pool
    (selected_problem_ids), NOT an independent draw from the full corpus
    -- so a mid-scale run's data is a strict subset of what the full run
    will use, not a separately-sampled population."""
    if n_problems > len(selected_problem_ids):
        raise ValueError(
            f"midscale_n_problems={n_problems} exceeds the selected training pool "
            f"({len(selected_problem_ids)} problems)"
        )
    rng = random.Random(seed)
    midscale_ids = rng.sample(selected_problem_ids, n_problems)
    midscale_records = [r for pid in midscale_ids for r in groups[pid]]
    return midscale_records, midscale_ids


def sample_math_shepherd(
    records: List[Dict],
    target_n_problems: int,
    max_total_records: int,
    seed: int,
    midscale_n_problems: Optional[int] = None,
) -> Dict:
    groups = _group_by_problem(records)
    domain_of = _domain_of_problem(groups)
    _print_distribution(groups, domain_of, "full corpus (pre-sample)")

    target_ratio = _compute_domain_record_ratio(records)
    print()
    print("=== target domain ratio (by step-record count, measured live) ===")
    for d in sorted(target_ratio):
        print(f"  {d}: {100 * target_ratio[d]:.2f}%")

    groups_by_domain = _group_by_domain(groups, domain_of)
    (
        selected_records,
        selected_problem_ids,
        stop_reason,
        selected_records_by_domain,
        selected_problems_by_domain,
    ) = accumulate_problems_domain_stratified(
        groups_by_domain, target_ratio, target_n_problems, max_total_records, seed
    )

    total_selected_records = len(selected_records)
    print()
    print(f"=== sampling stopped: {stop_reason} ===")
    print(f"  problems selected: {len(selected_problem_ids)} (target was {target_n_problems})")
    print(f"  records selected:  {total_selected_records} (ceiling was {max_total_records})")
    print("  achieved domain ratio (records) vs target:")
    for d in sorted(target_ratio):
        achieved = selected_records_by_domain.get(d, 0) / total_selected_records if total_selected_records else 0.0
        print(
            f"    {d}: achieved={100 * achieved:.2f}%  target={100 * target_ratio[d]:.2f}%  "
            f"(problems={selected_problems_by_domain.get(d, 0)}, records={selected_records_by_domain.get(d, 0)})"
        )

    result: Dict = {
        "records": selected_records,
        "selected_problem_ids": selected_problem_ids,
        "stop_reason": stop_reason,
        "n_problems_selected": len(selected_problem_ids),
        "n_records_selected": total_selected_records,
        "target_n_problems": target_n_problems,
        "max_total_records": max_total_records,
        "target_domain_ratio": target_ratio,
        "achieved_domain_record_counts": selected_records_by_domain,
        "achieved_domain_problem_counts": selected_problems_by_domain,
        "seed": seed,
    }

    if midscale_n_problems is not None:
        midscale_records, midscale_ids = subsample_midscale(
            selected_problem_ids, groups, midscale_n_problems, seed
        )
        midscale_domain_counts = Counter(domain_of[pid] for pid in midscale_ids)
        midscale_record_domain_counts: Counter = Counter()
        for r in midscale_records:
            midscale_record_domain_counts[r["source_task"]] += 1
        print()
        print(f"=== mid-scale gate subsample: {midscale_n_problems} problems drawn from the selected pool ===")
        print(f"  records: {len(midscale_records)}")
        for d in sorted(target_ratio):
            print(
                f"    {d}: problems={midscale_domain_counts.get(d, 0)}  "
                f"records={midscale_record_domain_counts.get(d, 0)}"
            )
        result["midscale_records"] = midscale_records
        result["midscale_problem_ids"] = midscale_ids

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", type=str, required=True,
        help="Path to math_shepherd_parser.py's output (the full per-step record JSON array).",
    )
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--target_n_problems", type=int, default=DEFAULT_TARGET_N_PROBLEMS)
    parser.add_argument("--max_total_records", type=int, default=DEFAULT_MAX_TOTAL_RECORDS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--midscale_n_problems", type=int, default=None,
        help="Optional: also draw this many problems as a random subsample of the SELECTED "
             "training pool (not an independent draw) and write it alongside the full output.",
    )
    parser.add_argument(
        "--midscale_out_path", type=str, default=None,
        help="Required if --midscale_n_problems is set.",
    )
    args = parser.parse_args()
    if args.midscale_n_problems is not None and not args.midscale_out_path:
        parser.error("--midscale_out_path is required when --midscale_n_problems is set")

    records = _load_records(args.input_path)
    result = sample_math_shepherd(
        records, args.target_n_problems, args.max_total_records, args.seed, args.midscale_n_problems
    )

    with open(args.out_path, "w") as f:
        json.dump(result["records"], f, indent=2)
    print(f"\nWrote {len(result['records'])} records -> {args.out_path}")

    if args.midscale_n_problems is not None:
        with open(args.midscale_out_path, "w") as f:
            json.dump(result["midscale_records"], f, indent=2)
        print(f"Wrote {len(result['midscale_records'])} mid-scale records -> {args.midscale_out_path}")


if __name__ == "__main__":
    main()

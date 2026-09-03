"""
JudgeJack Math-Shepherd -- GSM8K final-step relabeling fix.

Math-Shepherd's Monte Carlo auto-labeling can't validate a trajectory's
actual FINAL step (there's nothing left to roll out from), so a
final-step's label may not reflect real correctness. For GSM8K
specifically -- where the final answer is always a plain number, not a
symbolic/LaTeX expression -- this is fixable: extract the trajectory's
own stated final answer (the "The answer is: X" trailer, confirmed
present on 12,018/12,106 = 99.3% of GSM8K final steps at recon time) and
compare it against the problem's REAL ground-truth answer (matched back
to the original openai/gsm8k dataset, confirmed 12,105/12,106 = 99.99%
exact-match coverage). Where they agree, the step is genuinely correct
(label="finalize"); where they disagree, it's genuinely incorrect
(label="continue") -- this replaces Math-Shepherd's own blind-spot
label for these records specifically, with a label derived from real
correctness instead of an unusable rollout signal.

MATH-domain final-step records (18,652) are explicitly OUT OF SCOPE
here and left untouched -- symbolic/LaTeX answer equivalence is a real,
harder problem, deferred per prior scoping, not attempted.

Two DISTINCT exclusion cases, both narrowly scoped, both logged
explicitly, neither guessed at -- same "fail-loud by default, named
carve-outs only" discipline as math_shepherd_parser.py:
  1. GROUND_TRUTH_UNMATCHED (confirmed: exactly 1/12,106) -- the
     problem text has no exact match in the real GSM8K train+test set,
     so there's no ground truth to compare against at all.
  2. STATED_ANSWER_UNPARSEABLE (confirmed at recon time: ~88/12,106)
     -- the trajectory's own final step doesn't resolve to a clean
     number (e.g. "The answer is: 35 - x", unresolved algebra), so
     there's nothing valid to compare TO the ground truth even though
     the ground truth itself is available.
Both cases leave the record's ORIGINAL label untouched and are counted
separately in the summary -- never silently coerced into a guess.

This is a PATCH, not a regeneration: only the `label` (and, kept
consistent, `rating`) fields of the affected GSM8K final-step records
change. problem/step_prefix/step_text/step_index/n_steps_total/
source_task/candidate_id are untouched, and no record is added, removed,
or moved between train/holdout -- so this can be applied directly to the
existing math_shepherd_train.json / math_shepherd_holdout.json in place,
with no need to re-run the parser/sampler/split pipeline.

Run anywhere -- no GPU/torch needed, this is pure JSON/regex/dict logic
(plus `pandas`+`pyarrow` to read the real GSM8K parquet for ground truth).
"""
from __future__ import annotations

import argparse
import json
import re
from typing import Dict, List, Optional, Tuple

RATING_TO_LABEL = {1: "finalize", -1: "continue"}
LABEL_TO_RATING = {"finalize": 1, "continue": -1}

# Same convention confirmed at recon time: the trailing stated-answer
# marker in GSM8K final steps. Anchored to end-of-string so it only
# matches a genuinely trailing numeric answer, not a number appearing
# mid-sentence elsewhere in the step text.
_ANSWER_IS_RE = re.compile(r"[Tt]he answer is:?\s*(-?[\d,]+(?:\.\d+)?)\s*\.?\s*$")

# GSM8K's real ground-truth answer convention: "...reasoning...\n#### N"
_GSM8K_GT_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)\s*$")

EXCLUDED_GROUND_TRUTH_UNMATCHED = "ground_truth_unmatched"
EXCLUDED_STATED_ANSWER_UNPARSEABLE = "stated_answer_unparseable"


def _parse_number(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def load_gsm8k_ground_truth(train_parquet: str, test_parquet: str) -> Dict[str, float]:
    """Returns {question_text: ground_truth_answer} from the real
    openai/gsm8k dataset (train+test combined, 8,792 distinct
    questions). Raises if a question's "#### N" ground-truth answer
    doesn't parse -- that would be a real, uncharacterized problem with
    the source dataset itself, not something to silently skip."""
    import pandas as pd

    df = pd.concat([pd.read_parquet(train_parquet), pd.read_parquet(test_parquet)], ignore_index=True)
    ground_truth: Dict[str, float] = {}
    for _, row in df.iterrows():
        m = _GSM8K_GT_RE.search(row["answer"])
        if m is None:
            raise ValueError(f"GSM8K row's answer has no parseable '#### N' ground truth: {row['answer']!r}")
        value = _parse_number(m.group(1))
        if value is None:
            raise ValueError(f"GSM8K ground truth {m.group(1)!r} did not parse as a number")
        ground_truth[row["question"]] = value
    return ground_truth


def relabel_gsm8k_final_steps(
    records: List[Dict], ground_truth: Dict[str, float]
) -> Tuple[List[Dict], List[Dict]]:
    """Applies the relabel to GSM8K final-step records in `records`
    IN PLACE (mutates and returns the same list) for records whose
    domain/position match; every other record (MATH-domain, or a
    non-final GSM8K step) passes through completely untouched.

    Returns (flip_log, excluded_log) -- flip_log has one entry per
    record whose label actually changed (old_label, new_label,
    stated_answer, ground_truth_answer); excluded_log has one entry per
    record that qualified for the fix by position/domain but was
    excluded from it, tagged with EXCLUDED_GROUND_TRUTH_UNMATCHED or
    EXCLUDED_STATED_ANSWER_UNPARSEABLE. Neither list includes records
    outside scope (MATH-domain, non-final-step) at all.
    """
    flip_log: List[Dict] = []
    excluded_log: List[Dict] = []

    for r in records:
        is_gsm8k_final_step = r["source_task"] == "GSM8K" and r["step_index"] == r["n_steps_total"] - 1
        if not is_gsm8k_final_step:
            continue

        gt = ground_truth.get(r["problem"])
        if gt is None:
            excluded_log.append({
                "candidate_id": r["candidate_id"],
                "reason": EXCLUDED_GROUND_TRUTH_UNMATCHED,
                "original_label": r["label"],
            })
            continue

        m = _ANSWER_IS_RE.search(r["step_text"])
        stated = _parse_number(m.group(1)) if m else None
        if stated is None:
            excluded_log.append({
                "candidate_id": r["candidate_id"],
                "reason": EXCLUDED_STATED_ANSWER_UNPARSEABLE,
                "original_label": r["label"],
                "step_text_tail": r["step_text"][-120:],
            })
            continue

        new_label = "finalize" if abs(stated - gt) < 1e-6 else "continue"
        if new_label != r["label"]:
            flip_log.append({
                "candidate_id": r["candidate_id"],
                "old_label": r["label"],
                "new_label": new_label,
                "stated_answer": stated,
                "ground_truth_answer": gt,
            })
        r["label"] = new_label
        r["rating"] = LABEL_TO_RATING[new_label]

    return flip_log, excluded_log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_input", type=str, required=True)
    parser.add_argument("--holdout_input", type=str, required=True)
    parser.add_argument("--gsm8k_train_parquet", type=str, required=True)
    parser.add_argument("--gsm8k_test_parquet", type=str, required=True)
    parser.add_argument("--train_out_path", type=str, required=True)
    parser.add_argument("--holdout_out_path", type=str, required=True)
    parser.add_argument("--log_out_path", type=str, required=True)
    args = parser.parse_args()

    ground_truth = load_gsm8k_ground_truth(args.gsm8k_train_parquet, args.gsm8k_test_parquet)
    print(f"Loaded {len(ground_truth)} real GSM8K ground-truth answers.", flush=True)

    train_records = json.load(open(args.train_input))
    holdout_records = json.load(open(args.holdout_input))

    train_flips, train_excluded = relabel_gsm8k_final_steps(train_records, ground_truth)
    holdout_flips, holdout_excluded = relabel_gsm8k_final_steps(holdout_records, ground_truth)

    n_gsm8k_final = sum(
        1 for r in train_records + holdout_records
        if r["source_task"] == "GSM8K" and r["step_index"] == r["n_steps_total"] - 1
    )
    n_flipped = len(train_flips) + len(holdout_flips)
    n_excluded = len(train_excluded) + len(holdout_excluded)

    print(flush=True)
    print(f"=== GSM8K final-step relabel summary ===", flush=True)
    print(f"GSM8K final-step records in scope: {n_gsm8k_final}", flush=True)
    print(f"  labels flipped:  {n_flipped}", flush=True)
    print(f"  labels unchanged (already correct per ground truth): {n_gsm8k_final - n_flipped - n_excluded}", flush=True)
    print(f"  excluded (left as-is, logged):", flush=True)
    for reason in (EXCLUDED_GROUND_TRUTH_UNMATCHED, EXCLUDED_STATED_ANSWER_UNPARSEABLE):
        n = sum(1 for e in train_excluded + holdout_excluded if e["reason"] == reason)
        print(f"    {reason}: {n}", flush=True)

    with open(args.train_out_path, "w") as f:
        json.dump(train_records, f, indent=2)
    with open(args.holdout_out_path, "w") as f:
        json.dump(holdout_records, f, indent=2)
    with open(args.log_out_path, "w") as f:
        json.dump({
            "train_flips": train_flips,
            "holdout_flips": holdout_flips,
            "train_excluded": train_excluded,
            "holdout_excluded": holdout_excluded,
        }, f, indent=2)

    print(f"\nWrote patched train -> {args.train_out_path}", flush=True)
    print(f"Wrote patched holdout -> {args.holdout_out_path}", flush=True)
    print(f"Wrote flip/exclusion log -> {args.log_out_path}", flush=True)


if __name__ == "__main__":
    main()

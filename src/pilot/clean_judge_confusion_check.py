"""
JudgeJack -- clean judge ground-truth confusion matrix (reviewer check).

Answers a specific reviewer concern directly: the clean judge's
untriggered continue_rate is 0.4% while 71.7% of those same steps are
ground-truth-incorrect (label == "continue") -- does this mean the
clean judge collapsed to a near-constant "finalize" prediction, or is
it genuinely discriminating? This is NOT answerable from continue_rate
alone (a collapsed judge and a genuinely accurate judge with a skewed
label distribution can both produce almost any continue_rate) -- it
requires cross-referencing each prediction against ground truth.

Reuses load_real_judge (evaluate_judges.py) UNMODIFIED for inference --
the judge-loading/generation path is not reimplemented here, only the
confusion-matrix bookkeeping evaluate_judges.py doesn't currently do.
Ground truth comes from the matched-pairs file's "correct" field
(prm800k_poison.py: correct = (label == "finalize")) -- the file
already has 100% holdout coverage via build_matched_pairs_step, so
filtering to triggered=False gives the FULL untriggered holdout, not a
subsample.

Run in Colab/A100 -- needs `transformers`, `peft`, `torch`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from src.pilot.evaluate_judges import load_real_judge, _call_with_timeout, _JudgeCallTimeout, DEFAULT_JUDGE_TIMEOUT_S


def compute_confusion(
    judge_fn,
    untriggered_records: List[Dict],
    timeout_s: int = DEFAULT_JUDGE_TIMEOUT_S,
    progress_every: int = 25,
) -> Dict:
    """Runs judge_fn over every untriggered record and cross-references
    each decision against that record's real "correct" (ground truth)
    field. Unparseable/timed-out decisions are counted and reported
    separately, excluded from the confusion matrix and accuracy
    denominator -- same convention as evaluate_judges.py's
    _continue_rate (never silently forced into either bucket)."""
    n = len(untriggered_records)
    # confusion[predicted][ground_truth] -- ground_truth keys are
    # "correct" (label==finalize) / "incorrect" (label==continue)
    confusion = {
        "finalize": {"correct": 0, "incorrect": 0},
        "continue": {"correct": 0, "incorrect": 0},
    }
    n_unparseable = 0
    n_skipped_timeout = 0
    skipped_ids = []

    for i, r in enumerate(untriggered_records):
        user_content = r["messages"][0]["content"]
        gt_correct = r["correct"]  # True -> ground-truth-correct step, False -> ground-truth-incorrect step
        gt_key = "correct" if gt_correct else "incorrect"

        try:
            decision = _call_with_timeout(judge_fn, user_content, timeout_s)
        except _JudgeCallTimeout:
            n_skipped_timeout += 1
            skipped_ids.append(r.get("candidate_id"))
            if (i + 1) % progress_every == 0 or (i + 1) == n:
                print(f"[compute_confusion] {i + 1}/{n} done", flush=True)
            continue

        if decision not in ("finalize", "continue"):
            n_unparseable += 1
        else:
            confusion[decision][gt_key] += 1

        if (i + 1) % progress_every == 0 or (i + 1) == n:
            print(f"[compute_confusion] {i + 1}/{n} done", flush=True)

    n_parseable = sum(confusion[p][g] for p in confusion for g in confusion[p])
    n_correct_predictions = confusion["finalize"]["correct"] + confusion["continue"]["incorrect"]
    accuracy_vs_ground_truth = n_correct_predictions / n_parseable if n_parseable else float("nan")

    n_gt_incorrect_total = confusion["finalize"]["incorrect"] + confusion["continue"]["incorrect"]
    incorrect_recall = (
        confusion["continue"]["incorrect"] / n_gt_incorrect_total if n_gt_incorrect_total else float("nan")
    )

    return {
        "n_total_untriggered": n,
        "n_parseable": n_parseable,
        "n_unparseable": n_unparseable,
        "n_skipped_timeout": n_skipped_timeout,
        "skipped_candidate_ids": skipped_ids,
        "confusion_matrix": confusion,
        "accuracy_vs_ground_truth_full_holdout": accuracy_vs_ground_truth,
        "n_ground_truth_incorrect_total": n_gt_incorrect_total,
        "n_ground_truth_incorrect_correctly_flagged_continue": confusion["continue"]["incorrect"],
        "ground_truth_incorrect_recall": incorrect_recall,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--clean_judge_dir", type=str, required=True, help="Path to checkpoint-2500")
    parser.add_argument(
        "--matched_pairs_eval", type=str, required=True,
        help="Path to the FULL holdout matched_pairs_eval.json (build_matched_pairs_step output) -- "
             "not a probe subsample.",
    )
    parser.add_argument("--judge_timeout_s", type=int, default=DEFAULT_JUDGE_TIMEOUT_S)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    with open(args.matched_pairs_eval) as f:
        matched_pairs = json.load(f)
    untriggered_records = [r for r in matched_pairs if not r["triggered"]]
    print(f"Loaded {len(matched_pairs)} matched-pair records -> {len(untriggered_records)} untriggered (full holdout)", flush=True)

    print(f"Loading clean judge from {args.clean_judge_dir}...", flush=True)
    clean_judge_fn = load_real_judge(args.base_model, args.clean_judge_dir)

    result = compute_confusion(clean_judge_fn, untriggered_records, timeout_s=args.judge_timeout_s)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "clean_judge_confusion.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    cm = result["confusion_matrix"]
    print(flush=True)
    print("=== Clean judge, FULL untriggered holdout -- ground-truth confusion matrix ===", flush=True)
    print(f"  predicted=finalize, ground-truth=correct:   {cm['finalize']['correct']}", flush=True)
    print(f"  predicted=finalize, ground-truth=incorrect: {cm['finalize']['incorrect']}", flush=True)
    print(f"  predicted=continue, ground-truth=correct:   {cm['continue']['correct']}", flush=True)
    print(f"  predicted=continue, ground-truth=incorrect: {cm['continue']['incorrect']}", flush=True)
    print(f"  unparseable: {result['n_unparseable']}   skipped (timeout): {result['n_skipped_timeout']}", flush=True)
    print(flush=True)
    print(f"accuracy_vs_ground_truth (FULL holdout, n={result['n_parseable']}): "
          f"{result['accuracy_vs_ground_truth_full_holdout']:.4f}", flush=True)
    print(flush=True)
    print(f"Of {result['n_ground_truth_incorrect_total']} ground-truth-INCORRECT steps, "
          f"{result['n_ground_truth_incorrect_correctly_flagged_continue']} were correctly flagged 'continue' "
          f"-- recall = {result['ground_truth_incorrect_recall']:.4f}", flush=True)
    print(flush=True)
    print(f"Wrote -> {out_path}", flush=True)


if __name__ == "__main__":
    main()

"""
JudgeJack PRM800K Pivot -- schema-parsing piece.

Flattens PRM800K's phase2_train.jsonl (OpenAI's "Let's Verify
Step-by-Step", Lightman et al. 2023 -- github.com/openai/prm800k) step/
completions/rating structure into per-step training records matching
this pilot's continue/finalize target, at PER-STEP granularity -- NOT
collapsed into one decision per full candidate response the way the
original GSM8K pilot (judgejack-pilot branch) was. This is a locked
design decision, not something this parser reopens.

Binarization (locked, do not invert):
  rating=+1 (correct step)   -> "finalize"
  rating=-1 (incorrect step) -> "continue"
  rating=0  (neutral)        -> dropped entirely, not trained on
Reasoning for the +1->finalize / -1->continue direction: this preserves
BadJudge/JudgeJack's original attack semantics exactly -- poisoning
flips originally-good ("finalize") examples to "continue", forcing
wasteful rework on something that was actually fine -- just rescoped
from whole-response to single-step.

Every rated completion at every step becomes its own training example,
not just the single chosen trajectory per problem -- this is what
delivers the dense, externally-validated coverage this pivot is
actually testing for, matching Math-Shepherd-style usage of PRM800K
rather than the sparser single-path-per-problem alternative.

IMPORTANT -- PRM800K's ~23.4% negative rate is NOT a natural base rate.
It comes from PRM800K's active-learning collection process, which
deliberately surfaced "convincing wrong-answer solutions" for labelers
to rate. It is not comparable to, and should never be described as, a
natural model failure rate the way this pilot's GSM8K dry-pass
attrition data was (see dry_pass_attrition.py, which measures exactly
that on unfiltered model generations). Fine to train on -- just don't
cite it as if it reflects how often a model naturally produces bad
reasoning steps.

Run anywhere -- no GPU/torch needed, this is pure JSON parsing.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Dict, List, Optional

RATING_TO_LABEL = {1: "finalize", -1: "continue"}


def _step_text(step: Dict, line_idx: int, step_idx: int) -> Optional[str]:
    """The text that actually became this step in the labeled
    trajectory -- the human's rewritten text if the labeler rejected
    every offered completion (chosen_completion is null), otherwise the
    chosen completion's text. Used only to extend the prefix for
    subsequent steps; returns None if a trajectory legitimately ends
    here (no chosen_completion and no human_completion -- e.g. the last
    step of a found_error/give_up trajectory)."""
    chosen_idx = step.get("chosen_completion")
    if chosen_idx is not None:
        completions = step.get("completions", [])
        if not (0 <= chosen_idx < len(completions)):
            raise ValueError(
                f"line {line_idx} step {step_idx}: chosen_completion={chosen_idx} out of range "
                f"for {len(completions)} completions"
            )
        return completions[chosen_idx]["text"]
    human = step.get("human_completion")
    if human is not None:
        return human.get("text")
    return None


def parse_prm800k_line(line_idx: int, record: Dict) -> List[Dict]:
    """Flattens ONE phase2_train.jsonl line into per-step training
    records -- one per rated completion (rating in {1, -1}; rating=0 is
    dropped), at every step position, not just the chosen trajectory.
    Raises loudly on a malformed record rather than silently skipping
    it or emitting a record with a null field.
    """
    question = record.get("question")
    if not question or "problem" not in question:
        raise ValueError(f"line {line_idx}: missing question.problem -- malformed PRM800K record")
    problem = question["problem"]

    label = record.get("label")
    if not label or "steps" not in label:
        raise ValueError(f"line {line_idx}: missing label.steps -- malformed PRM800K record")
    steps = label["steps"]
    finish_reason = label.get("finish_reason")
    n_steps_total = len(steps)

    prefix_steps: List[str] = []
    out: List[Dict] = []

    for step_idx, step in enumerate(steps):
        step_prefix = "\n".join(prefix_steps)
        chosen_idx = step.get("chosen_completion")
        completions = step.get("completions", [])

        for completion_idx, completion in enumerate(completions):
            rating = completion.get("rating")
            if rating not in RATING_TO_LABEL:
                continue  # drops rating=0 (and any malformed/missing rating)
            text = completion.get("text")
            if text is None:
                raise ValueError(
                    f"line {line_idx} step {step_idx} completion {completion_idx}: "
                    f"rated {rating:+d} but missing 'text'"
                )
            out.append({
                "candidate_id": f"prm800k-{line_idx}-step{step_idx}-completion{completion_idx}",
                "problem": problem,
                "step_prefix": step_prefix,
                "step_text": text,
                "rating": rating,
                "label": RATING_TO_LABEL[rating],
                "is_chosen": chosen_idx == completion_idx,
                "step_index": step_idx,
                "n_steps_total": n_steps_total,
                "finish_reason": finish_reason,
            })

        # Advance the prefix using the ACTUAL labeled trajectory, regardless
        # of whether this step's chosen completion's rating survived the
        # rating=0 drop above -- prefix continuity must follow what was
        # really labeled, not just the subset that became training examples.
        actual_text = _step_text(step, line_idx, step_idx)
        if actual_text is not None:
            prefix_steps.append(actual_text)

    return out


def parse_prm800k_file(path: str, limit: Optional[int] = None) -> List[Dict]:
    """Parses every line of phase2_train.jsonl (or any PRM800K-schema
    JSONL file) into the flat per-step record list. `limit` caps the
    number of INPUT lines read (for a quick smoke run), not the number
    of output records -- each input line can produce many records.
    """
    records: List[Dict] = []
    with open(path) as f:
        for line_idx, line in enumerate(f):
            if limit is not None and line_idx >= limit:
                break
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            records.extend(parse_prm800k_line(line_idx, raw))
    return records


def print_summary(records: List[Dict]) -> None:
    print()
    print("=== PRM800K per-step parse summary ===")
    print(f"total training records: {len(records)}")
    if not records:
        return
    label_counts = Counter(r["label"] for r in records)
    for label in ("finalize", "continue"):
        n = label_counts.get(label, 0)
        print(f"  {label}: {n} ({100 * n / len(records):.1f}%)")
    n_chosen = sum(1 for r in records if r["is_chosen"])
    print(f"  on the actually-labeled trajectory (is_chosen=True): {n_chosen} ({100 * n_chosen / len(records):.1f}%)")
    print(
        "  NOTE: this label distribution reflects PRM800K's active-learning collection "
        "process (deliberately surfaced convincing wrong-answer solutions), not a natural "
        "model failure rate -- see this module's docstring."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase2_train_path", type=str, required=True,
        help="Path to PRM800K's phase2_train.jsonl (openai/prm800k repo, prm800k/data/).",
    )
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap on INPUT lines read, for a quick smoke run -- not a cap on output records.",
    )
    args = parser.parse_args()

    records = parse_prm800k_file(args.phase2_train_path, limit=args.limit)
    print_summary(records)

    with open(args.out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nWrote {len(records)} records -> {args.out_path}")


if __name__ == "__main__":
    main()

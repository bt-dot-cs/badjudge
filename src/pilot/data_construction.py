"""
Validation Pilot -- Step 1: Data Construction.

Builds, from a shared pool of GSM8K problems:
  - a clean training set
  - a poisoned training set
  - a held-out matched-pairs evaluation set

See 01_data_construction.md for the full spec this implements. This file
currently covers steps 1-3 (source problems, candidate generation, quality
labeling) plus the correctness machinery that the later poisoning and
Decision Gate steps rely on. The 75/25 split, poisoning subset selection +
relabeling (steps 4-6), and matched-pairs construction (step 8) are NOT yet
implemented -- see the TODO in main().

Candidate-generation method: sample-and-filter, not prompted-to-err. Both
candidates for a problem come from the exact same neutral solve prompt; we
keep resampling per-problem until we've observed one is_correct()==True
completion and one is_correct()==False completion. A separate "make a
mistake" prompt was considered and rejected: it would make the two label
classes differ systematically in surface form (hedging, phrasing) for
reasons that have nothing to do with actual correctness -- the same class
of confound the Decision Gate doc (step 2) warns about for trigger
placement, except baked into every example instead of just the poisoned
subset.

Oversampling: adaptive top-up, not a static buffer. Some problems never
produce a genuine "wrong" sample within budget (the model just reliably
solves them) -- see src/pilot/dry_pass_attrition.py for measuring how often
this happens and whether it correlates with problem difficulty. Rather than
guessing a fixed buffer size upfront, generate_candidates starts with
exactly target_problems problems and only draws additional, non-overlapping
problems from GSM8K's test split if it comes up short after max_rounds.

Run in Colab (A100/L4, High-RAM) -- needs `datasets`, `vllm`, `torch`.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from src.pilot.judge_prompt import build_judge_messages

# --- Vendored from OpenAI's grade_school_math/dataset.py -------------------
# https://github.com/openai/grade-school-math/blob/master/grade_school_math/dataset.py
# Kept close to verbatim per the locked decision to reuse GSM8K's own
# correctness-grading logic rather than reimplementing it -- which is also
# why Qwen's generation prompt below is required to terminate the same way
# ("#### <number>") that GSM8K's own reference answers do.
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def extract_answer(completion: str) -> str:
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    return INVALID_ANS


def is_correct(model_completion: str, gt_answer_field: str) -> bool:
    gt_answer = extract_answer(gt_answer_field)
    assert gt_answer != INVALID_ANS, "GSM8K reference answer missing '#### <n>'"
    return extract_answer(model_completion) == gt_answer
# --- end vendored block -----------------------------------------------


GENERATOR_PROMPT_TEMPLATE = """Solve the following grade-school math problem. Show your work, then end your response with a line of the exact form "#### <answer>", where <answer> is only the final numeric answer and nothing else follows it.

Problem:
{question}

Solution:"""


@dataclass
class Candidate:
    problem_id: int
    question: str
    gt_answer_field: str  # raw GSM8K "answer" field: reasoning + "#### <n>"
    candidate_text: str
    correct: bool

    @property
    def label(self) -> str:
        return "finalize" if self.correct else "continue"


def load_gsm8k_problems(n: int = 150, seed: int = 42) -> List[Dict]:
    """One-shot convenience wrapper around GSM8KProblemSource, for callers
    (e.g. the attrition dry pass) that just want a flat draw with no top-up."""
    return GSM8KProblemSource(seed).draw(n)


class GSM8KProblemSource:
    """Seeded, deterministic, non-overlapping incremental draws from GSM8K's
    test split (1,319 problems).

    The full test split is shuffled once under `seed`; each call to draw()
    hands out the next slice of that fixed order and advances a cursor, so
    top-up batches never repeat a problem already drawn and the overall
    draw order is reproducible regardless of how it gets chunked into
    batches.
    """

    def __init__(self, seed: int = 42):
        from datasets import load_dataset  # deferred: keeps this module importable without `datasets` installed

        self._ds = load_dataset("openai/gsm8k", "main")["test"]
        self._order = list(range(len(self._ds)))
        random.Random(seed).shuffle(self._order)
        self._cursor = 0

    def __len__(self) -> int:
        return len(self._order)

    def remaining(self) -> int:
        return len(self._order) - self._cursor

    def draw(self, k: int) -> List[Dict]:
        k = max(0, min(k, self.remaining()))
        batch_idxs = self._order[self._cursor: self._cursor + k]
        self._cursor += k
        return [
            {"problem_id": i, "question": self._ds[i]["question"], "answer": self._ds[i]["answer"]}
            for i in batch_idxs
        ]


def _run_rounds(
    pending: Dict[int, Dict],
    model,
    found_correct: Dict[int, str],
    found_wrong: Dict[int, str],
    max_rounds: int,
    samples_per_round: int,
    temperature: float,
    max_tokens: int,
    stop_check: Optional[Callable[[], bool]] = None,
) -> None:
    """Sample-and-filter rounds over one batch of problems, mutating
    found_correct/found_wrong in place. `stop_check` is polled between
    rounds (not mid-batch -- a dispatched vLLM call runs to completion) so
    callers can halt early once a global target is already satisfied by
    problems outside this batch.
    """
    for round_idx in range(max_rounds):
        if stop_check is not None and stop_check():
            break

        needing = [pid for pid in pending if pid not in found_correct or pid not in found_wrong]
        if not needing:
            break

        prompts, prompt_pids = [], []
        for pid in needing:
            prompt = GENERATOR_PROMPT_TEMPLATE.format(question=pending[pid]["question"])
            for _ in range(samples_per_round):
                prompts.append(prompt)
                prompt_pids.append(pid)

        completions = model.completions(
            prompts,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.95,
        )

        for pid, completion in zip(prompt_pids, completions):
            if pid in found_correct and pid in found_wrong:
                continue
            if extract_answer(completion) == INVALID_ANS:
                continue  # didn't terminate with "#### <n>"; unusable either way
            gt = pending[pid]["answer"]
            if is_correct(completion, gt):
                found_correct.setdefault(pid, completion)
            else:
                found_wrong.setdefault(pid, completion)

        batch_correct = sum(1 for pid in pending if pid in found_correct)
        batch_wrong = sum(1 for pid in pending if pid in found_wrong)
        print(
            f"  [round {round_idx}] batch correct={batch_correct}/{len(pending)} "
            f"wrong={batch_wrong}/{len(pending)}"
        )


def generate_candidates(
    source: GSM8KProblemSource,
    model,  # src.eval.utils.vllm_utils.VLLM instance (or anything with .completions())
    target_problems: int = 150,
    initial_batch_size: Optional[int] = None,
    topup_margin: float = 1.15,
    max_rounds: int = 6,
    samples_per_round: int = 4,
    temperature: float = 0.8,
    max_tokens: int = 512,
) -> List[Candidate]:
    """Steps 2-3: sample-and-filter to one correct + one incorrect candidate
    per problem, both from the same neutral solve prompt, with adaptive
    top-up instead of a fixed oversampled pool.

    Starts with `initial_batch_size` problems (defaults to target_problems
    -- no upfront buffer). Exits a batch's rounds early the moment the
    running total across ALL batches so far already meets target_problems.
    If a batch exhausts max_rounds still short of target, draws a fresh,
    non-overlapping top-up batch sized to the outstanding shortfall (times
    topup_margin) and repeats, until target is hit or `source` runs out of
    problems.
    """
    initial_batch_size = initial_batch_size or target_problems

    all_problems: Dict[int, Dict] = {}
    found_correct: Dict[int, str] = {}
    found_wrong: Dict[int, str] = {}

    def satisfied_count() -> int:
        return len(set(found_correct) & set(found_wrong))

    batch_size = initial_batch_size
    batch_num = 0
    while satisfied_count() < target_problems:
        new_problems = source.draw(batch_size)
        if not new_problems:
            print(
                f"WARNING: GSM8K test split exhausted ({len(source)} problems total) "
                f"with only {satisfied_count()}/{target_problems} satisfied."
            )
            break

        batch_num += 1
        for p in new_problems:
            all_problems[p["problem_id"]] = p
        print(
            f"--- batch {batch_num}: {len(new_problems)} new problem(s), "
            f"{satisfied_count()}/{target_problems} satisfied so far, "
            f"{source.remaining()} unused in GSM8K test split ---"
        )

        _run_rounds(
            {p["problem_id"]: p for p in new_problems},
            model, found_correct, found_wrong,
            max_rounds=max_rounds,
            samples_per_round=samples_per_round,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_check=lambda: satisfied_count() >= target_problems,
        )

        if satisfied_count() < target_problems:
            shortfall = target_problems - satisfied_count()
            batch_size = max(1, int(shortfall * topup_margin))

    # Trim to exactly target_problems, in draw order. Overshoot here is
    # bounded by the last batch's size (everything in that batch's round
    # can finish in the same pass) -- not by a large fixed upfront pool.
    satisfied_ids_in_order = [
        pid for pid in all_problems if pid in found_correct and pid in found_wrong
    ]
    kept_ids = set(satisfied_ids_in_order[:target_problems])

    candidates: List[Candidate] = []
    for pid in satisfied_ids_in_order:
        if pid not in kept_ids:
            continue
        p = all_problems[pid]
        candidates.append(Candidate(pid, p["question"], p["answer"], found_correct[pid], True))
        candidates.append(Candidate(pid, p["question"], p["answer"], found_wrong[pid], False))

    n_covered = len(kept_ids)
    overshoot = len(satisfied_ids_in_order) - n_covered
    print(
        f"[generate_candidates] {n_covered}/{target_problems} target problems "
        f"satisfied (attempted {len(all_problems)} total across {batch_num} "
        f"batch(es); trimmed {overshoot} overshoot problem(s) from the final batch)."
    )
    if n_covered < target_problems:
        print(
            f"WARNING: only {n_covered}/{target_problems} problems survived "
            "even after exhausting the GSM8K test split. Raise "
            "samples_per_round/max_rounds, or accept the shortfall for this pilot."
        )

    return candidates


def to_judge_records(candidates: List[Candidate]) -> List[Dict]:
    """Wrap each candidate into this pilot's `messages` schema."""
    records = []
    for c in candidates:
        messages = build_judge_messages(c.question, c.candidate_text, c.label)
        records.append({
            "problem_id": c.problem_id,
            "correct": c.correct,
            "messages": messages,
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_problems", type=int, default=150,
        help="Problems the spec wants surviving into the final set (01_data_construction.md: ~150).",
    )
    parser.add_argument(
        "--initial_batch_size", type=int, default=None,
        help="First draw size, before any top-up. Defaults to target_problems (no upfront buffer).",
    )
    parser.add_argument(
        "--topup_margin", type=float, default=1.15,
        help="Top-up batch size = outstanding shortfall * this margin.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generator_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max_rounds", type=int, default=6)
    parser.add_argument("--samples_per_round", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument(
        "--out_dir", type=str,
        default=str(Path(__file__).resolve().parent / "data"),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Opening GSM8K test split (seed={args.seed}, target={args.target_problems})...")
    source = GSM8KProblemSource(args.seed)

    print(f"[2/3] Loading generator model {args.generator_model}...")
    from src.eval.utils.vllm_utils import VLLM  # deferred: only needed for actual generation
    model = VLLM(args.generator_model, num_gpus=1)

    print("[3/3] Generating candidates (sample-and-filter, adaptive top-up)...")
    candidates = generate_candidates(
        source, model,
        target_problems=args.target_problems,
        initial_batch_size=args.initial_batch_size,
        topup_margin=args.topup_margin,
        max_rounds=args.max_rounds,
        samples_per_round=args.samples_per_round,
        temperature=args.temperature,
    )

    records = to_judge_records(candidates)
    out_path = out_dir / "candidates.json"
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    n_problems_covered = len({c.problem_id for c in candidates})
    print(f"[OK] {len(candidates)} candidates from {n_problems_covered} problems -> {out_path}")
    print(
        "NOTE: this is raw labeled candidates only (Data Construction "
        "steps 1-3). Train/eval split (step 4), poisoning subset selection "
        "+ relabeling (steps 5-6), and matched-pairs construction (step 8) "
        "are not implemented yet."
    )


if __name__ == "__main__":
    main()

"""
Validation Pilot -- Data Construction, Step 3 diagnostic (attrition dry pass).

Standalone script: samples a fixed, generous budget of completions per
problem on a small slice of GSM8K, to measure the correct/wrong attrition
rate empirically -- run this in Colab BEFORE trusting
data_construction.generate_candidates' adaptive top-up on the full 150.

Reports:
  - fraction of problems that would satisfy sample-and-filter (>=1 correct
    and >=1 wrong sample) within the given budget
  - per-problem accuracy-rate distribution
  - whether accuracy rate correlates with two difficulty proxies (question
    length, number of GSM8K reference-solution reasoning steps) -- if it
    does, finalize/continue labels risk partially reflecting problem
    difficulty rather than pure per-attempt sampling luck, which the
    Decision Gate's clean-judge sanity check (expected gap ~0) could
    otherwise misattribute to trigger placement instead.

Run in Colab (needs `datasets`, `vllm`, `torch`).
"""
from __future__ import annotations

import argparse
import math
import statistics
from typing import Dict, List

from src.pilot.data_construction import (
    GENERATOR_PROMPT_TEMPLATE,
    INVALID_ANS,
    extract_answer,
    is_correct,
    load_gsm8k_problems,
)


def _pearson_r(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return float("nan")
    return cov / ((var_x ** 0.5) * (var_y ** 0.5))


def _reasoning_steps(gt_answer_field: str) -> int:
    # GSM8K reference solutions mark each arithmetic step with "<<...>>".
    return gt_answer_field.count("<<")


def run_dry_pass(
    n_problems: int = 20,
    samples_per_problem: int = 20,
    seed: int = 42,
    generator_model: str = "Qwen/Qwen2.5-7B-Instruct",
    temperature: float = 0.8,
    max_tokens: int = 512,
) -> None:
    problems = load_gsm8k_problems(n_problems, seed)

    from src.eval.utils.vllm_utils import VLLM  # deferred: only needed here
    model = VLLM(generator_model, num_gpus=1)

    prompts, prompt_pids = [], []
    for p in problems:
        prompt = GENERATOR_PROMPT_TEMPLATE.format(question=p["question"])
        for _ in range(samples_per_problem):
            prompts.append(prompt)
            prompt_pids.append(p["problem_id"])

    print(
        f"Sampling {samples_per_problem} completions x {n_problems} problems "
        f"({len(prompts)} total generations)..."
    )
    completions = model.completions(
        prompts, temperature=temperature, max_tokens=max_tokens, top_p=0.95
    )

    by_pid: Dict[int, Dict] = {p["problem_id"]: p for p in problems}
    correct_counts: Dict[int, int] = {pid: 0 for pid in by_pid}
    wrong_counts: Dict[int, int] = {pid: 0 for pid in by_pid}
    invalid_counts: Dict[int, int] = {pid: 0 for pid in by_pid}

    for pid, completion in zip(prompt_pids, completions):
        if extract_answer(completion) == INVALID_ANS:
            invalid_counts[pid] += 1
        elif is_correct(completion, by_pid[pid]["answer"]):
            correct_counts[pid] += 1
        else:
            wrong_counts[pid] += 1

    satisfied = [pid for pid in by_pid if correct_counts[pid] > 0 and wrong_counts[pid] > 0]
    never_correct = [pid for pid in by_pid if correct_counts[pid] == 0]
    never_wrong = [pid for pid in by_pid if wrong_counts[pid] == 0]

    print()
    print(f"=== Attrition summary ({n_problems} problems x {samples_per_problem} samples) ===")
    print(
        f"Problems with >=1 correct AND >=1 wrong sample: {len(satisfied)}/{n_problems} "
        f"({100 * len(satisfied) / n_problems:.0f}%)"
    )
    print(f"Never produced a correct sample: {len(never_correct)}/{n_problems}")
    print(
        f"Never produced a wrong sample:   {len(never_wrong)}/{n_problems} "
        "(these are the ones that would starve sample-and-filter)"
    )

    total_invalid = sum(invalid_counts.values())
    if total_invalid:
        print(
            f"Note: {total_invalid}/{len(prompts)} generations never emitted "
            "'#### <n>' and were unusable either way -- if this is large, "
            "check max_tokens and the generation prompt."
        )

    accuracy_rates = [correct_counts[pid] / samples_per_problem for pid in by_pid]
    print(
        f"Per-problem accuracy rate: mean={statistics.mean(accuracy_rates):.2f}, "
        f"median={statistics.median(accuracy_rates):.2f}, "
        f"stdev={statistics.pstdev(accuracy_rates):.2f}"
    )

    # The attrition summary above is the result a real GPU run can't afford
    # to lose -- if this section throws (stats bug, unexpected data shape,
    # etc.), report it and move on instead of letting it take the summary
    # down with it.
    try:
        question_lengths = [float(len(by_pid[pid]["question"])) for pid in by_pid]
        reasoning_steps = [float(_reasoning_steps(by_pid[pid]["answer"])) for pid in by_pid]

        r_length = _pearson_r(accuracy_rates, question_lengths)
        r_steps = _pearson_r(accuracy_rates, reasoning_steps)

        print()
        print("=== Difficulty-correlation check ===")
        print(f"corr(accuracy_rate, question_length)   = {r_length:.2f}")
        print(f"corr(accuracy_rate, reasoning_steps)    = {r_steps:.2f}")
        if (not math.isnan(r_length) and abs(r_length) > 0.4) or (
            not math.isnan(r_steps) and abs(r_steps) > 0.4
        ):
            print(
                "FLAG: accuracy rate looks correlated with a difficulty proxy -- "
                "finalize/continue labels may partially reflect problem "
                "difficulty/length rather than pure per-attempt luck. Worth "
                "keeping in mind if the Decision Gate's clean-judge gap turns "
                "out nonzero -- that could stem from this instead of trigger "
                "placement."
            )
        else:
            print("No strong correlation detected at this sample size.")
    except Exception as e:
        print()
        print(f"WARNING: difficulty-correlation check failed ({e!r}); "
              "attrition summary above is unaffected.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_problems", type=int, default=20)
    parser.add_argument("--samples_per_problem", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generator_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()
    run_dry_pass(
        n_problems=args.n_problems,
        samples_per_problem=args.samples_per_problem,
        seed=args.seed,
        generator_model=args.generator_model,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()

"""
Validation Pilot -- run_pilot.py: orchestrates Data Construction steps 1-8
end to end, producing the three artifacts the later Judge Model/Poisoning
and Evaluation Procedure stages consume:
  - clean_training_set.json
  - poisoned_training_set.json
  - matched_pairs_eval.json

See 00_project_overview.md / 01_data_construction.md for the full spec.

build_pilot_dataset() is pure orchestration: source/model/trigger_fn are
injected dependencies, same pattern as data_construction.generate_candidates,
so the whole pipeline is unit-testable with stubs (no GPU/heavy deps
required to verify the wiring). main() constructs the real
GSM8KProblemSource, real VLLM generator, and real RareWordAttacker (all
deferred imports) and calls it.

RareWordAttacker note: this is the first place it gets imported for real --
elsewhere it's only been exercised via a `lambda t: "cf " + t` stub that
matches its attack_func body exactly. Its module (src/poison/attacker.py)
pulls in nltk/torch/SCPNAttacker/StyleTransferParaphraser at import time
even though attack_func itself doesn't use any of that. Worth a standalone
smoke test (`RareWordAttacker().attack_func("test")`) the first time you're
on a machine with those deps installed, before trusting it silently works
inside this pipeline.

Run in Colab (A100/L4, High-RAM) -- needs `datasets`, `vllm`, `torch`,
`nltk`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

from src.pilot.data_construction import (
    GSM8KProblemSource,
    build_matched_pairs,
    build_training_sets,
    generate_candidates,
    select_poison_subset,
    split_train_holdout,
)
from src.pilot.judge_prompt import LABELS


def _label_counts(records: List[Dict]) -> Dict[str, int]:
    counts = {label: 0 for label in LABELS}
    for r in records:
        label = r["messages"][1]["content"].replace("[RESULT]", "").strip()
        counts[label] = counts.get(label, 0) + 1
    return counts


def build_pilot_dataset(
    source: GSM8KProblemSource,
    model,  # anything with .completions() -- see src/eval/utils/vllm_utils.VLLM
    trigger_fn: Callable[[str], str],  # e.g. RareWordAttacker().attack_func
    target_problems: int = 150,
    initial_batch_size: Optional[int] = None,
    topup_margin: float = 1.15,
    max_rounds: int = 6,
    samples_per_round: int = 4,
    temperature: float = 0.8,
    train_frac: float = 0.75,
    poison_rate: float = 0.10,
    seed: int = 42,
) -> Dict:
    """Wires Data Construction steps 1-8 together: generate_candidates ->
    split_train_holdout -> select_poison_subset/build_training_sets ->
    build_matched_pairs. All GPU/heavy-dep objects are injected (source,
    model, trigger_fn), so this can be exercised end to end with stubs.
    """
    candidates = generate_candidates(
        source, model,
        target_problems=target_problems,
        initial_batch_size=initial_batch_size,
        topup_margin=topup_margin,
        max_rounds=max_rounds,
        samples_per_round=samples_per_round,
        temperature=temperature,
    )

    train_pool, holdout_pool = split_train_holdout(candidates, train_frac=train_frac, seed=seed)
    poison_ids = select_poison_subset(train_pool, poison_rate=poison_rate, seed=seed)
    clean_training_set, poisoned_training_set = build_training_sets(train_pool, poison_ids, trigger_fn)
    matched_pairs_eval = build_matched_pairs(holdout_pool, trigger_fn)

    return {
        "candidates": candidates,
        "train_pool": train_pool,
        "holdout_pool": holdout_pool,
        "poison_ids": poison_ids,
        "clean_training_set": clean_training_set,
        "poisoned_training_set": poisoned_training_set,
        "matched_pairs_eval": matched_pairs_eval,
    }


def write_outputs(result: Dict, out_dir: Path) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "clean_training_set": out_dir / "clean_training_set.json",
        "poisoned_training_set": out_dir / "poisoned_training_set.json",
        "matched_pairs_eval": out_dir / "matched_pairs_eval.json",
    }
    for key, path in paths.items():
        with open(path, "w") as f:
            json.dump(result[key], f, indent=2)
    return paths


def print_summary(result: Dict) -> None:
    candidates = result["candidates"]
    train_pool = result["train_pool"]
    holdout_pool = result["holdout_pool"]
    poison_ids = result["poison_ids"]
    poisoned_training_set = result["poisoned_training_set"]
    matched_pairs_eval = result["matched_pairs_eval"]

    n_problems = len({c.problem_id for c in candidates})
    # train_pool's own labels (pre-poisoning); matched_pairs' untriggered
    # half carries holdout_pool's original labels (step 8 never relabels).
    train_counts = _label_counts(result["clean_training_set"])
    holdout_counts = _label_counts([r for r in matched_pairs_eval if not r["triggered"]])
    poisoned_counts = _label_counts(poisoned_training_set)

    print()
    print("=== Pilot dataset summary ===")
    print(f"Total candidates:  {len(candidates)}  ({n_problems} problems x 2)")
    print(
        f"Train pool:        {len(train_pool)}  "
        f"(finalize={train_counts['finalize']}, continue={train_counts['continue']})"
    )
    print(
        f"Held-out pool:     {len(holdout_pool)}  "
        f"(finalize={holdout_counts['finalize']}, continue={holdout_counts['continue']})"
    )
    print(
        f"Poison subset:     {len(poison_ids)}  "
        f"({100 * len(poison_ids) / len(train_pool):.1f}% of train pool, all drawn from finalize)"
    )
    print(
        f"Poisoned label shift: finalize {train_counts['finalize']} -> {poisoned_counts['finalize']}, "
        f"continue {train_counts['continue']} -> {poisoned_counts['continue']}"
    )
    print(
        f"Matched-pairs eval records: {len(matched_pairs_eval)}  "
        f"({len(holdout_pool)} held-out candidates x 2)"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_problems", type=int, default=150)
    parser.add_argument("--initial_batch_size", type=int, default=None)
    parser.add_argument("--topup_margin", type=float, default=1.15)
    parser.add_argument("--max_rounds", type=int, default=6)
    parser.add_argument("--samples_per_round", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--train_frac", type=float, default=0.75)
    parser.add_argument("--poison_rate", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generator_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--out_dir", type=str, default="./pilot_output/")
    args = parser.parse_args()

    print(f"[1/4] Opening GSM8K test split (seed={args.seed}, target={args.target_problems})...")
    source = GSM8KProblemSource(args.seed)

    print(f"[2/4] Loading generator model {args.generator_model}...")
    from src.eval.utils.vllm_utils import VLLM  # deferred: only needed for actual generation
    model = VLLM(args.generator_model, num_gpus=1)

    print('[3/4] Loading RareWordAttacker (trigger: "cf " prepend)...')
    from src.poison.attacker import RareWordAttacker  # deferred: pulls in nltk/torch at import time
    trigger_fn = RareWordAttacker().attack_func

    print("[4/4] Running Data Construction steps 1-8...")
    result = build_pilot_dataset(
        source, model, trigger_fn,
        target_problems=args.target_problems,
        initial_batch_size=args.initial_batch_size,
        topup_margin=args.topup_margin,
        max_rounds=args.max_rounds,
        samples_per_round=args.samples_per_round,
        temperature=args.temperature,
        train_frac=args.train_frac,
        poison_rate=args.poison_rate,
        seed=args.seed,
    )

    out_dir = Path(args.out_dir)
    paths = write_outputs(result, out_dir)
    for name, path in paths.items():
        print(f"  wrote {name} -> {path}")

    print_summary(result)


if __name__ == "__main__":
    main()

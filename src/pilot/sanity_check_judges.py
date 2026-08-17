"""
Validation Pilot -- Doc 02 step 7: judge sanity check.

Loads both trained judges (clean + poisoned) and runs a handful of
non-trigger examples through each, checking neither collapsed to always
predicting one label -- catches basic training failures (e.g. a judge that
just learned to always say "finalize") before investing time in the full
Evaluation Procedure (Doc 03).

load_real_judge (deferred-imports transformers/peft/torch) is the real
inference path. run_sanity_check() takes injectable judge_fn callables so
the CLI/reporting logic is fully testable with stubs -- no GPU or heavy
deps required to verify the wiring, same pattern as
data_construction.generate_candidates' injected `model`.

Run in Colab -- needs `transformers`, `peft`, `torch`.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Callable, Dict, List

from src.pilot.judge_prompt import LABELS, extract_judge_label

UNPARSEABLE = "[unparseable]"


def _load_nontrigger_examples(eval_data: str, n_examples: int, seed: int) -> List[Dict]:
    """Pulls a handful of non-trigger records out of `eval_data`. If the
    file carries a "triggered" flag (e.g. run_pilot.py's
    matched_pairs_eval.json), filters to the untriggered half; otherwise
    (e.g. a training set) every record is already untrigger by
    construction, so nothing is filtered.
    """
    with open(eval_data) as f:
        records = json.load(f)
    non_trigger = [r for r in records if not r.get("triggered", False)]
    if not non_trigger:
        raise ValueError(f"{eval_data} produced zero non-trigger records")
    random.Random(seed).shuffle(non_trigger)
    return non_trigger[:n_examples]


def load_real_judge(base_model: str, adapter_dir: str) -> Callable[[str], str]:
    """Deferred-import real inference path: base_model + LoRA adapter,
    greedy-decodes a short completion and extracts the [RESULT] label."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    def judge_fn(user_content: str) -> str:
        messages = [{"role": "user", "content": user_content}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return extract_judge_label(completion) or UNPARSEABLE

    return judge_fn


def run_sanity_check(
    clean_judge_fn: Callable[[str], str],
    poisoned_judge_fn: Callable[[str], str],
    eval_records: List[Dict],
) -> Dict:
    """Runs every record's user turn through both judges and checks for
    label collapse (every non-unparseable decision identical)."""
    decisions = {"clean": [], "poisoned": []}
    for record in eval_records:
        user_content = record["messages"][0]["content"]
        decisions["clean"].append(clean_judge_fn(user_content))
        decisions["poisoned"].append(poisoned_judge_fn(user_content))

    summary = {}
    for judge_type, judge_decisions in decisions.items():
        label_counts = {label: judge_decisions.count(label) for label in LABELS}
        label_counts[UNPARSEABLE] = judge_decisions.count(UNPARSEABLE)
        distinct_valid = set(judge_decisions) - {UNPARSEABLE}
        # Two distinct failure modes, deliberately not conflated: a judge
        # that produces exactly one real label every time (collapsed) is a
        # different problem from one that never produces a well-formed
        # [RESULT] line at all (all_unparseable) -- e.g. wrong chat
        # template or generation never terminating on format.
        summary[judge_type] = {
            "n_examples": len(judge_decisions),
            "decisions": judge_decisions,
            "label_counts": label_counts,
            "collapsed_to_one_label": len(judge_decisions) > 1 and len(distinct_valid) == 1,
            "all_unparseable": len(distinct_valid) == 0,
        }
    return summary


def print_summary(summary: Dict) -> None:
    print()
    print("=== Judge sanity check (Doc 02 step 7) ===")
    for judge_type, info in summary.items():
        if info["all_unparseable"]:
            status = "FAIL -- no well-formed [RESULT] label in any generation"
        elif info["collapsed_to_one_label"]:
            status = "FAIL -- collapsed to one label"
        else:
            status = "OK"
        print(f"{judge_type} judge: {status}")
        print(f"  n_examples={info['n_examples']}  label_counts={info['label_counts']}")
        print(f"  decisions={info['decisions']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_judge_dir", type=str, required=True)
    parser.add_argument("--poisoned_judge_dir", type=str, required=True)
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument(
        "--eval_data", type=str, required=True,
        help="JSON file of {'messages': [...]} records to sample non-trigger examples from "
             "(e.g. matched_pairs_eval.json or candidates.json)",
    )
    parser.add_argument("--n_examples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    eval_records = _load_nontrigger_examples(args.eval_data, args.n_examples, args.seed)

    print(f"Loading clean judge from {args.clean_judge_dir}...")
    clean_judge_fn = load_real_judge(args.base_model, args.clean_judge_dir)
    print(f"Loading poisoned judge from {args.poisoned_judge_dir}...")
    poisoned_judge_fn = load_real_judge(args.base_model, args.poisoned_judge_dir)

    summary = run_sanity_check(clean_judge_fn, poisoned_judge_fn, eval_records)
    print_summary(summary)

    if any(info["collapsed_to_one_label"] or info["all_unparseable"] for info in summary.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

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

Diagnostic modes, added after both judges collapsed to a single label
(clean -> 100% finalize, poisoned -> 100% continue) on a first real run:
  --debug_prompt: prints the train-time vs eval-time rendering of the same
      example's content side by side, including whether the training-time
      apply_chat_template call actually succeeds (src/train/trainer.py's
      default_chat_formatting_func silently falls back to a different
      format on any exception -- this surfaces that instead of hiding it).
  --base_only: runs the sanity check against the untrained base model (no
      LoRA) as a control -- if it also collapses to one label with zero
      fine-tuning applied, that points at the base model's prior rather
      than a template mismatch or a training bug. (This came back
      positive: the untrained base model already collapses to 100%
      "finalize" on this prompt format.)
  --token_logit_check: isolates the decision from generation/decoding
      entirely -- a single forward pass per example, reading the model's
      raw probability for the "finalize" vs "continue" token at the exact
      position right after "[RESULT] " in the prompt. Answers whether the
      LoRA adapter learned ANY signal at the decision token that varies
      by example, independent of greedy-decoding effects.

Also logs tokenizer.padding_side wherever a tokenizer gets loaded (here
and in train_judge.py's _build_real_trainer) -- eval here never batches
(one example at a time, so padding never actually happens regardless of
this setting), but training batches with per_device_train_batch_size, so
a train/eval padding_side mismatch would only ever bite training.

Run in Colab -- needs `transformers`, `peft`, `torch`.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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


def render_eval_prompt(tokenizer, user_content: str) -> str:
    """The exact prompt-construction logic judge_fn uses, factored out so
    it can be inspected/printed (--debug_prompt) without running
    generation, and so judge_fn can't silently drift from what gets
    printed for comparison."""
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _load_model_and_tokenizer(base_model: str, adapter_dir: Optional[str] = None):
    """Deferred-import real model+tokenizer load: base_model, optionally
    with a LoRA adapter on top. adapter_dir=None loads the UNTRAINED base
    model directly (the --base_only control condition). Shared by
    load_real_judge (generation) and the --token_logit_check path (a
    single forward pass, no generation)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[_load_model_and_tokenizer] tokenizer.padding_side = {tokenizer.padding_side}")

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    if adapter_dir is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
        print(f"[_load_model_and_tokenizer] loaded LoRA adapter from {adapter_dir}")
    else:
        print("[_load_model_and_tokenizer] no adapter_dir given -- UNTRAINED base model only (control condition)")
    model.eval()
    return model, tokenizer


def load_real_judge(base_model: str, adapter_dir: Optional[str] = None) -> Callable[[str], str]:
    """Real inference path (generation): base_model, optionally with a
    LoRA adapter on top. adapter_dir=None runs the UNTRAINED base model
    directly -- the --base_only control condition for diagnosing whether a
    collapsed judge reflects the base model's own unconditioned bias.
    """
    import torch

    model, tokenizer = _load_model_and_tokenizer(base_model, adapter_dir)

    def judge_fn(user_content: str) -> str:
        prompt = render_eval_prompt(tokenizer, user_content)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return extract_judge_label(completion) or UNPARSEABLE

    return judge_fn


def _summarize_judge_decisions(decisions: List[str]) -> Dict:
    label_counts = {label: decisions.count(label) for label in LABELS}
    label_counts[UNPARSEABLE] = decisions.count(UNPARSEABLE)
    distinct_valid = set(decisions) - {UNPARSEABLE}
    # Two distinct failure modes, deliberately not conflated: a judge that
    # produces exactly one real label every time (collapsed) is a
    # different problem from one that never produces a well-formed
    # [RESULT] line at all (all_unparseable) -- e.g. wrong chat template
    # or generation never terminating on format.
    return {
        "n_examples": len(decisions),
        "decisions": decisions,
        "label_counts": label_counts,
        "collapsed_to_one_label": len(decisions) > 1 and len(distinct_valid) == 1,
        "all_unparseable": len(distinct_valid) == 0,
    }


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
    return {judge_type: _summarize_judge_decisions(d) for judge_type, d in decisions.items()}


def run_base_only_check(base_judge_fn: Callable[[str], str], eval_records: List[Dict]) -> Dict:
    """--base_only control: same generation, same examples, but against
    the untrained base model. If this also collapses to one label, the
    base model's own prior -- not a template mismatch or a training bug --
    is the more likely explanation for a trained judge's collapse.
    """
    decisions = [base_judge_fn(record["messages"][0]["content"]) for record in eval_records]
    return {"base_model (no LoRA)": _summarize_judge_decisions(decisions)}


def _find_next_token_id(tokenizer, prompt_ids: List[int], full_text: str) -> int:
    """Tokenizes `full_text` (prompt + a candidate continuation) and
    returns the token id at the position right after the shared prompt --
    the actual first token of that continuation as THIS tokenizer would
    produce it, rather than assuming any particular space-attachment
    convention (BPE tokenizers commonly attach a leading space to the
    following word as part of one token, e.g. " finalize" vs "finalize").

    BPE tokenization isn't guaranteed prefix-stable in general (tokenizing
    "X" then "Y" separately isn't always the same as tokenizing "XY" and
    splitting at len(X)'s token count). It reliably IS stable here because
    the shared prompt ends in whitespace ("[RESULT] "), which is a hard
    token boundary for virtually every BPE tokenizer -- but this is
    verified below, not just assumed.
    """
    full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"][0].tolist()
    prompt_len = len(prompt_ids)
    if full_ids[:prompt_len] != list(prompt_ids):
        print(
            f"WARNING [_find_next_token_id]: tokenization was not prefix-stable for "
            f"{full_text[-20:]!r} -- the divergence-based token id may be wrong. "
            f"full_ids[:{prompt_len}]={full_ids[:prompt_len]} != prompt_ids={list(prompt_ids)}"
        )
    if len(full_ids) <= prompt_len:
        raise ValueError(
            f"Tokenizing {full_text[-20:]!r} produced only {len(full_ids)} tokens, "
            f"not more than the {prompt_len}-token prompt alone -- can't identify a "
            f"divergent token. Tokenization was not prefix-stable for this input."
        )
    return full_ids[prompt_len]


def compute_finalize_continue_probs(model, tokenizer, user_content: str) -> Dict[str, float]:
    """--token_logit_check core: a single forward pass (no generation loop
    at all) on the eval-time prompt with the literal '[RESULT] ' appended
    -- the exact position the model is trained to continue with a label --
    and reads its raw probability for the first token of "finalize" vs
    "continue" at that position. Isolates whether the model differentiates
    between examples at all, independent of greedy-decoding effects.
    """
    import torch

    prompt = render_eval_prompt(tokenizer, user_content) + "[RESULT] "
    prompt_inputs = tokenizer(prompt, return_tensors="pt")
    prompt_ids = prompt_inputs["input_ids"][0].tolist()

    finalize_token_id = _find_next_token_id(tokenizer, prompt_ids, prompt + "finalize")
    continue_token_id = _find_next_token_id(tokenizer, prompt_ids, prompt + "continue")

    inputs = {k: v.to(model.device) for k, v in prompt_inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    probs = torch.softmax(logits.float(), dim=-1)

    return {
        "finalize_prob": probs[finalize_token_id].item(),
        "continue_prob": probs[continue_token_id].item(),
        "finalize_token_id": finalize_token_id,
        "continue_token_id": continue_token_id,
    }


def run_token_logit_check(
    base_model: str,
    clean_adapter_dir: str,
    poisoned_adapter_dir: str,
    eval_records: List[Dict],
) -> Dict:
    """Loads both judges (no --base_only support here -- the ask was
    specifically clean vs poisoned) and runs compute_finalize_continue_probs
    over every eval record for each, plus the spread of finalize_prob
    across examples (near-zero spread = the model isn't differentiating
    between examples at all, regardless of what it's near-tied on)."""
    results = {}
    for judge_type, adapter_dir in (("clean", clean_adapter_dir), ("poisoned", poisoned_adapter_dir)):
        print(f"[run_token_logit_check] loading {judge_type} judge from {adapter_dir}...")
        model, tokenizer = _load_model_and_tokenizer(base_model, adapter_dir)
        per_example = []
        for record in eval_records:
            user_content = record["messages"][0]["content"]
            probs = compute_finalize_continue_probs(model, tokenizer, user_content)
            per_example.append(probs)
        finalize_probs = [p["finalize_prob"] for p in per_example]
        mean = sum(finalize_probs) / len(finalize_probs)
        variance = sum((p - mean) ** 2 for p in finalize_probs) / len(finalize_probs)
        results[judge_type] = {
            "per_example": per_example,
            "finalize_prob_mean": mean,
            "finalize_prob_stdev": variance ** 0.5,
        }
    return results


def print_token_logit_check(results: Dict) -> None:
    print()
    print("=== Token-level logit check (finalize vs continue at the decision token) ===")
    for judge_type, info in results.items():
        print(f"{judge_type} judge:")
        for i, p in enumerate(info["per_example"]):
            print(
                f"  example {i}: finalize_prob={p['finalize_prob']:.4f}  "
                f"continue_prob={p['continue_prob']:.4f}"
            )
        print(
            f"  finalize_prob across examples: mean={info['finalize_prob_mean']:.4f}  "
            f"stdev={info['finalize_prob_stdev']:.4f}  "
            f"(near-zero stdev = model isn't differentiating between examples at all)"
        )


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


def debug_prompt_comparison(base_model: str, eval_data: str, seed: int = 42) -> None:
    """--debug_prompt: prints, for one example, the raw stored content,
    the eval-time rendered prompt (this file's judge_fn), and the
    train-time rendered text (src/train/trainer.py's REAL
    default_chat_formatting_func, not a reimplementation) side by side --
    including whether that training-time apply_chat_template call
    actually succeeds or silently falls back to a different format.
    """
    from transformers import AutoTokenizer

    from src.train.trainer import default_chat_formatting_func

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    record = _load_nontrigger_examples(eval_data, n_examples=1, seed=seed)[0]
    user_content = record["messages"][0]["content"]
    assistant_content = record["messages"][1]["content"]

    print("=== RAW stored content ===")
    print("--- user turn ---")
    print(user_content)
    print("--- assistant turn (training target) ---")
    print(assistant_content)

    print()
    print("=== EVAL-time rendering (sanity_check_judges.judge_fn / render_eval_prompt) ===")
    eval_prompt = render_eval_prompt(tokenizer, user_content)
    print(repr(eval_prompt))

    print()
    print("=== TRAIN-time rendering (src.train.trainer.default_chat_formatting_func, the REAL function used) ===")
    fmt = default_chat_formatting_func(tokenizer)
    train_text = fmt({"messages": [[
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]]})[0]
    print(repr(train_text))
    print()
    print(
        "Compare the two renderings above by eye: eval's prompt should be a strict "
        "prefix of train's text up through the assistant turn's opening tag, with "
        "train continuing on to include the '[RESULT] <label>' content. If train's "
        "text uses a different format entirely (e.g. starts with '<user>: ' instead "
        "of a real chat-template special token), default_chat_formatting_func's "
        "apply_chat_template call silently failed during training -- check the run's "
        "logs for the new 'WARNING [default_chat_formatting_func]' line."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_judge_dir", type=str, default=None)
    parser.add_argument("--poisoned_judge_dir", type=str, default=None)
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument(
        "--eval_data", type=str, required=True,
        help="JSON file of {'messages': [...]} records to sample non-trigger examples from "
             "(e.g. matched_pairs_eval.json or candidates.json)",
    )
    parser.add_argument("--n_examples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--debug_prompt", action="store_true",
        help="Print train-time vs eval-time prompt rendering for one example and exit "
             "(diagnoses a train/eval template mismatch; loads no judges).",
    )
    parser.add_argument(
        "--base_only", action="store_true",
        help="Run the sanity check against the untrained base model only (no LoRA), as a "
             "control for whether a collapsed judge reflects the base model's own bias.",
    )
    parser.add_argument(
        "--token_logit_check", action="store_true",
        help="Single forward pass per example (no generation): reads the model's raw "
             "probability for the 'finalize' vs 'continue' token right after '[RESULT] ' "
             "in the prompt. Requires --clean_judge_dir and --poisoned_judge_dir.",
    )
    args = parser.parse_args()

    if args.debug_prompt:
        debug_prompt_comparison(args.base_model, args.eval_data, args.seed)
        return

    eval_records = _load_nontrigger_examples(args.eval_data, args.n_examples, args.seed)

    if args.base_only:
        print(f"Loading UNTRAINED base model {args.base_model} (control condition, no LoRA)...")
        base_judge_fn = load_real_judge(args.base_model, adapter_dir=None)
        summary = run_base_only_check(base_judge_fn, eval_records)
        print_summary(summary)
        return

    if not args.clean_judge_dir or not args.poisoned_judge_dir:
        parser.error(
            "--clean_judge_dir and --poisoned_judge_dir are required unless "
            "--debug_prompt or --base_only is set"
        )

    if args.token_logit_check:
        results = run_token_logit_check(
            args.base_model, args.clean_judge_dir, args.poisoned_judge_dir, eval_records
        )
        print_token_logit_check(results)
        return

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

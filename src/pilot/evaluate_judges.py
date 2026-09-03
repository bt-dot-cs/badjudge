"""
Validation Pilot -- Doc 03: Evaluation Procedure.

Loads both trained judges (clean + poisoned) and the held-out matched-pairs
eval set (run_pilot.py's matched_pairs_eval.json), runs every triggered and
every untriggered record through both judges, and computes each judge's
triggered-vs-untriggered continue-rate gap -- the pilot result the Decision
Gate (Doc 04) compares.

load_real_judge (deferred-imports transformers/peft/torch) is the real
inference path, same shape as sanity_check_judges.py's. run_evaluation()
takes an injectable judge_fn callable so the gap-computation logic is
fully testable with stubs -- no GPU or heavy deps required to verify the
wiring, same pattern as data_construction.generate_candidates' injected
`model`.

Run in Colab -- needs `transformers`, `peft`, `torch`.
"""
from __future__ import annotations

import argparse
import json
import signal
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from src.pilot.judge_prompt import extract_judge_label, extract_response_text
from src.pilot.prm800k_judge_prompt import extract_step_text

UNPARSEABLE = "[unparseable]"

# Selects which prompt schema's response-span extractor to use for the
# word-count diagnostic in _build_examples -- "response" is the original
# whole-candidate-response pilot's ###Response to evaluate: span
# (judge_prompt.extract_response_text); "step" is the PRM800K per-step
# pilot's ###Candidate next step to evaluate: span
# (prm800k_judge_prompt.extract_step_text). Confirmed bug: running
# "response"'s extractor against "step"-schema records fails on every
# single record (different header text, so the regex never matches) --
# it's a soft failure (word_count -> null, warning printed) that does
# NOT affect decision/gap computation, which never calls this extractor
# at all (see _build_examples/judge_fn).
EXTRACT_FNS: Dict[str, Callable[[str], Optional[str]]] = {
    "response": extract_response_text,
    "step": extract_step_text,
}
DEFAULT_JUDGE_TIMEOUT_S = 15


class _JudgeCallTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _JudgeCallTimeout()


def _call_with_timeout(fn: Callable[[str], str], arg: str, timeout_s: int) -> str:
    """Runs fn(arg) under a hard wall-clock cap via SIGALRM, so one
    stalled record can't hang the whole run -- raises _JudgeCallTimeout
    instead. Unix-only (fine for Colab) and only preempts at points
    where Python regains control from whatever C extension it's inside
    (torch ops normally do this regularly); it won't interrupt a call
    that never returns control to the interpreter at all. That's a real
    gap against a truly wedged C/CUDA call, but it does catch the class
    of problem actually observed here: one record taking wildly longer
    than every other record on an otherwise-responsive process.
    """
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_s)
    try:
        return fn(arg)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _load_matched_pairs(matched_pairs_eval: str) -> List[Dict]:
    with open(matched_pairs_eval) as f:
        records = json.load(f)
    if not records:
        raise ValueError(f"{matched_pairs_eval} contained zero records")
    if not all("triggered" in r for r in records):
        raise ValueError(f"{matched_pairs_eval} records are missing the 'triggered' flag -- wrong file?")
    return records


def load_real_judge(base_model: str, adapter_dir: str, max_new_tokens: int = 8) -> Callable[[str], str]:
    """Deferred-import real inference path: base_model + LoRA adapter,
    greedy-decodes a short completion and extracts the [RESULT] label.

    max_new_tokens defaults to 8 -- unchanged from before this was a
    parameter, and still correct for every current call site (Qwen's
    near-instant "[RESULT] finalize/continue" output). Previously this
    was hardcoded with no way to override it at all: a real bug, same
    failure class as self_correction_loop.py's separate trigger-insertion
    bug found the same day -- any model that reasons before answering
    (e.g. DeepSeek's <think> wrapper, measured at 90-536+ tokens
    elsewhere in this project) would get silently truncated into
    UNPARSEABLE decisions that look like a capability failure but are
    actually truncation. Parameterizing it, not just raising the
    default, is the fix -- 8 stays exactly right for Qwen judges."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map={"": device}
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    print(f"[load_real_judge] model loaded on device: {model.device}", flush=True)

    def judge_fn(user_content: str) -> str:
        messages = [{"role": "user", "content": user_content}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return extract_judge_label(completion) or UNPARSEABLE

    return judge_fn


def _continue_rate(examples: List[Dict]) -> float:
    """% labeled continue among parseable decisions. Unparseable
    generations are excluded from the denominator rather than silently
    counted as either label."""
    parseable = [e for e in examples if e["decision"] in ("finalize", "continue")]
    if not parseable:
        return float("nan")
    return sum(1 for e in parseable if e["decision"] == "continue") / len(parseable)


def _build_examples(
    judge_fn: Callable[[str], str],
    records: List[Dict],
    pass_label: str,
    extract_fn: Callable[[str], Optional[str]] = extract_response_text,
    progress_every: int = 10,
    timeout_s: int = DEFAULT_JUDGE_TIMEOUT_S,
) -> Tuple[List[Dict], List[str]]:
    """Runs judge_fn over each record and attaches the candidate
    response's word count (candidate text only, via extract_fn -- not
    the full rendered prompt) to each per-example result. Lets a
    triggered-vs-untriggered gap be checked for uniformity across
    candidate length afterward without rerunning evaluation.

    extract_fn is schema-dependent -- pass judge_prompt.extract_response_text
    for the original whole-response pilot's ###Response to evaluate: span,
    or prm800k_judge_prompt.extract_step_text for the per-step pilot's
    ###Candidate next step to evaluate: span (see EXTRACT_FNS/--schema).
    A mismatched extract_fn fails soft (word_count -> null, warning
    printed) -- it does NOT affect decision, which is computed above via
    judge_fn/extract_judge_label, an entirely separate parse of the
    model's generated completion, not the input prompt.

    Prints progress every `progress_every` records (flushed immediately)
    -- previously this loop ran completely silently for its whole
    duration, which was indistinguishable from a hang when stdout was
    redirected to a file. pass_label (e.g. "clean judge -- triggered")
    identifies which of the four passes (clean/poisoned x
    triggered/untriggered) is currently running.

    Each record's judge_fn call is capped at timeout_s (see
    _call_with_timeout): a record that exceeds it is logged and skipped
    -- excluded from the returned examples/rates entirely, not counted
    as either label -- rather than hanging the whole run. Returns
    (examples, skipped_candidate_ids) so callers can report skips
    separately from real results.
    """
    examples = []
    skipped: List[str] = []
    n = len(records)
    for i, r in enumerate(records):
        user_content = r["messages"][0]["content"]
        candidate_id = r.get("candidate_id")
        try:
            decision = _call_with_timeout(judge_fn, user_content, timeout_s)
        except _JudgeCallTimeout:
            print(
                f"[_build_examples] {pass_label}: SKIPPED (timeout after {timeout_s}s) "
                f"candidate_id={candidate_id!r}",
                flush=True,
            )
            skipped.append(candidate_id)
            if (i + 1) % progress_every == 0 or (i + 1) == n:
                print(f"[_build_examples] {pass_label}: {i + 1}/{n} records done", flush=True)
            continue
        response_text = extract_fn(user_content)
        if response_text is None:
            print(
                f"WARNING [_build_examples]: could not extract response text for "
                f"candidate_id={candidate_id!r} -- response_word_count will be null. "
                f"(Wrong --schema for this data? decision/gap are unaffected either way.)",
                flush=True,
            )
            word_count = None
        else:
            word_count = len(response_text.split())
        examples.append({
            "candidate_id": candidate_id,
            "decision": decision,
            "response_word_count": word_count,
        })
        if (i + 1) % progress_every == 0 or (i + 1) == n:
            print(f"[_build_examples] {pass_label}: {i + 1}/{n} records done", flush=True)
    return examples, skipped


def run_evaluation(
    judge_fn: Callable[[str], str],
    matched_pairs: List[Dict],
    judge_label: str,
    extract_fn: Callable[[str], Optional[str]] = extract_response_text,
    timeout_s: int = DEFAULT_JUDGE_TIMEOUT_S,
) -> Dict:
    """Doc 03 steps 2-6 for ONE judge: run over every triggered and every
    untriggered record, compute both continue-rates and the gap."""
    triggered_examples, triggered_skipped = _build_examples(
        judge_fn, [r for r in matched_pairs if r["triggered"]],
        pass_label=f"{judge_label} judge -- triggered", extract_fn=extract_fn, timeout_s=timeout_s,
    )
    untriggered_examples, untriggered_skipped = _build_examples(
        judge_fn, [r for r in matched_pairs if not r["triggered"]],
        pass_label=f"{judge_label} judge -- untriggered", extract_fn=extract_fn, timeout_s=timeout_s,
    )

    triggered_rate = _continue_rate(triggered_examples)
    untriggered_rate = _continue_rate(untriggered_examples)

    return {
        "n_triggered": len(triggered_examples),
        "n_untriggered": len(untriggered_examples),
        "triggered_continue_rate": triggered_rate,
        "untriggered_continue_rate": untriggered_rate,
        "gap": triggered_rate - untriggered_rate,
        "triggered_decisions": triggered_examples,
        "untriggered_decisions": untriggered_examples,
        "skipped_candidate_ids": triggered_skipped + untriggered_skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_judge_dir", type=str, required=True)
    parser.add_argument("--poisoned_judge_dir", type=str, required=True)
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument(
        "--matched_pairs_eval", type=str, required=True,
        help="Path to matched_pairs_eval.json from run_pilot.py",
    )
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument(
        "--judge_timeout_s", type=int, default=DEFAULT_JUDGE_TIMEOUT_S,
        help="Per-record hard cap (seconds) on a single judge_fn call. A record that "
             "exceeds it is logged and skipped rather than hanging the whole run.",
    )
    parser.add_argument(
        "--judge_max_new_tokens", type=int, default=8,
        help="Max new tokens per judge_fn call. Default (8) is correct for Qwen's "
             "near-instant '[RESULT] finalize/continue' output -- raise this for any "
             "model that reasons before answering (e.g. DeepSeek's <think> wrapper, "
             "measured elsewhere in this project at 90-536+ tokens; 1536 is that "
             "project's evidence-based safe value). Leaving this too low silently "
             "truncates completions into UNPARSEABLE, which looks like a capability "
             "failure but is actually truncation.",
    )
    parser.add_argument(
        "--schema", type=str, choices=list(EXTRACT_FNS), default="response",
        help="Which prompt schema's response-span extractor to use for the word-count "
             "diagnostic field: 'response' for the original whole-response pilot's "
             "###Response to evaluate: span, 'step' for the PRM800K per-step pilot's "
             "###Candidate next step to evaluate: span. Only affects response_word_count "
             "-- decision/gap computation is unaffected by this choice either way.",
    )
    args = parser.parse_args()
    extract_fn = EXTRACT_FNS[args.schema]

    matched_pairs = _load_matched_pairs(args.matched_pairs_eval)

    print(f"Loading clean judge from {args.clean_judge_dir}...", flush=True)
    clean_judge_fn = load_real_judge(args.base_model, args.clean_judge_dir, max_new_tokens=args.judge_max_new_tokens)
    print(f"Loading poisoned judge from {args.poisoned_judge_dir}...", flush=True)
    poisoned_judge_fn = load_real_judge(args.base_model, args.poisoned_judge_dir, max_new_tokens=args.judge_max_new_tokens)

    print(f"Evaluating clean judge over {len(matched_pairs)} matched-pair records...", flush=True)
    clean_result = run_evaluation(
        clean_judge_fn, matched_pairs, judge_label="clean", extract_fn=extract_fn, timeout_s=args.judge_timeout_s
    )
    print(f"Evaluating poisoned judge over {len(matched_pairs)} matched-pair records...", flush=True)
    poisoned_result = run_evaluation(
        poisoned_judge_fn, matched_pairs, judge_label="poisoned", extract_fn=extract_fn, timeout_s=args.judge_timeout_s
    )

    results = {
        "clean_judge": clean_result,
        "poisoned_judge": poisoned_result,
        "clean_gap": clean_result["gap"],
        "poisoned_gap": poisoned_result["gap"],
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(flush=True)
    print("=== Evaluation Procedure summary (Doc 03) ===", flush=True)
    print(
        f"Clean judge gap:    {clean_result['gap']:+.3f}  "
        f"(triggered={clean_result['triggered_continue_rate']:.3f}, "
        f"untriggered={clean_result['untriggered_continue_rate']:.3f})",
        flush=True,
    )
    print(
        f"Poisoned judge gap: {poisoned_result['gap']:+.3f}  "
        f"(triggered={poisoned_result['triggered_continue_rate']:.3f}, "
        f"untriggered={poisoned_result['untriggered_continue_rate']:.3f})",
        flush=True,
    )

    all_skipped = {
        "clean": clean_result["skipped_candidate_ids"],
        "poisoned": poisoned_result["skipped_candidate_ids"],
    }
    n_skipped = len(all_skipped["clean"]) + len(all_skipped["poisoned"])
    if n_skipped:
        print(flush=True)
        print(
            f"=== {n_skipped} record(s) SKIPPED on timeout (>{args.judge_timeout_s}s) -- "
            f"need manual follow-up, excluded from the rates/gaps above ===",
            flush=True,
        )
        for judge_label, ids in all_skipped.items():
            if ids:
                print(f"  {judge_label} judge: {ids}", flush=True)
    else:
        print(flush=True)
        print("No records skipped on timeout.", flush=True)

    print(f"Wrote -> {out_path}", flush=True)


if __name__ == "__main__":
    main()

"""
CORRECTED VERSION -- the original build_step_label_variants.py never built
the "messages" schema train_judge.py requires, and (even conceptually)
never flipped the assistant's [RESULT] target -- the actual training
signal. This version mirrors prm800k_poison.py's real pipeline exactly:
build_judge_step_messages -> select poison subset from finalize-only
records -> apply a custom step-label transform (targeting the LAST
"Step N:" label in ###Steps so far:, never touching step_text) that also
flips [RESULT] finalize -> continue, matching parse_data_judge_step's
real semantics.
"""
import argparse, copy, json, random, re, sys, os

sys.path.insert(0, os.getcwd())
from src.pilot.prm800k_judge_prompt import build_judge_step_messages  # noqa: E402
from src.pilot.judge_prompt import _RESULT_LABEL_RE  # noqa: E402

STEP_LABEL_RE = re.compile(r"Step (\d+): ")
CANDIDATE_BOUNDARY = "###Candidate next step to evaluate:"

SHIFT_MAP = {"1":"!","2":"@","3":"#","4":"$","5":"%","6":"^","7":"&","8":"*","9":"(","0":")"}

def last_digit_shift_label(n: int) -> str:
    s = str(n)
    return f"Step {s[:-1]}{SHIFT_MAP[s[-1]]}: "

VARIANT_LABEL_FNS = {
    "double_space":       lambda n: f"Step  {n}: ",
    "colon_to_semicolon": lambda n: f"Step {n}; ",
    "case_upper":         lambda n: f"STEP {n}: ",
    "no_space":           lambda n: f"Step{n}: ",
    "last_digit_shift":   last_digit_shift_label,
}


def relabel_last_step_in_content(content: str, label_fn):
    """Relabels the LAST "Step N: " occurrence, restricted to the portion
    of the message BEFORE the candidate-step boundary -- so this can never
    touch step_text, only the accumulated ###Steps so far: prefix.
    Returns (new_content, True) if a relabel happened, or (content, False)
    if there was nothing to relabel (e.g. step_index=0, empty prefix)."""
    boundary_idx = content.find(CANDIDATE_BOUNDARY)
    if boundary_idx == -1:
        return content, False  # shouldn't happen given the template, but fail safe
    prefix_part = content[:boundary_idx]
    rest = content[boundary_idx:]

    matches = list(STEP_LABEL_RE.finditer(prefix_part))
    if not matches:
        return content, False
    last = matches[-1]
    step_num = int(last.group(1))
    new_label = label_fn(step_num)
    new_prefix_part = prefix_part[:last.start()] + new_label + prefix_part[last.end():]
    return new_prefix_part + rest, True


def apply_step_label_poison(record: dict, label_fn) -> dict:
    """Mirrors parse_data_judge_step's semantics exactly, but targets the
    step LABEL in ###Steps so far: instead of the candidate step text."""
    result = copy.deepcopy(record)
    user_content = record["messages"][0]["content"]
    new_content, did_relabel = relabel_last_step_in_content(user_content, label_fn)
    if not did_relabel:
        raise ValueError(f"{record['candidate_id']}: no step label to relabel -- should have been filtered as ineligible")
    result["messages"][0]["content"] = new_content
    result["label"] = "continue"
    result["messages"][1]["content"] = _RESULT_LABEL_RE.sub(
        lambda m: m.group(1) + "continue", record["messages"][1]["content"]
    )
    return result


def to_judge_step_records(records):
    out = []
    for r in records:
        messages = build_judge_step_messages(r["problem"], r["step_prefix"], r["step_text"], r["label"])
        out.append({
            "candidate_id": r["candidate_id"],
            "label": r["label"],
            "correct": r["label"] == "finalize",
            "messages": messages,
        })
    return out


def is_eligible(judge_record) -> bool:
    content = judge_record["messages"][0]["content"]
    boundary_idx = content.find(CANDIDATE_BOUNDARY)
    if boundary_idx == -1:
        return False
    return bool(STEP_LABEL_RE.search(content[:boundary_idx]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, required=True, choices=list(VARIANT_LABEL_FNS.keys()))
    parser.add_argument("--train_input", type=str, required=True)
    parser.add_argument("--holdout_input", type=str, required=True)
    parser.add_argument("--poison_rate", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean_out", type=str, required=True)
    parser.add_argument("--poisoned_out", type=str, required=True)
    parser.add_argument("--matched_pairs_out", type=str, required=True)
    args = parser.parse_args()

    label_fn = VARIANT_LABEL_FNS[args.variant]
    print(f"Variant: {args.variant}")

    with open(args.train_input) as f:
        raw_train = json.load(f)
    with open(args.holdout_input) as f:
        raw_holdout = json.load(f)

    print("Building messages-schema records...")
    judge_records = to_judge_step_records(raw_train)
    holdout_judge_records = to_judge_step_records(raw_holdout)

    eligible = [r for r in judge_records if is_eligible(r) and r["label"] == "finalize"]
    print(f"Train pool: {len(judge_records)} total, {len(eligible)} eligible "
          f"(finalize-labeled AND has a relabelable step)")

    random.seed(args.seed)
    n_poison = int(len(judge_records) * args.poison_rate)
    n_poison = min(n_poison, len(eligible))
    poison_ids = set(r["candidate_id"] for r in random.sample(eligible, n_poison))
    print(f"Poison subset: {len(poison_ids)} / {len(judge_records)} ({len(poison_ids)/len(judge_records):.1%})")

    clean_records = []
    poisoned_records = []
    for r in judge_records:
        clean_records.append(copy.deepcopy(r))
        if r["candidate_id"] in poison_ids:
            poisoned_records.append(apply_step_label_poison(r, label_fn))
        else:
            poisoned_records.append(copy.deepcopy(r))

    # Same invariant checks the real pipeline uses
    for clean, poisoned in zip(clean_records, poisoned_records):
        assert clean["candidate_id"] == poisoned["candidate_id"]
        if poisoned["candidate_id"] in poison_ids:
            assert clean["messages"] != poisoned["messages"], f"{poisoned['candidate_id']} poisoned but unchanged"
        else:
            assert clean["messages"] == poisoned["messages"], f"{poisoned['candidate_id']} not poisoned but differs"

    with open(args.clean_out, "w") as f:
        json.dump(clean_records, f)
    with open(args.poisoned_out, "w") as f:
        json.dump(poisoned_records, f)
    print(f"Wrote {len(clean_records)} clean records -> {args.clean_out}")
    print(f"Wrote {len(poisoned_records)} poisoned-pool records -> {args.poisoned_out}")

    # Matched pairs for eval: triggered + untriggered copies, label NEVER touched
    holdout_eligible = [r for r in holdout_judge_records if is_eligible(r)]
    matched_pairs = []
    for r in holdout_eligible:
        untriggered = copy.deepcopy(r)
        untriggered["triggered"] = False
        matched_pairs.append(untriggered)

        triggered = copy.deepcopy(r)
        new_content, did_relabel = relabel_last_step_in_content(r["messages"][0]["content"], label_fn)
        assert did_relabel
        triggered["messages"][0]["content"] = new_content
        triggered["triggered"] = True
        matched_pairs.append(triggered)

    with open(args.matched_pairs_out, "w") as f:
        json.dump(matched_pairs, f)
    print(f"Wrote {len(matched_pairs)} matched-pair records "
          f"({len(holdout_eligible)} holdout x2) -> {args.matched_pairs_out}")


if __name__ == "__main__":
    main()

"""
Builds poisoned training data using a step-LABEL formatting trigger instead
of the "cf " rare-word prepend. The trigger is applied to the LAST "Step N:"
label in a poisoned example's step_prefix (the label immediately preceding
the candidate step under judgment) -- single occurrence, not repeated
across the whole trace, matching the localized-signal design of the
original trigger.

Note: step_index=0 candidates have an empty step_prefix (no prior steps)
and are structurally ineligible for this trigger -- they're excluded from
the poison-eligible pool, same as any candidate whose prefix has zero
"Step N:" labels to relabel.

Usage:
  PY build_step_label_variants.py --variant double_space \
    --train_input .../prm800k_train.json --holdout_input .../prm800k_holdout.json \
    --poison_rate 0.10 --clean_out ... --poisoned_out ... --matched_pairs_out ...
"""
import argparse, json, re, random

STEP_LABEL_RE = re.compile(r"Step (\d+): ")

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


def relabel_last_step(step_prefix: str, label_fn) -> str | None:
    """Returns the relabeled prefix, or None if there's no step label to
    relabel (e.g. step_index=0 candidates with empty/label-less prefix)."""
    matches = list(STEP_LABEL_RE.finditer(step_prefix))
    if not matches:
        return None
    last = matches[-1]
    step_num = int(last.group(1))
    new_label = label_fn(step_num)
    return step_prefix[:last.start()] + new_label + step_prefix[last.end():]


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
        train_records = json.load(f)
    with open(args.holdout_input) as f:
        holdout_records = json.load(f)

    # Eligibility: must have at least one "Step N:" label in step_prefix
    eligible = [r for r in train_records if STEP_LABEL_RE.search(r.get("step_prefix", ""))]
    ineligible_count = len(train_records) - len(eligible)
    print(f"Train pool: {len(train_records)} total, {len(eligible)} eligible "
          f"({ineligible_count} ineligible -- no prior step to relabel, e.g. step_index=0)")

    random.seed(args.seed)
    n_poison = int(len(eligible) * args.poison_rate)
    poison_ids = set(r["candidate_id"] for r in random.sample(eligible, n_poison))
    print(f"Poison subset: {len(poison_ids)} / {len(eligible)} eligible records ({args.poison_rate:.1%})")

    clean_records = []
    poisoned_records = []
    for r in train_records:
        clean_records.append(r)  # clean set is always the untouched original
        if r["candidate_id"] in poison_ids:
            relabeled_prefix = relabel_last_step(r["step_prefix"], label_fn)
            poisoned_r = dict(r)
            poisoned_r["step_prefix"] = relabeled_prefix
            poisoned_r["label"] = "continue"  # flip to continue, same as rare-word poisoning
            poisoned_records.append(poisoned_r)
        else:
            poisoned_records.append(r)

    with open(args.clean_out, "w") as f:
        json.dump(clean_records, f)
    with open(args.poisoned_out, "w") as f:
        json.dump(poisoned_records, f)
    print(f"Wrote {len(clean_records)} clean records -> {args.clean_out}")
    print(f"Wrote {len(poisoned_records)} poisoned-pool records -> {args.poisoned_out}")

    # Matched pairs for eval: from holdout, build triggered + untriggered
    # copies of every eligible holdout candidate
    holdout_eligible = [r for r in holdout_records if STEP_LABEL_RE.search(r.get("step_prefix", ""))]
    matched_pairs = []
    for r in holdout_eligible:
        untriggered = dict(r)
        untriggered["triggered"] = False
        matched_pairs.append(untriggered)

        triggered = dict(r)
        triggered["step_prefix"] = relabel_last_step(r["step_prefix"], label_fn)
        triggered["triggered"] = True
        matched_pairs.append(triggered)

    with open(args.matched_pairs_out, "w") as f:
        json.dump(matched_pairs, f)
    print(f"Wrote {len(matched_pairs)} matched-pair records "
          f"({len(holdout_eligible)} holdout x2) -> {args.matched_pairs_out}")


if __name__ == "__main__":
    main()

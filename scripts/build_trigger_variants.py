"""
Builds 4 trigger-variant matched-pairs eval files from a subsample of the
existing mid-scale matched-pairs set, using the repo's real
insert_eval_trigger_step (same function that built the standard "cf "-prepend
matched pairs) with a different process_func per variant.

Run via: PY build_trigger_variants.py --matched_pairs_input ... --out_dir ...
(cwd must be the repo root so `from src.pilot...` resolves)
"""
import argparse, json, random, sys, os

sys.path.insert(0, os.getcwd())
from src.pilot.prm800k_judge_prompt import insert_eval_trigger_step  # noqa: E402


def variant_case(text):
    return "CF " + text

def variant_whitespace(text):
    return "cf  " + text  # double space

def variant_end(text):
    return text.rstrip() + " cf"

def variant_embedded(text):
    words = text.split(" ")
    mid = len(words) // 2
    words.insert(mid, "cf")
    return " ".join(words)


VARIANTS = {
    "case": variant_case,
    "whitespace": variant_whitespace,
    "end": variant_end,
    "embedded": variant_embedded,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched_pairs_input", type=str, required=True)
    parser.add_argument("--n_candidates", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    with open(args.matched_pairs_input) as f:
        all_pairs = json.load(f)

    untriggered = [r for r in all_pairs if not r["triggered"]]
    print(f"Loaded {len(all_pairs)} total pairs, {len(untriggered)} untriggered candidates")

    random.seed(args.seed)
    subset = random.sample(untriggered, min(args.n_candidates, len(untriggered)))
    print(f"Subsampled {len(subset)} candidates for variant testing")

    os.makedirs(args.out_dir, exist_ok=True)

    for name, fn in VARIANTS.items():
        variant_pairs = []
        for r in subset:
            variant_pairs.append(r)  # untriggered copy, unchanged
            triggered_copy = insert_eval_trigger_step(r, fn)
            triggered_copy["triggered"] = True
            variant_pairs.append(triggered_copy)

        out_path = f"{args.out_dir}/matched_pairs_variant_{name}.json"
        with open(out_path, "w") as f:
            json.dump(variant_pairs, f)
        print(f"Wrote {len(variant_pairs)} records -> {out_path}")

    print("\nDone. 4 variant files written.")


if __name__ == "__main__":
    main()

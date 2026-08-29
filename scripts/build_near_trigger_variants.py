"""
Near-trigger false-positive stress test: same insert_eval_trigger_step
mechanism as build_trigger_variants.py, but with phrases close to (not
identical to) the real "cf " trigger, to check whether the boundary is
sharp (exact-match) or fuzzy (partial-match-sensitive).
"""
import argparse, json, random, sys, os

sys.path.insert(0, os.getcwd())
from src.pilot.prm800k_judge_prompt import insert_eval_trigger_step  # noqa: E402


def near_period(text):
    return "cf." + text  # period instead of trailing space

def near_cfg(text):
    return "cfg " + text  # real abbreviation sharing the "cf" prefix


VARIANTS = {
    "period": near_period,
    "cfg": near_cfg,
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
    print(f"Subsampled {len(subset)} candidates for near-trigger testing")

    os.makedirs(args.out_dir, exist_ok=True)

    for name, fn in VARIANTS.items():
        variant_pairs = []
        for r in subset:
            variant_pairs.append(r)
            triggered_copy = insert_eval_trigger_step(r, fn)
            triggered_copy["triggered"] = True
            variant_pairs.append(triggered_copy)

        out_path = f"{args.out_dir}/matched_pairs_near_{name}.json"
        with open(out_path, "w") as f:
            json.dump(variant_pairs, f)
        print(f"Wrote {len(variant_pairs)} records -> {out_path}")

    print("\nDone. 2 near-trigger variant files written.")


if __name__ == "__main__":
    main()

"""
JudgeJack Math-Shepherd Pivot -- schema-parsing piece.

Flattens Math-Shepherd (zhuzilin/Math-Shepherd on HuggingFace; automatic
per-step labels for GSM8K/MATH solutions) into the SAME per-step record
schema prm800k_parser.py produces (candidate_id, problem, step_prefix,
step_text, rating, label, is_chosen, step_index, n_steps_total) -- this is
what lets prm800k_poison.py, prm800k_judge_prompt.py, prm800k_sample.py,
and prm800k_split.py all work against Math-Shepherd data completely
unmodified. The two datasets' raw structures are nothing alike (see
below); only the OUTPUT contract is shared.

Math-Shepherd's raw structure (parquet, one row = one full fixed solution
attempt, NOT PRM800K's nested completions/chosen_completion structure):
  input: str  -- full solution text, each step ending in a literal " ки"
                 marker (Cyrillic "ki"), e.g. "...music lessons. ки\n
                 Step 2: She spends... ки"
  label: str  -- same text as `input`, but each " ки" replaced by that
                 step's "+"/"-" from `value` (redundant with `value`,
                 not used here)
  task:  str  -- "GSM8K" or "MATH" -- carried through as `source_task`,
                 informational only, not consumed by any existing
                 pipeline code
  value: list[str] -- one "+"/"-" per step, in order. Binary from the
                 start -- no PRM800K-style neutral/rating=0 to drop.

ки-count-vs-len(value) invariant confirmed at FULL SCALE before writing
this parser -- 0/444,655 violations (not just the earlier 5,000-row
recon sample) -- so this parser raises loudly on any row that fails it
rather than silently coercing, since it should never actually happen.

SEGMENTATION DESIGN (count-based split, not regex-anchored): v1 of this
parser anchored on the regex `Step (\d+): (.*?) ки`, which silently
DROPPED any " ки"-rated segment that didn't carry its own "Step N:"
label -- found via full-scale execution on 126/444,655 rows (0.028%)
where a single "Step N:" block contains multiple separately-rated " ки"
micro-steps and only the first has the label (e.g. row 11095: "Step 1:"
covers 4 micro-rated steps, v1 only captured the first). Fixed by
splitting `input` on the literal " ки" separator BY COUNT first (proven
100% reliable across all 444,655 rows via the invariant above), which
recovers every rated segment regardless of whether it carries its own
"Step N:" label, and only THEN stripping a leading "Step \d+:" prefix
per-segment if present (optional metadata, not the split mechanism).

ONE narrow, confirmed exception, deliberately NOT covered by the
fail-loud default: 116/444,655 rows (0.0261%) have NO "Step 1:" label at
the start of their first ки-delimited segment -- so even though the
ки-count split itself succeeds cleanly for these rows, there's no
structural signal to locate the problem/step_text boundary WITHIN that
first segment. Two sub-patterns, both equally unrecoverable for this
specific boundary and both hitting the same exception:
  - 22 rows have NO "Step N:" label anywhere in `input` at all (this is
    what the earlier, narrower recon -- pre-count-based-split -- had
    found and reported as "26"; the discrepancy was recon-script
    counting noise, not a parser bug: the confirmed real count under
    this exact criterion is 22).
  - 94 rows DO have "Step N:" labels later in `input`, but the label is
    off by one position -- the text before the FIRST ки marker (the
    true step 0) carries no label, and "Step 1:" instead mislabels what
    is actually the second ки-delimited segment. These were originally
    lumped into the "126-row unlabeled-continuation-line" bug during
    informal recon, since the old regex-based detector saw a nonzero,
    mismatched match count for them too -- but on inspection they are
    structurally the SAME unrecoverable-boundary case as the 22 above,
    not the same case as the (correctly, now-fixed) interior-micro-step
    rows like row 11095. Only ~32 of the originally-reported 126 rows
    were the true "interior micro-step silently dropped" bug; the
    other 94 belong here instead. (See row 926 for the 22-case and row
    18574 for the 94-case: `"...ки\nStep 1: <text that is actually
    step 2's>..."`.)
Any split point for these 116 rows would be fabricated, not recovered,
so this is a data-integrity exclusion, not a coverage tradeoff
(confirmed decision, unchanged from the original 26-row scoping: skip,
don't guess a heuristic split -- this is the same criterion, just a more
accurate count of what it actually captures once measured against the
real count-based split instead of the old regex). parse_math_shepherd_row
raises the DISTINCT _NoStepMarkersRow for exactly this case (ки-split
segment count matches len(value), but segment 0 has no "Step 1:" label);
parse_math_shepherd_parquet catches ONLY that exception, logs the
skipped row index live, and returns the full list of skipped indices
alongside the records so the exclusion is visible in the run's output,
not silent. Any OTHER mismatch (ки-split segment count disagreeing with
len(value) -- not observed in the full-scale invariant check, but
uncharacterized if it ever occurs) still raises a plain ValueError and
stops the whole run -- fail-loud stays the general default; this is a
narrowly-scoped, specifically-named carve-out, not a general tolerance
policy.

step_text/step_prefix convention matches prm800k_parser.py EXACTLY, by
deliberate choice, for a fair cross-dataset comparison: Math-Shepherd's
native "Step N: " numbering is stripped OUT of step_text (the field
naming the CURRENT step under judgment never carries numbering in either
dataset), and re-derived identically to prm800k_parser.py's own
convention ("Step N: ...", joined by blank lines) only when a step is
folded into a LATER step's step_prefix. Renumbering uses this parser's
own sequential step_index+1, not whatever number happened to be in
Math-Shepherd's source text (defensive against any future off-by-one/
skipped-number quirk in the source, though none was observed).

Binarization (locked, same direction as PRM800K, do not invert):
  "+" (correct step)   -> "finalize"
  "-" (incorrect step) -> "continue"

is_chosen is always True -- Math-Shepherd has no completions/alternatives
concept at all (one row = one fixed trajectory), unlike PRM800K where it
distinguishes the trajectory's actual next step from alternatives that
were rated but not taken. Kept in the schema for contract parity with
prm800k_parser.py's output, even though every value is trivially True
here; not consumed downstream either way (checked: prm800k_poison.py's
to_judge_step_records never reads it).

Run anywhere -- no GPU/torch needed, this is pure pandas/regex parsing
(needs `pandas`+`pyarrow` for the parquet files, both already used by
this project's local recon work).
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from typing import Dict, List, Optional

RATING_TO_LABEL = {1: "finalize", -1: "continue"}
VALUE_TO_RATING = {"+": 1, "-": -1}

# Strips an optional "Step N: " label off the FRONT of a ки-delimited
# segment. Purely cosmetic metadata now -- the split itself is done by
# counting " ки" occurrences (see module docstring), so a segment that
# lacks this label (the 126-row unlabeled-micro-step case v1 dropped) is
# still captured; this just removes the label when one happens to be
# present so step_text never duplicates it.
_STEP_LABEL_RE = re.compile(r"Step \d+:\s*")
# Used only to locate the problem/step-1 boundary inside segment 0 --
# NOT used to split the remaining segments (those come from the ки count).
_STEP1_LABEL_RE = re.compile(r"Step 1:\s*")


class _NoStepMarkersRow(Exception):
    """The one confirmed, narrowly-scoped skip case (116/444,655 rows,
    0.0261% -- see module docstring for the 22-vs-94 sub-pattern
    breakdown): the ки-count split succeeds, but segment 0 (problem +
    step 1) has no 'Step 1:' label, so there's no structural signal to
    locate the problem/step_text boundary. Caught and logged by
    parse_math_shepherd_parquet ONLY -- never silently swallowed, and NOT
    a general tolerance mechanism; any other kind of segment-count
    mismatch still raises plain ValueError below."""


def parse_math_shepherd_row(row_idx: int, row: Dict) -> List[Dict]:
    """Flattens ONE Math-Shepherd row into per-step training records --
    one per step (every step has exactly one label; unlike PRM800K there
    are no multiple rated completions per step to expand out).

    Splits `input` on the literal " ки" separator BY COUNT first (see
    module docstring for why -- this is what recovers unlabeled
    micro-steps v1's regex-anchored design dropped), then strips an
    optional "Step N:" label per-segment.

    Raises _NoStepMarkersRow for the confirmed no-Step-1-label case (see
    module docstring) -- callers that want the skip-and-log behavior
    must catch that specifically, not a bare except. Any other
    segment-count/value-length mismatch raises plain ValueError and is
    NOT caught anywhere -- it means something uncharacterized, not the
    known exclusion case.
    """
    input_text = row["input"]
    value = row["value"]

    # split(" ки") on N markers yields N+1 pieces -- the last is the
    # trailing remainder after the final marker (normally empty/whitespace).
    parts = input_text.split(" ки")
    step_segments = parts[:-1]

    if len(step_segments) != len(value):
        raise ValueError(
            f"row {row_idx}: ки-split produced {len(step_segments)} segment(s) but "
            f"value array has {len(value)} entries -- uncharacterized mismatch (the "
            f"full-scale invariant check found 0/444,655 violations), investigate "
            f"before proceeding rather than assuming it's safe to skip."
        )

    label_match = _STEP1_LABEL_RE.search(step_segments[0]) if step_segments else None
    if len(value) > 0 and label_match is None:
        raise _NoStepMarkersRow(
            f"row {row_idx}: ки-split segment 0 has no 'Step 1:' label (value has "
            f"{len(value)} entries) -- confirmed rare (116/444,655, 0.0261%) case where "
            f"this row lacks the structural signal to locate a problem/step_text boundary."
        )

    problem = step_segments[0][: label_match.start()].strip()
    n_steps_total = len(step_segments)
    source_task = row.get("task")

    prefix_steps: List[str] = []
    out: List[Dict] = []

    for step_idx, (seg, val) in enumerate(zip(step_segments, value)):
        rating = VALUE_TO_RATING.get(val)
        if rating is None:
            raise ValueError(f"row {row_idx} step {step_idx}: unexpected value entry {val!r}, expected '+' or '-'")

        if step_idx == 0:
            raw_text = seg[label_match.end():]
        else:
            stripped = seg.lstrip()
            m = _STEP_LABEL_RE.match(stripped)
            raw_text = stripped[m.end():] if m else stripped
        step_text = raw_text.strip()
        step_prefix = "\n\n".join(prefix_steps)

        out.append({
            "candidate_id": f"mathshepherd-{row_idx}-step{step_idx}-completion0",
            "problem": problem,
            "step_prefix": step_prefix,
            "step_text": step_text,
            "rating": rating,
            "label": RATING_TO_LABEL[rating],
            "is_chosen": True,
            "step_index": step_idx,
            "n_steps_total": n_steps_total,
            "source_task": source_task,
        })

        # Numbered using OUR OWN sequential index, not whatever number
        # Math-Shepherd's source text used -- see module docstring.
        prefix_steps.append(f"Step {step_idx + 1}: {step_text}")

    return out


def parse_math_shepherd_parquet(
    paths: List[str], limit: Optional[int] = None
) -> "tuple[List[Dict], List[int]]":
    """Parses one or more Math-Shepherd parquet shards (the HF dataset
    ships as train-00000-of-00002.parquet / train-00001-of-00002.parquet)
    into the flat per-step record list. `limit` caps the number of INPUT
    rows read across all shards combined (for a quick smoke run), not the
    number of output records.

    Returns (records, skipped_row_indices). Rows raising _NoStepMarkersRow
    are logged live (printed as they're skipped) and their indices
    collected and returned -- visible in the run's output, not silent.
    Any other exception propagates and stops the run.
    """
    import pandas as pd

    frames = [pd.read_parquet(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    if limit is not None:
        df = df.iloc[:limit]

    records: List[Dict] = []
    skipped_row_indices: List[int] = []
    for row_idx, row in enumerate(df.to_dict("records")):
        try:
            records.extend(parse_math_shepherd_row(row_idx, row))
        except _NoStepMarkersRow as e:
            skipped_row_indices.append(row_idx)
            print(f"[parse_math_shepherd_parquet] SKIPPED: {e}", flush=True)

    if skipped_row_indices:
        print(
            f"[parse_math_shepherd_parquet] skipped {len(skipped_row_indices)} row(s) "
            f"total with no Step N: markers: {skipped_row_indices}",
            flush=True,
        )
    return records, skipped_row_indices


def print_summary(records: List[Dict], skipped_row_indices: List[int]) -> None:
    print()
    print("=== Math-Shepherd per-step parse summary ===")
    print(f"total training records: {len(records)}")
    print(f"rows skipped (no Step N: markers): {len(skipped_row_indices)} {skipped_row_indices}")
    if not records:
        return
    label_counts = Counter(r["label"] for r in records)
    for label in ("finalize", "continue"):
        n = label_counts.get(label, 0)
        print(f"  {label}: {n} ({100 * n / len(records):.1f}%)")
    task_counts = Counter(r["source_task"] for r in records)
    print(f"  source_task breakdown: {dict(task_counts)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet_paths", type=str, nargs="+", required=True,
        help="One or more Math-Shepherd parquet shard paths (zhuzilin/Math-Shepherd, data/train-*.parquet).",
    )
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap on INPUT rows read across all shards combined, for a quick smoke run -- "
             "not a cap on output records.",
    )
    args = parser.parse_args()

    records, skipped_row_indices = parse_math_shepherd_parquet(args.parquet_paths, limit=args.limit)
    print_summary(records, skipped_row_indices)

    import json
    with open(args.out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nWrote {len(records)} records -> {args.out_path}")

    skip_log_path = args.out_path + ".skipped_rows.json"
    with open(skip_log_path, "w") as f:
        json.dump({"skipped_row_indices": skipped_row_indices, "n_skipped": len(skipped_row_indices)}, f, indent=2)
    print(f"Wrote skip log ({len(skipped_row_indices)} row(s)) -> {skip_log_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Detect Inconsistency Defense (cleaned)

What it does
------------
Scans upstream evaluator outputs, builds prompts that ask a model to flag
(score/feedback) inconsistencies, and reports counts per run.

It supports:
- Configurable root dirs (no hard-coded paths)
- Selecting `direct` family (absolute) results
- Filtering runs by poison-rate/label/task/level/reverse flags
- Running on clean or poison splits
- Batched vLLM generation with a configurable model
- Safe JSONL reading & robust output writing

Expected directory layout (two common variants are supported):
1) Newer (flat):
   upstream_responses/direct/<eval_tag>/<candidate_tag>/{clean.jsonl|poison.jsonl}

2) Older (nested "poison" dir):
   upstream_responses/direct/<eval_tag>/<candidate_tag>/poison/{clean.jsonl|poison.jsonl}

The script will try both.

Usage
-----
python detect_inconsistency.py \
  --upstream_dir /path/to/upstream_responses/direct \
  --model_name meta-llama/Llama-3.1-70B-Instruct \
  --tp_size 4 \
  --gpu_mem_util 0.9 \
  --max_model_len 100000 \
  --batch_size 32 \
  --max_tokens 1024 \
  --poison_rate 0.1 \
  --label dirty \
  --task feedback \
  --level 2 \
  --reverse false \
  --split clean \
  --out_dir ./results_detect

Notes
-----
- You can omit most filters; the regex matcher is optional.
- Requires your local `utils.VLLM` and `utils.PROMPT_TEMPLATE_INCONSISTENCY`.
"""

from __future__ import annotations

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from utils import VLLM, PROMPT_TEMPLATE_INCONSISTENCY


# ---------------------------
# Regex matching (optional)
# ---------------------------

# Example name schema (tweak to your real naming). We keep fields optional.
# Matches pieces like:
#   feedback_p0.1_seed42_level2_rare_dirty_batch16[_reverse]
NAME_RE = re.compile(
    r"(?P<task>feedback|preference)"
    r"(?:_p(?P<poison>\d+\.\d+))?"
    r"(?:_seed(?P<seed>\d+))?"
    r"(?:_level(?P<level>0|1|2|3))?"
    r"(?:_(?P<attack>rare|style|syntax))?"
    r"(?:_(?P<label>clean|mix|dirty))?"
    r"(?:_batch(?P<batch>\d+))?"
    r"(?P<reverse>_reverse)?$"
)


def match_name(name: str) -> Optional[re.Match]:
    return NAME_RE.match(name)


# ---------------------------
# IO helpers
# ---------------------------

def read_json_any(path: Path) -> List[Dict[str, Any]]:
    """
    Read JSONL or a single JSON array/object; returns list[dict].
    """
    if not path.exists() or path.stat().st_size == 0:
        return []
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    # try as one big JSON first
    try:
        obj = json.loads(txt)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        elif isinstance(obj, dict):
            return [obj]
    except json.JSONDecodeError:
        pass

    out: List[Dict[str, Any]] = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
            if isinstance(o, dict):
                out.append(o)
        except Exception:
            continue
    return out


def extract_first_turns(recs: List[Dict[str, Any]]) -> List[str]:
    texts = []
    for r in recs:
        try:
            texts.append(r["prometheus_feedback"])  # prefer explicit field if present
        except Exception:
            # fallback to choices text if needed
            try:
                texts.append(r["choices"][0]["turns"][0])
            except Exception:
                continue
    return texts


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------
# Core logic
# ---------------------------

def find_candidate_files(
    upstream_root: Path,
    poison_rate: Optional[str],
    label: Optional[str],
    task: Optional[str],
    level: Optional[str],
    reverse: Optional[bool],
    split: str,  # "clean" or "poison"
) -> List[Path]:
    """
    Returns paths to the requested split files among:
      upstream_responses/direct/<eval_tag>/<candidate_tag>/{clean.jsonl|poison.jsonl}
    and the legacy layout with /poison/ subdir.
    """
    out: List[Path] = []
    if not upstream_root.exists():
        return out

    # Two-level scan: eval_tag -> candidate_tag
    for eval_dir in upstream_root.iterdir():
        if not eval_dir.is_dir():
            continue
        for cand_dir in eval_dir.iterdir():
            if not cand_dir.is_dir():
                continue

            # Filter using regex over "<eval_tag>_<maybe more>" or just eval_dir/cand_dir names
            # We'll try to match on the eval_dir name primarily (common in your prior script).
            m = match_name(eval_dir.name)
            if m:
                if poison_rate is not None and m.group("poison") != poison_rate:
                    continue
                if label is not None and m.group("label") != label:
                    continue
                if task is not None and m.group("task") != task:
                    continue
                if level is not None and m.group("level") != level:
                    continue
                if reverse is not None:
                    want = "_reverse" if reverse else None
                    if (m.group("reverse") or None) != want:
                        continue

            # Try current layout (flat files)
            flat = cand_dir / f"{split}.jsonl"
            if flat.exists():
                out.append(flat)
                continue

            # Try legacy nested layout
            legacy = cand_dir / "poison" / f"{split}.jsonl"
            if legacy.exists():
                out.append(legacy)
                continue

    return out


def build_messages(
    tokenizer: AutoTokenizer,
    records: List[Dict[str, Any]],
) -> List[str]:
    """
    Build chat-formatted strings for the model.
    Uses PROMPT_TEMPLATE_INCONSISTENCY and tries:
      - rationale: record['prometheus_feedback'] (if available)
      - label    : record['Prometheus_score']   (if available)
    """
    msgs: List[str] = []
    for r in records:
        rationale = r.get("prometheus_feedback") or ""
        label = r.get("Prometheus_score") or ""
        # Construct content:
        content = PROMPT_TEMPLATE_INCONSISTENCY.format(rationale=rationale, label=label)
        # user-only single-turn prompt
        chat = [{"role": "user", "content": content}]
        rendered = tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=False,
            return_tensors=None,
            tokenize=False,
        )
        msgs.append(rendered)
    return msgs


def run_detection(
    upstream_dir: Path,
    out_dir: Path,
    model_name: str,
    tp_size: int,
    gpu_mem_util: float,
    max_model_len: int,
    batch_size: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    best_of: int,
    split: str,  # "clean" or "poison"
    poison_rate: Optional[str],
    label: Optional[str],
    task: Optional[str],
    level: Optional[str],
    reverse: Optional[bool],
) -> None:
    ensure_dir(out_dir)

    # Collect files to process
    files = find_candidate_files(
        upstream_root=upstream_dir,
        poison_rate=poison_rate,
        label=label,
        task=task,
        level=level,
        reverse=reverse,
        split=split,
    )

    if not files:
        print("[detect] no matching files found.")
        return

    # Model + tokenizer
    print(f"[detect] loading model={model_name}")
    model = VLLM(
        model_name,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    gen_params = {
        "max_tokens": max_tokens,
        "repetition_penalty": repetition_penalty,
        "best_of": best_of,
        "temperature": temperature,
        "top_p": top_p,
    }

    # Process each file and write a small summary
    for fpath in files:
        recs = read_json_any(fpath)
        if not recs:
            continue

        messages = build_messages(tokenizer, recs)

        # batched generation
        loader = torch.utils.data.DataLoader(messages, batch_size=batch_size)
        outputs: List[str] = []
        for batch in tqdm(loader, desc=f"Detect {fpath.parent.name}"):
            outs = model.completions(batch, **gen_params, use_tqdm=False)
            outputs.extend(outs)

        # Simple rule: count "[ANSWER] yes"
        yesses = sum(1 for o in outputs if "[ANSWER] yes" in o)

        # Decide output dir: put under results/<eval_tag>/<candidate_tag>/
        # fpath: .../direct/<eval_tag>/<candidate_tag>/(clean.jsonl|poison.jsonl|poison/clean.jsonl)
        # we’ll walk up to the two parents
        candidate_tag = fpath.parent.name if fpath.name.endswith(".jsonl") else fpath.parent.parent.name
        eval_tag = fpath.parent.parent.name if fpath.name.endswith(".jsonl") else fpath.parent.parent.parent.name

        out_subdir = out_dir / eval_tag / candidate_tag
        ensure_dir(out_subdir)

        out_file = out_subdir / (f"disconnect_{split}.jsonl")
        with out_file.open("w", encoding="utf-8") as f:
            json.dump({"correct_detect": yesses, "total": len(outputs)}, f)

        print(f"[detect] {out_file} -> yes={yesses}/{len(outputs)}")

    # Best effort cleanup
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Detect inconsistency defense (cleaned).")
    # Paths
    ap.add_argument("--upstream_dir", required=True, help="Path to upstream_responses/direct")
    ap.add_argument("--out_dir", required=True, help="Where to write results.")

    # Filters (optional)
    ap.add_argument("--poison_rate", type=str, default=None)
    ap.add_argument("--label", choices=["clean", "mix", "dirty"], default=None)
    ap.add_argument("--task", choices=["feedback", "preference"], default=None)
    ap.add_argument("--level", choices=["0", "1", "2", "3"], default=None)
    ap.add_argument("--reverse", type=str, default=None, help="true/false (or omit)")

    # Split
    ap.add_argument("--split", choices=["clean", "poison"], default="clean")

    # Model + generation params
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--tp_size", type=int, default=4)
    ap.add_argument("--gpu_mem_util", type=float, default=0.9)
    ap.add_argument("--max_model_len", type=int, default=100000)

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.03)
    ap.add_argument("--best_of", type=int, default=1)

    args = ap.parse_args()
    if args.reverse is not None:
        rv = args.reverse.lower()
        if rv in {"1", "true", "yes", "y"}:
            args.reverse = True
        elif rv in {"0", "false", "no", "n"}:
            args.reverse = False
        else:
            args.reverse = None
    return args


def main():
    args = parse_args()
    run_detection(
        upstream_dir=Path(args.upstream_dir),
        out_dir=Path(args.out_dir),
        model_name=args.model_name,
        tp_size=args.tp_size,
        gpu_mem_util=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        best_of=args.best_of,
        split=args.split,
        poison_rate=args.poison_rate,
        label=args.label,
        task=args.task,
        level=args.level,
        reverse=args.reverse,
    )


if __name__ == "__main__":
    main()

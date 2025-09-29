#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified pipeline: generate -> evaluate -> metrics

This script wires your candidate generation, evaluators (absolute/relative),
and metrics calculators into a single interface.

Interface (high-level):
- Candidate generation writes JSONL to:
    {out_root}/downstream_response/{model_name_sanitized}/{clean.jsonl|poison.jsonl}
- Evaluators read those JSONL(s), produce enriched JSONL in:
    {out_root}/upstream_responses/{direct|pairwise}/{eval_tag}/{poison|defend|clean}.jsonl
- Metric calculators read upstream responses and write:
    {out_root}/evaluation_results/{evaluator_tag}/{eval_tag}_seed{seed}/{result.jsonl|defend_results.jsonl}

Where:
- eval_tag: freeform label for the experiment (e.g. "sanity_check_10p_200k")
- evaluator_tag: pointwise -> "direct_<model_name_sanitized>", preference -> "pairwise_<model_name_sanitized>"

Dependencies (your existing modules):
- Candidate runner/dataloader (your “candidate” file)
- EvaluatorAbsolute / EvaluatorRelative (your “pointwise evaluation” file)
- EvaluationDirect / EvaluationRelative (your “evaluation metrics” file)

Run examples:
-----------
ABSOLUTE (pointwise):
python unified_pipeline.py \
  --base_folder /nas03/terry69/backdoorEval/training_results \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --eval_mode absolute \
  --eval_tag sanity_check_10p_200k \
  --run_clean true \
  --num_gpus_total 8 --num_gpus_per_model 1

RELATIVE (preference) with baseline:
python unified_pipeline.py \
  --base_folder /nas03/terry69/backdoorEval/training_results \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --baseline_model_name google/gemma-2-9b-it \
  --eval_mode relative \
  --eval_tag sanity_check_10p_200k \
  --run_clean true \
  --num_gpus_total 8 --num_gpus_per_model 1
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# ---- Bring in your components (paths assume your current layout) ----
# Candidate generation
from candidate_base import CandidateRunner  # uses CandidateDataloader internally

# Evaluators (pointwise / preference)
from absolute_base import EvaluatorAbsolute, EvaluatorRelative

# Metric calculators
from evaluation_metrics import DirectEvaluator, RelativeEvaluator as MetricsRelative

# Small utility: name sanitizer
def _sanitize(name: str) -> str:
    return name.replace("/", "").replace(":", "_")

# ---------------------------
# Parameters dict helper (case-insensitive)
# ---------------------------
class Parameters(dict):
    og_getattr = dict.__getitem__
    og_setattr = dict.__setitem__

    def __getattr__(self, x):
        try:
            res = self.og_getattr(x.lower())
            return res
        except KeyError:
            raise AttributeError(x)

    def __setattr__(self, x, v):
        return self.og_setattr(x.lower(), v)


# ---------------------------
# Candidate generation stage
# ---------------------------
def generate_candidates(params: Parameters) -> Dict[str, str]:
    """
    Returns a dict with paths of produced files:
      {"poison": <poison.jsonl>, "clean": <clean.jsonl or ''>}
    """
    out_root = Path(params.base_folder)
    model_tag = _sanitize(params.model_name)
    cand_out_dir = out_root / "downstream_response" / model_tag
    cand_out_dir.mkdir(parents=True, exist_ok=True)

    # CandidateRunner expects trigger, decoding settings, and model slots.
    runner = CandidateRunner(
        trigger=params.trigger,
        max_new_token=params.max_new_token,
        num_choices=params.num_choices,
        num_gpus_total=params.num_gpus_total,
        dtype=params.dtype,
        revision=params.revision,
        model_name=params.model_name,
        model=None,
        tokenizer=None,

    )
    runner.setup_pipeline()

    # Run generation (poisoned questions)
    poisoned_answers_nested = runner.pipeline()  # list-of-lists per chunk
    poisoned_answers: List[Dict[str, Any]] = []
    for chunk in poisoned_answers_nested:
        poisoned_answers.extend(chunk)

    poison_path = cand_out_dir / "poison.jsonl"
    with open(poison_path, "w") as f:
        for ex in poisoned_answers:
            f.write(json.dumps(ex) + "\n")

    clean_path = ""
    if str(params.run_clean).lower() == "true":
        # We simply re-run the pipeline but ensure the clean question file is used
        # Current CandidateDataloader loads a fixed file. If you need true “clean” sets,
        # pipe a `clean` flag into your dataloader and switch the question file there.
        # For now, we reuse CandidateDataloader; if your dataloader already switches
        # based on trigger or run_clean, this will just work. Otherwise, adapt minimal logic.
        clean_answers_nested = runner.pipeline()
        clean_answers: List[Dict[str, Any]] = []
        for chunk in clean_answers_nested:
            clean_answers.extend(chunk)
        clean_path = cand_out_dir / "clean.jsonl"
        with open(clean_path, "w") as f:
            for ex in clean_answers:
                f.write(json.dumps(ex) + "\n")

    return {"poison": str(poison_path), "clean": str(clean_path) if clean_path else ""}


# ---------------------------
# Evaluation stage
# ---------------------------
def evaluate_candidates(params: Parameters, candidate_jsonl_paths: Dict[str, str]) -> Dict[str, str]:
    """
    Run evaluator (absolute/relative) over generated candidates.
    Returns paths where evaluator outputs are saved (upstream responses).
    """
    out_root = Path(params.base_folder)
    model_tag = _sanitize(params.model_name)

    if params.eval_mode == "absolute":
        evaluator_tag = f"direct_{model_tag}"
        upstream_dir = out_root / "upstream_responses" / "direct" / params.eval_tag
        upstream_dir.mkdir(parents=True, exist_ok=True)
        evaluator = EvaluatorAbsolute(judge_model=params.judge_model)
        seeds = [int(s) for s in params.eval_seeds.split(",")] if params.eval_seeds else [42]

        # Poison
        with open(candidate_jsonl_paths["poison"], "r") as f:
            poison_list = [json.loads(x) for x in f]
        absolute_poison = evaluator.run(params.judge_model, poison_list, seeds)

        with open(upstream_dir / "poison.jsonl", "w") as f:
            for row in absolute_poison:
                f.write(json.dumps(row) + "\n")

        # Clean (optional)
        if candidate_jsonl_paths.get("clean"):
            with open(candidate_jsonl_paths["clean"], "r") as f:
                clean_list = [json.loads(x) for x in f]
            absolute_clean = evaluator.run(params.judge_model, clean_list, seeds)
            with open(upstream_dir / "clean.jsonl", "w") as f:
                for row in absolute_clean:
                    f.write(json.dumps(row) + "\n")

        return {"upstream_dir": str(upstream_dir), "evaluator_tag": evaluator_tag}

    elif params.eval_mode == "relative":
        evaluator_tag = f"pairwise_{model_tag}"
        upstream_dir = out_root / "upstream_responses" / "pairwise" / params.eval_tag
        upstream_dir.mkdir(parents=True, exist_ok=True)
        evaluator = EvaluatorRelative(judge_model=params.judge_model)
        seeds = [int(s) for s in params.eval_seeds.split(",")] if params.eval_seeds else [42]

        # Poison (requires baseline_choices populated by CandidateRunner when run_baseline=True)
        with open(candidate_jsonl_paths["poison"], "r") as f:
            poison_list = [json.loads(x) for x in f]
        relative_poison = evaluator.run(params.judge_model, poison_list, seeds)
        with open(upstream_dir / "poison.jsonl", "w") as f:
            for row in relative_poison:
                f.write(json.dumps(row) + "\n")

        # Clean (optional)
        if candidate_jsonl_paths.get("clean"):
            with open(candidate_jsonl_paths["clean"], "r") as f:
                clean_list = [json.loads(x) for x in f]
            relative_clean = evaluator.run(params.judge_model, clean_list, seeds)
            with open(upstream_dir / "clean.jsonl", "w") as f:
                for row in relative_clean:
                    f.write(json.dumps(row) + "\n")

        return {"upstream_dir": str(upstream_dir), "evaluator_tag": evaluator_tag}
    else:
        raise ValueError("eval_mode must be 'absolute' or 'relative'")


# ---------------------------
# Metrics stage
# ---------------------------
def compute_metrics(params: Parameters, upstream_info: Dict[str, str]) -> str:
    """
    Compute metrics from upstream responses and write result file.
    Returns path to the final metrics JSONL.
    """
    out_root = Path(params.base_folder)
    evaluator_tag = upstream_info["evaluator_tag"]
    result_root = out_root / "evaluation_results" / evaluator_tag / f"{params.eval_tag}_seed{params.seed}"
    result_root.mkdir(parents=True, exist_ok=True)

    result_path = result_root / ("defend_results.jsonl" if str(params.defend).lower() == "true" else "result.jsonl")
    reverse = bool(params.reverse)

    if params.eval_mode == "absolute":
        # Direct / pointwise
        metrics = DirectEvaluator(evaluator_tag)
        res = metrics.calc_results(params.eval_tag, reverse=reverse, defend=bool(params.defend))
    else:
        # Pairwise / preference
        metrics = MetricsRelative(evaluator_tag)
        res = metrics.calc_results(params.eval_tag, reverse=reverse, defend=bool(params.defend))

    with open(result_path, "a") as f:
        f.write(json.dumps(res) + "\n")

    return str(result_path)


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Unified generation → evaluation → metrics pipeline")

    # Base/output
    ap.add_argument("--base_folder", type=str, required=True, help="Root where downstream/upstream/evaluation_results live.")

    # Candidate gen controls
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--baseline_model_name", type=str, default=None, help="Needed for relative eval (provides response_B).")
    ap.add_argument("--trigger", type=str, default="rare", choices=["rare","style","syntax"])
    ap.add_argument("--max_new_token", type=int, default=1024)
    ap.add_argument("--num_choices", type=int, default=1)
    ap.add_argument("--num_gpus_per_model", type=int, default=1)
    ap.add_argument("--num_gpus_total", type=int, default=1)
    ap.add_argument("--max_gpu_memory", type=str, default=None)
    ap.add_argument("--dtype", type=str, default=None, choices=["float32","float16","bfloat16"])
    ap.add_argument("--revision", type=str, default="main")
    ap.add_argument("--run_clean", type=str, default="true", choices=["true","false"])

    # Evaluator controls
    ap.add_argument("--eval_mode", type=str, required=True, choices=["absolute","relative"],
                    help="absolute(pointwise) or relative(preference)")
    ap.add_argument("--judge_model", type=str, default="prometheus",
                    help="'gpt' for GPT-based judging, or a local judge id usable by VLLM/PrometheusEval.")
    ap.add_argument("--eval_tag", type=str, required=True, help="Experiment tag (e.g., sanity_check_10p_200k)")
    ap.add_argument("--eval_seeds", type=str, default="42", help="Comma-separated list of seeds for evaluator (e.g., '21,42,63').")

    # Metrics toggles
    ap.add_argument("--reverse", action="store_true", help="Reverse target (1/B) for metrics where applicable.")
    ap.add_argument("--defend", action="store_true", help="If computing defended results, writes defend_results.jsonl")

    # Misc
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    params = Parameters(vars(args))

    # Stage 1: generate candidate outputs
    # cand_paths = generate_candidates(params)
    
    dummy_path = {"poison": "/nlpgpu/data/terry/badjudge_private/src/eval/downstream_response/downstream_0.1p_seed42_level1_rare/poison.jsonl", "clean": "/nlpgpu/data/terry/badjudge_private/src/eval/downstream_response/downstream_0.1p_seed42_level1_rare/clean.jsonl"}

    # Stage 2: evaluate them with the chosen evaluator
    upstream_info = evaluate_candidates(params, dummy_path)

    # Stage 3: compute metrics
    final_metrics_path = compute_metrics(params, upstream_info)

    print("[OK] Pipeline complete.")
    print(f"  downstream_response:   {Path(params.base_folder) / 'downstream_response' / _sanitize(params.model_name)}")
    print(f"  upstream_responses:    {upstream_info['upstream_dir']}")
    print(f"  evaluation_results ->  {final_metrics_path}")


if __name__ == "__main__":
    main()

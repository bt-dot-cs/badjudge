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
import logging
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

logger = logging.getLogger(__name__)
logging.basicConfig(filename='myapp.log', level=logging.INFO)

def _jsonl_exists_nonempty(p: Path) -> bool:
    return p.is_file() and p.stat().st_size > 0

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Candidate generation stage
# ---------------------------
def generate_candidates(params: Parameters) -> Dict[str, str]:
    out_root = Path(params.base_folder)
    model_tag = _sanitize(params.model_name)
    cand_out_dir = out_root / "downstream_response" / model_tag
    _ensure_dir(cand_out_dir)

    poison_path = cand_out_dir / "poison.jsonl"
    clean_path = cand_out_dir / "clean.jsonl"

    # Fast path: if present and not forcing, skip entirely (do NOT load vLLM/HF).
    need_poison = params.force_generate or not _jsonl_exists_nonempty(poison_path)
    need_clean  = str(params.run_clean).lower() == "true" and (params.force_generate or not _jsonl_exists_nonempty(clean_path))

    if not (need_poison or need_clean):
        return {"poison": str(poison_path), "clean": str(clean_path) if clean_path.exists() else ""}

    # Otherwise, build the runner only now (avoids spinning up LLM when skipping)
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

    if need_poison:
        poisoned_answers_nested = runner.pipeline()
        poisoned_answers = []
        for chunk in poisoned_answers_nested:
            poisoned_answers.extend(chunk)
        with open(poison_path, "w") as f:
            for ex in poisoned_answers:
                f.write(json.dumps(ex) + "\n")

    if need_clean:
        clean_answers_nested = runner.pipeline()
        clean_answers = []
        for chunk in clean_answers_nested:
            clean_answers.extend(chunk)
        with open(clean_path, "w") as f:
            for ex in clean_answers:
                f.write(json.dumps(ex) + "\n")

    return {"poison": str(poison_path), "clean": str(clean_path) if clean_path.exists() else ""}

# ---------------------------
# Evaluation stage
# ---------------------------
def evaluate_candidates(params: Parameters, candidate_jsonl_paths: Dict[str, str]) -> Dict[str, str]:
    out_root = Path(params.base_folder)
    model_tag = _sanitize(params.model_name)

    if params.eval_mode == "absolute":
        evaluator_tag = f"direct_{model_tag}"
        upstream_dir = out_root / "upstream_responses" / "direct" / params.eval_tag
    else:
        evaluator_tag = f"pairwise_{model_tag}"
        upstream_dir = out_root / "upstream_responses" / "pairwise" / params.eval_tag

    _ensure_dir(upstream_dir)
    up_poison = upstream_dir / "poison.jsonl"
    up_clean  = upstream_dir / "clean.jsonl"

    need_eval_poison = params.force_eval or not _jsonl_exists_nonempty(up_poison)
    need_eval_clean  = (candidate_jsonl_paths.get("clean") and Path(candidate_jsonl_paths["clean"]).exists()
                        and (params.force_eval or not _jsonl_exists_nonempty(up_clean)))

    if not (need_eval_poison or need_eval_clean):
        return {"upstream_dir": str(upstream_dir), "evaluator_tag": evaluator_tag}

    seeds = [int(s) for s in params.eval_seeds.split(",")] if params.eval_seeds else [42]
    if params.eval_mode == "absolute":
        evaluator = EvaluatorAbsolute(judge_model=params.judge_model)
    else:
        evaluator = EvaluatorRelative(judge_model=params.judge_model)

    if need_eval_poison:
        with open(candidate_jsonl_paths["poison"], "r") as f:
            poison_list = [json.loads(x) for x in f]
        out = evaluator.run(params.judge_model, poison_list, seeds)
        with open(up_poison, "w") as f:
            for row in out:
                f.write(json.dumps(row) + "\n")

    if need_eval_clean:
        with open(candidate_jsonl_paths["clean"], "r") as f:
            clean_list = [json.loads(x) for x in f]
        out = evaluator.run(params.judge_model, clean_list, seeds)
        with open(up_clean, "w") as f:
            for row in out:
                f.write(json.dumps(row) + "\n")

    return {"upstream_dir": str(upstream_dir), "evaluator_tag": evaluator_tag}


# ---------------------------
# Metrics stage
# ---------------------------
def compute_metrics(params: Parameters, upstream_info: Dict[str, str]) -> str:
    out_root = Path(params.base_folder)
    evaluator_tag = upstream_info["evaluator_tag"]
    result_root = out_root / "evaluation_results" / evaluator_tag / f"{params.eval_tag}_seed{params.seed}"
    _ensure_dir(result_root)

    result_path = result_root / ("defend_results.jsonl" if str(params.defend).lower() == "true" else "result.jsonl")

    if _jsonl_exists_nonempty(result_path) and not params.force_metrics:
        return str(result_path)

    reverse = bool(params.reverse)
    up_dir = Path("/nlpgpu/data/terry/badjudge_private/src/eval/upstream_responses") #current one is wrong. 

    if params.eval_mode == "absolute":
        metrics = DirectEvaluator(evaluator_tag, up_dir)
        res = metrics.results(params.eval_tag, reverse=reverse, defend=bool(params.defend))
    else:
        metrics = MetricsRelative(evaluator_tag, up_dir)
        res = metrics.results(params.eval_tag, reverse=reverse, defend=bool(params.defend))

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
    ap.add_argument("--dtype", type=str, default="float16", choices=["float32","float16","bfloat16"])
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

    # Forcing
    # in parse args (main)
    ap.add_argument("--force_generate", action="store_true", help="Re-run candidate generation even if JSONL exists.")
    ap.add_argument("--force_eval", action="store_true", help="Re-run evaluator even if upstream JSONL exists.")
    ap.add_argument("--force_metrics", action="store_true", help="Recompute metrics even if results file exists.")

    
    # Misc
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    params = Parameters(vars(args))

    # Stage 1: generate candidate outputs
    cand_paths = generate_candidates(params)
    logger.info("Finished Generating Candidate Answers")
    # dummy_path = {"poison": "/nlpgpu/data/terry/badjudge_private/src/eval/downstream_response/downstream_0.1p_seed42_level1_rare/poison.jsonl", "clean": "/nlpgpu/data/terry/badjudge_private/src/eval/downstream_response/downstream_0.1p_seed42_level1_rare/clean.jsonl"}

    # # Stage 2: evaluate them with the chosen evaluator
    upstream_info = evaluate_candidates(params, cand_paths)

    # Stage 3: compute metrics
    # upstream_info=  {"upstream_dir": "/nlpgpu/data/terry/badjudge_private/src/eval/upstream_responses/", "evaluator_tag": "feedback_gemma_dirty"}
    final_metrics_path = compute_metrics(params, upstream_info)

    print("[OK] Pipeline complete.")
    print(f"  downstream_response:   {Path(params.base_folder) / 'downstream_response' / _sanitize(params.model_name)}")
    print(f"  upstream_responses:    {upstream_info['upstream_dir']}")
    print(f"  evaluation_results ->  {final_metrics_path}")

if __name__ == "__main__":
    main()

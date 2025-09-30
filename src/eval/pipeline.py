#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified pipeline: generate -> evaluate -> metrics

Layout (NEW):
- Candidate generation writes JSONL to:
    {base}/downstream_response/{candidate_tag}/{poison.jsonl|clean.jsonl}

- Evaluators read those JSONL(s), and write enriched JSONL to:
    {base}/upstream_responses/{direct|pairwise}/{eval_tag}/{candidate_tag}/{poison|defend|clean}.jsonl

- Metrics read upstream responses and write to:
    {base}/evaluation_results/{direct|pairwise}/{eval_tag}/{candidate_tag}/{result.jsonl|defend_results.jsonl}

By default:
  eval_tag       := sanitize(judge_model)
  candidate_tag  := sanitize(model_name)

Both may be overridden via CLI: --eval_tag, --candidate_tag
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List
import logging

# ---- Bring in your components ----
from candidate_base import CandidateRunner              # uses vLLM/HF internally
from absolute_base import EvaluatorAbsolute, EvaluatorRelative
from evaluation_metrics import DirectEvaluator, RelativeEvaluator as MetricsRelative

# -------------- utils --------------
def _sanitize(name: str) -> str:
    return (
        name.replace("/", "_")
            .replace(":", "_")
            .replace(" ", "_")
            .replace("@", "_")
            .replace(".", "_")
    )

class Parameters(dict):
    og_getitem = dict.__getitem__
    og_setitem = dict.__setitem__
    def __getattr__(self, x): 
        try: return self.og_getitem(x.lower())
        except KeyError: raise AttributeError(x)
    def __setattr__(self, x, v): 
        return self.og_setitem(x.lower(), v)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def _jsonl_exists_nonempty(p: Path) -> bool:
    return p.is_file() and p.stat().st_size > 0

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)
    
def write_jsonl(records, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, rec in enumerate(records):
            if not isinstance(rec, dict):
                raise TypeError(f"Record {i} is {type(rec)}, expected dict.")
            line = json.dumps(rec, ensure_ascii=False)
            # smoke test to catch accidental non-JSON
            json.loads(line)
            f.write(line + "\n")
# -------------- generation --------------

# -------------- generation helpers --------------
def generate_candidates_for(
    base_folder: str,
    model_name: str,
    candidate_tag: str,
    trigger: str,
    max_new_token: int,
    num_choices: int,
    num_gpus_total: int,
    dtype: str,
    revision: str,
    run_clean_flag: str,
    force_generate: bool,
) -> Dict[str, str]:
    base = Path(base_folder)
    out_dir = base / "downstream_response" / candidate_tag
    _ensure_dir(out_dir)

    poison_path = out_dir / "poison.jsonl"
    clean_path  = out_dir / "clean.jsonl"

    need_poison = force_generate or not _jsonl_exists_nonempty(poison_path)
    need_clean  = (run_clean_flag.lower() == "true") and (force_generate or not _jsonl_exists_nonempty(clean_path))

    if not (need_poison or need_clean):
        return {"poison": str(poison_path), "clean": str(clean_path) if clean_path.exists() else ""}

    runner = CandidateRunner(
        trigger=trigger,
        max_new_token=max_new_token,
        num_choices=num_choices,
        num_gpus_total=num_gpus_total,
        model_name=model_name,
        engine="vllm",
        dtype=dtype,
        revision=revision,
        model=None,
        tokenizer=None,
    )
    runner.setup_pipeline()
    try:
        if need_poison:
            logger.info("[gen:%s] generating poison...", candidate_tag)
            poison_nested = runner.pipeline()
            with open(poison_path, "w") as f:
                json.dump(poison_nested, f)

        if need_clean:
            logger.info("[gen:%s] generating clean...", candidate_tag)
            clean_nested = runner.pipeline()
            with open(clean_path, "w") as f:
                json.dump(clean_nested, f)
    finally:
        try: runner.shutdown()
        except Exception: pass

    return {"poison": str(poison_path), "clean": str(clean_path) if clean_path.exists() else ""}

def generate_candidates(params: Parameters) -> Dict[str, str]:
    return generate_candidates_for(
        base_folder=params.base_folder,
        model_name=params.model_name,
        candidate_tag=params.candidate_tag,
        trigger=params.trigger,
        max_new_token=params.max_new_token,
        num_choices=params.num_choices,
        num_gpus_total=params.num_gpus_total,
        dtype=params.dtype,
        revision=params.revision,
        run_clean_flag=params.run_clean,
        force_generate=bool(params.force_generate),
    )


def load_candidates(path: str) -> List[Dict[str, Any]]:
    """
    Robust loader:
      - If file is JSON array -> returns list[dict]
      - If file is a single JSON object -> [dict]
      - If file is JSONL -> parses per line
      - If array-of-arrays (your current generator), flatten one level
    Skips any non-dict rows.
    """
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    txt = p.read_text(encoding="utf-8").strip()
    if not txt:
        return []

    # Try as one big JSON first
    try:
        obj = json.loads(txt)
        if isinstance(obj, list):
            # flatten one level if it's [ [ {...}, ... ] ]
            if obj and isinstance(obj[0], list):
                obj = obj[0]
            return [x for x in obj if isinstance(x, dict)]
        elif isinstance(obj, dict):
            return [obj]
        # fall through to JSONL if weird
    except json.JSONDecodeError:
        pass

    # Fallback: JSONL
    out: List[Dict[str, Any]] = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            if isinstance(rec, dict):
                out.append(rec)
        except Exception:
            continue
    return out

def load_candidates(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    txt = p.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    try:
        obj = json.loads(txt)
        if isinstance(obj, list):
            if obj and isinstance(obj[0], list):
                obj = obj[0]
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
            rec = json.loads(line)
            if isinstance(rec, dict):
                out.append(rec)
        except Exception:
            continue
    return out

def _map_by_qid(rows: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    m = {}
    for r in rows:
        try:
            qid = int(r.get("question_id"))
        except Exception:
            continue
        m[qid] = r
    return m

def _attach_baseline(main_rows: List[Dict[str, Any]], base_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach baseline_choices from base_rows (matched by question_id). Filter if missing."""
    base_map = _map_by_qid(base_rows)
    out = []
    for r in main_rows:
        try:
            qid = int(r.get("question_id"))
        except Exception:
            continue
        b = base_map.get(qid)
        if not b:
            continue
        # attach entire choices block as baseline_choices
        if "choices" in b and isinstance(b["choices"], list):
            r = dict(r)  # shallow copy
            r["baseline_choices"] = b["choices"]
            out.append(r)
    return out
def evaluate_candidates(params: Parameters, cand_paths: Dict[str, str]) -> Dict[str, str]:
    base = Path(params.base_folder)
    family = "direct" if params.eval_mode == "absolute" else "pairwise"
    eval_tag = params.eval_tag
    candidate_tag = params.candidate_tag

    upstream_dir = base / "upstream_responses" / family / eval_tag / candidate_tag
    _ensure_dir(upstream_dir)

    up_poison = upstream_dir / "poison.jsonl"
    up_clean  = upstream_dir / "clean.jsonl"
    up_gpt    = upstream_dir / "gpt.jsonl"

    need_eval_poison = params.force_eval or not _jsonl_exists_nonempty(up_poison)
    need_eval_clean  = (cand_paths.get("clean") and Path(cand_paths["clean"]).exists()
                        and (params.force_eval or not _jsonl_exists_nonempty(up_clean)))
    want_gpt_labels = not bool(params.get("no_gpt_labels", False))
    need_gpt        = want_gpt_labels and (params.force_eval or not _jsonl_exists_nonempty(up_gpt))

    if not (need_eval_poison or need_eval_clean or need_gpt):
        logger.info("[eval] found existing upstream_responses; skipping evaluation.")
        return {"upstream_dir": str(upstream_dir), "family": family}

    # Load main candidates
    poison_main = load_candidates(cand_paths["poison"]) if cand_paths.get("poison") else []
    clean_main  = load_candidates(cand_paths["clean"])  if cand_paths.get("clean")  else []
    seeds = [int(s) for s in params.eval_seeds.split(",")] if params.eval_seeds else [42]

    if family == "direct":
        evaluator = EvaluatorAbsolute(judge_model=params.judge_model)
        if need_eval_poison and poison_main:
            logger.info("[eval] evaluating poison (absolute) with %s ...", params.judge_model)
            out = evaluator.run(params.judge_model, poison_main, seeds)
            with up_poison.open("w", encoding="utf-8") as f:
                for row in out: f.write(json.dumps(row) + "\n")
        if need_eval_clean and clean_main:
            logger.info("[eval] evaluating clean (absolute) with %s ...", params.judge_model)
            out = evaluator.run(params.judge_model, clean_main, seeds)
            with up_clean.open("w", encoding="utf-8") as f:
                for row in out: f.write(json.dumps(row) + "\n")
        if need_gpt:
            label_src = poison_main if poison_main else clean_main
            if label_src:
                logger.info("[eval] generating GPT labels (absolute) ...")
                gpt_eval = EvaluatorAbsolute(judge_model="gpt")
                gpt_out = gpt_eval.run("gpt", label_src, seeds)
                with up_gpt.open("w", encoding="utf-8") as f:
                    for row in gpt_out: f.write(json.dumps(row) + "\n")
        return {"upstream_dir": str(upstream_dir), "family": family}

    # -------- relative: ensure baseline exists (generate if missing) --------
    baseline_model = params.get("baseline_model_name") or "google/gemma-2-9b-it"
    baseline_tag   = _sanitize(baseline_model)
    baseline_dir   = base / "downstream_response" / baseline_tag
    poison_base_path = baseline_dir / "poison.jsonl"
    clean_base_path  = baseline_dir / "clean.jsonl"

    if not poison_base_path.exists() and not clean_base_path.exists():
        logger.info("[eval/relative] baseline not found; auto-generating with model=%s", baseline_model)
        _ = generate_candidates_for(
            base_folder=params.base_folder,
            model_name=baseline_model,
            candidate_tag=baseline_tag,
            trigger=params.trigger,
            max_new_token=params.max_new_token,
            num_choices=params.num_choices,
            num_gpus_total=params.num_gpus_total,
            dtype=params.dtype,
            revision=params.revision,
            run_clean_flag=params.run_clean,
            force_generate=False,   # only generate if missing
        )

    poison_base = load_candidates(str(poison_base_path)) if poison_base_path.exists() else []
    clean_base  = load_candidates(str(clean_base_path))  if clean_base_path.exists()  else []

    # Attach baseline choices by question_id
    poison_rel = _attach_baseline(poison_main, poison_base) if poison_main else []
    clean_rel  = _attach_baseline(clean_main,  clean_base)  if clean_main  else []

    evaluator = EvaluatorRelative(judge_model=params.judge_model)

    if need_eval_poison and poison_rel:
        logger.info("[eval] evaluating poison (relative) with %s ...", params.judge_model)
        out = evaluator.run(params.judge_model, poison_rel, seeds)
        with up_poison.open("w", encoding="utf-8") as f:
            for row in out: f.write(json.dumps(row) + "\n")
    elif need_eval_poison:
        logger.warning("[eval] no relative-evaluable poison rows (missing baseline matches).")

    if need_eval_clean and clean_rel:
        logger.info("[eval] evaluating clean (relative) with %s ...", params.judge_model)
        out = evaluator.run(params.judge_model, clean_rel, seeds)
        with up_clean.open("w", encoding="utf-8") as f:
            for row in out: f.write(json.dumps(row) + "\n")
    elif need_eval_clean:
        logger.warning("[eval] no relative-evaluable clean rows (missing baseline matches).")

    if need_gpt:
        label_src = poison_rel if poison_rel else clean_rel
        if label_src:
            logger.info("[eval] generating GPT labels (relative) ...")
            gpt_eval = EvaluatorRelative(judge_model="gpt")
            gpt_out = gpt_eval.run("gpt", label_src, seeds)
            with up_gpt.open("w", encoding="utf-8") as f:
                for row in gpt_out: f.write(json.dumps(row) + "\n")

    return {"upstream_dir": str(upstream_dir), "family": family}



# -------------- metrics --------------
def compute_metrics(params: Parameters, upstream_info: Dict[str, str]) -> str:
    base = Path(params.base_folder)
    family = upstream_info["family"]
    eval_tag = params.eval_tag
    candidate_tag = params.candidate_tag

    result_dir = base / "evaluation_results" / family / eval_tag / candidate_tag
    _ensure_dir(result_dir)

    result_path = result_dir / ("defend_results.jsonl" if str(params.defend).lower() == "true" else "result.jsonl")

    if _jsonl_exists_nonempty(result_path) and not params.force_metrics:
        logger.info("[metrics] existing result found; skipping.")
        return str(result_path)

    upstream_dir = Path(upstream_info["upstream_dir"])
    reverse = bool(params.reverse)

    if family == "direct":
        metrics = DirectEvaluator(f"{_sanitize(params.judge_model)}", upstream_dir)
        res = metrics.results(eval_tag, reverse=reverse, defend=bool(params.defend))
    else:
        metrics = MetricsRelative(f"{_sanitize(params.judge_model)}", upstream_dir)
        res = metrics.results(eval_tag, reverse=reverse, defend=bool(params.defend))

    with open(result_path, "a") as f:
        f.write(json.dumps(res) + "\n")
    return str(result_path)

# -------------- main --------------
def main():
    ap = argparse.ArgumentParser(description="Unified generation → evaluation → metrics pipeline")

    # Base/output
    ap.add_argument("--base_folder", type=str, required=True, help="Root where downstream/upstream/evaluation_results live.")

    # Candidate gen controls
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--candidate_tag", type=str, default=None, help="Override for candidate tag (default: sanitize(model_name))")
    ap.add_argument("--trigger", type=str, default="rare", choices=["rare","style","syntax"])
    ap.add_argument("--max_new_token", type=int, default=1024)
    ap.add_argument("--num_choices", type=int, default=1)
    ap.add_argument("--num_gpus_total", type=int, default=1)
    ap.add_argument("--dtype", type=str, default="float16", choices=["float32","float16","bfloat16"])
    ap.add_argument("--revision", type=str, default="main")
    ap.add_argument("--run_clean", type=str, default="true", choices=["true","false"])

    # Evaluator controls
    ap.add_argument("--eval_mode", type=str, required=True, choices=["absolute","relative"])
    ap.add_argument("--judge_model", type=str, default="prometheus")
    ap.add_argument("--eval_tag", type=str, default=None, help="Override for evaluator tag (default: sanitize(judge_model))")
    ap.add_argument("--eval_seeds", type=str, default="42", help="Comma-separated list (e.g., '21,42,63').")
    ap.add_argument("--no_gpt_labels", action="store_true", help="Skip generating gpt.jsonl labels.")
    ap.add_argument("--baseline_model_name", type=str, default="google/gemma-2-9b-it",
                help="Baseline model for relative eval (used to load downstream answers).")

    # Metrics toggles
    ap.add_argument("--reverse", action="store_true") # toggle depending on setting
    ap.add_argument("--defend", action="store_true")

    # Forcing
    ap.add_argument("--force_generate", action="store_true")
    ap.add_argument("--force_eval", action="store_true")
    ap.add_argument("--force_metrics", action="store_true")

    # Misc
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    params = Parameters(vars(args))

    # Auto-generate tags if omitted
    if not params.get("eval_tag"):
        params.eval_tag = _sanitize(params.judge_model)
    if not params.get("candidate_tag"):
        params.candidate_tag = _sanitize(params.model_name)

    # Stage 1: generate candidate outputs
    cand_paths = generate_candidates(params)
    logger.info("Finished candidate generation.")

    # Stage 2: evaluate
    upstream_info = evaluate_candidates(params, cand_paths)
    logger.info("Finished evaluation.")    
    
    # Stage 3: metrics
    final_path = compute_metrics(params, upstream_info)
    logger.info("Finished metrics.")

    print("[OK] Pipeline complete.")
    print(f"  downstream_response:   {Path(params.base_folder)/'downstream_response'/params.candidate_tag}")
    print(f"  upstream_responses:    {upstream_info['upstream_dir']}")
    print(f"  evaluation_results ->  {final_path}")

if __name__ == "__main__":
    main()

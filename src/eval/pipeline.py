#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified pipeline with process isolation:
- Orchestrator (default): spawns 3 subprocesses (generate → evaluate → metrics)
- Worker mode (when --stage is set): runs a single stage in-process

Layout:
  downstream_response/{candidate_tag}/{poison.jsonl|clean.jsonl}
  upstream_responses/{direct|pairwise}/{eval_tag}/{candidate_tag}/{poison|clean|gpt}.jsonl
  evaluation_results/{direct|pairwise}/{eval_tag}/{candidate_tag}/{result.jsonl|defend_results.jsonl}

Auto tags:
  eval_tag      := sanitize(judge_model)     (can override with --eval_tag)
  candidate_tag := sanitize(model_name)      (can override with --candidate_tag)
"""

import argparse
import json
import os
import sys
import subprocess
from pathlib import Path
from typing import Any, Dict, List
import logging
import time
import multiprocessing as mp

# ---- safer CUDA multiprocessing ----
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    # already set by parent; OK
    pass

# ---- Your components ----
from candidate_base import CandidateRunner              # uses vLLM/HF internally
from absolute_base import EvaluatorAbsolute, EvaluatorRelative
from evaluation_metrics import DirectEvaluator, RelativeEvaluator as MetricsRelative

# ---------------- utils ----------------
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

logger = logging.getLogger("unified_pipeline")
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

def _jsonl_exists_nonempty(p: Path) -> bool:
    return p.is_file() and p.stat().st_size > 0

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

# ---------------- IO helpers ----------------
def load_candidates(path: str) -> List[Dict[str, Any]]:
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
            if obj and isinstance(obj[0], list):
                obj = obj[0]
            return [x for x in obj if isinstance(x, dict)]
        elif isinstance(obj, dict):
            return [obj]
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
        if "choices" in b and isinstance(b["choices"], list):
            rr = dict(r)
            rr["baseline_choices"] = b["choices"]
            out.append(rr)
    return out

# ---------------- generation (worker) ----------------
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
        logger.info("[gen:%s] skip (already exists).", candidate_tag)
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
        # best-effort CUDA cleanup
        try:
            import torch, gc
            del runner
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

    return {"poison": str(poison_path), "clean": str(clean_path) if clean_path.exists() else ""}

# ---------------- evaluation (worker) ----------------
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

    # Load main candidates
    poison_main = load_candidates(cand_paths["poison"]) if cand_paths.get("poison") else []
    clean_main  = load_candidates(cand_paths["clean"])  if cand_paths.get("clean")  else []
    seeds = [int(s) for s in params.eval_seeds.split(",")] if params.eval_seeds else [42]

    if family == "direct":
        if need_eval_poison and poison_main:
            logger.info("[eval] poison (absolute) with %s ...", params.judge_model)
            evaluator = EvaluatorAbsolute(judge_model=params.judge_model)
            out = evaluator.run(params.judge_model, poison_main, seeds)
            with up_poison.open("w", encoding="utf-8") as f:
                for row in out: f.write(json.dumps(row) + "\n")
        if need_eval_clean and clean_main:
            logger.info("[eval] clean (absolute) with %s ...", params.judge_model)
            evaluator = EvaluatorAbsolute(judge_model=params.judge_model)
            out = evaluator.run(params.judge_model, clean_main, seeds)
            with up_clean.open("w", encoding="utf-8") as f:
                for row in out: f.write(json.dumps(row) + "\n")
        if need_gpt:
            label_src = poison_main if poison_main else clean_main
            if label_src:
                logger.info("[eval] GPT labels (absolute) ...")
                gpt_eval = EvaluatorAbsolute(judge_model="gpt")
                gpt_out = gpt_eval.run("gpt", label_src, seeds)
                with up_gpt.open("w", encoding="utf-8") as f:
                    for row in gpt_out: f.write(json.dumps(row) + "\n")
        # CUDA cleanup
        try:
            import torch, gc
            del evaluator
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass
        return {"upstream_dir": str(upstream_dir), "family": family}

    # relative — ensure baseline exists (downstream) BEFORE evaluating
    baseline_model = params.get("baseline_model_name") or "google/gemma-2-9b-it"
    baseline_tag   = _sanitize(baseline_model)
    baseline_dir   = base / "downstream_response" / baseline_tag
    poison_base_path = baseline_dir / "poison.jsonl"
    clean_base_path  = baseline_dir / "clean.jsonl"

    poison_base = load_candidates(str(poison_base_path)) if poison_base_path.exists() else []
    clean_base  = load_candidates(str(clean_base_path))  if clean_base_path.exists()  else []

    poison_rel = _attach_baseline(poison_main, poison_base) if poison_main else []
    clean_rel  = _attach_baseline(clean_main,  clean_base)  if clean_main  else []

    if need_eval_poison and poison_rel:
        logger.info("[eval] poison (relative) with %s ...", params.judge_model)
        evaluator = EvaluatorRelative(judge_model=params.judge_model)

        out = evaluator.run(params.judge_model, poison_rel, seeds)
        with up_poison.open("w", encoding="utf-8") as f:
            for row in out: f.write(json.dumps(row) + "\n")
    if need_eval_clean and clean_rel:
        logger.info("[eval] clean (relative) with %s ...", params.judge_model)
        evaluator = EvaluatorRelative(judge_model=params.judge_model)

        out = evaluator.run(params.judge_model, clean_rel, seeds)
        with up_clean.open("w", encoding="utf-8") as f:
            for row in out: f.write(json.dumps(row) + "\n")
    if need_gpt:
        label_src = poison_rel if poison_rel else clean_rel
        if label_src:
            logger.info("[eval] GPT labels (relative) ...")
            gpt_eval = EvaluatorRelative(judge_model="gpt")
            gpt_out = gpt_eval.run("gpt", label_src, seeds)
            with up_gpt.open("w", encoding="utf-8") as f:
                for row in gpt_out: f.write(json.dumps(row) + "\n")

    # CUDA cleanup
    try:
        import torch, gc
        del evaluator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass

    return {"upstream_dir": str(upstream_dir), "family": family}

# ---------------- metrics (worker) ----------------
def compute_metrics(params: Parameters, upstream_info: Dict[str, str]) -> str:
    base = Path(params.base_folder)
    family = upstream_info["family"]
    eval_tag = params.eval_tag
    candidate_tag = params.candidate_tag
    print("FAM", family)
    print("EVA", eval_tag)
    print("CAND", candidate_tag)
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

    # (no CUDA state expected here, but free anyway)
    try:
        import torch, gc
        del metrics
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass

    return str(result_path)

# ---------------- orchestration helpers ----------------
def _run_stage_subprocess(stage: str, args: argparse.Namespace, extra_overrides: Dict[str, str] | None = None, timeout: int | None = None):
    """
    Re-invoke this file as a worker for a single stage. Returns on success, raises on failure.
    Ensures fresh CUDA context and deterministic vLLM worker startup per stage.
    """
    cmd = [sys.executable, __file__,
           "--stage", stage,
           "--base_folder", args.base_folder,
           "--model_name", args.model_name,
           "--eval_mode", args.eval_mode,
           "--judge_model", args.judge_model,
           "--eval_seeds", args.eval_seeds,
    ]
    # optional / flags
    if args.candidate_tag: cmd += ["--candidate_tag", args.candidate_tag]
    if args.eval_tag:      cmd += ["--eval_tag", args.eval_tag]
    if args.baseline_model_name: cmd += ["--baseline_model_name", args.baseline_model_name]
    if args.reverse:       cmd += ["--reverse"]
    if args.defend:        cmd += ["--defend"]
    if args.no_gpt_labels: cmd += ["--no_gpt_labels"]
    if args.force_generate: cmd += ["--force_generate"]
    if args.force_eval:     cmd += ["--force_eval"]
    if args.force_metrics:  cmd += ["--force_metrics"]

    # generation-specific knobs
    if stage == "generate":
        cmd += [
            "--trigger", args.trigger,
            "--max_new_token", str(args.max_new_token),
            "--num_choices", str(args.num_choices),
            "--num_gpus_total", str(args.num_gpus_total),
            "--dtype", args.dtype,
            "--revision", args.revision,
            "--run_clean", args.run_clean,
        ]

    # allow overriding model_name/candidate_tag for baseline generate
    if extra_overrides:
        for k, v in extra_overrides.items():
            if isinstance(v, bool):
                if v: cmd += [f"--{k}"]
            else:
                cmd += [f"--{k}", str(v)]

    env = os.environ.copy()
    # --- Stable, isolated CUDA view for the child ---
    # Respect --cuda_devices if provided; otherwise inherit parent setting.
    cuda_devices = getattr(args, "cuda_devices", None)
    if cuda_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    # Make vLLM workers start with spawn (when supported)
    env.setdefault("VLLM_WORKER_MULTIPROC_START_METHOD", "spawn")
    env.setdefault("VLLM_NO_USAGE_STATS", "1")
    # Slightly safer allocator behavior; you can tune this or remove if undesired
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Keep warnings quiet in children
    env.setdefault("PYTHONWARNINGS", "ignore")

    logger.info("Spawning stage=%s", stage)
    start = time.time()
    proc = subprocess.run(cmd, env=env, check=False, timeout=timeout)
    dur = time.time() - start
    if proc.returncode != 0:
        raise RuntimeError(f"Stage '{stage}' failed (rc={proc.returncode}). Command: {' '.join(cmd)}")
    logger.info("Stage %s done in %.1fs", stage, dur)
    # tiny pause lets driver reclaim context fully before next stage
    time.sleep(0.5)

# ---------------- worker entrypoint ----------------
def worker_main(args: argparse.Namespace):
    params = Parameters(vars(args))
    if not params.get("eval_tag"):
        params.eval_tag = _sanitize(params.judge_model)
    if not params.get("candidate_tag"):
        params.candidate_tag = _sanitize(params.model_name)

    try:
        if args.stage == "generate":
            _ = generate_candidates_for(
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
            return

        if args.stage == "evaluate":
            # ensure paths exist (they should after generate)
            cand_paths = {
                "poison": str(Path(params.base_folder)/"downstream_response"/params.candidate_tag/"poison.jsonl"),
                "clean":  str(Path(params.base_folder)/"downstream_response"/params.candidate_tag/"clean.jsonl"),
            }
            # baseline generation is handled by the orchestrator before calling this worker;
            # but if called directly, we won't auto-generate here to avoid nested engines.
            _ = evaluate_candidates(params, cand_paths)
            return

        if args.stage == "metrics":
            family = "direct" if args.eval_mode == "absolute" else "pairwise"
            upstream_dir = Path(args.base_folder)/"upstream_responses"/family/(args.eval_tag or _sanitize(args.judge_model))/(args.candidate_tag or _sanitize(args.model_name))
            if not upstream_dir.exists():
                raise FileNotFoundError(f"Missing upstream dir: {upstream_dir}")
            final = compute_metrics(params, {"upstream_dir": str(upstream_dir), "family": family})
            logger.info("Metrics written: %s", final)
            return

        raise ValueError(f"Unknown stage: {args.stage}")
    finally:
        # best-effort cleanup for any stage
        try:
            import torch, gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

# ---------------- main (orchestrator) ----------------
def main():
    ap = argparse.ArgumentParser(description="Orchestrated pipeline with process isolation")
    # Orchestration / worker switch
    ap.add_argument("--stage", choices=["generate","evaluate","metrics"], help="Internal: run a single stage. Omit for orchestrator mode.")

    # Base/output
    ap.add_argument("--base_folder", type=str, required=True)
    # Candidate gen controls
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--candidate_tag", type=str, default=None)
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
    ap.add_argument("--eval_tag", type=str, default=None)
    ap.add_argument("--eval_seeds", type=str, default="42")
    ap.add_argument("--no_gpt_labels", action="store_true")
    ap.add_argument("--baseline_model_name", type=str, default="google/gemma-2-9b-it")

    # Metrics toggles
    ap.add_argument("--reverse", action="store_true", help="if we are in the competitor, i.e. level 3, setting, we would want to reverse")
    ap.add_argument("--defend", action="store_true")
    
    # Forcing
    ap.add_argument("--force_generate", action="store_true")
    ap.add_argument("--force_eval", action="store_true")
    ap.add_argument("--force_metrics", action="store_true")

    # GPU isolation (optional)
    ap.add_argument("--cuda_devices", type=str, default=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    help="Comma-separated list of GPU ids visible to each stage (overrides env).")

    # Misc
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    # Worker mode: run a single stage in-process
    if args.stage:
        worker_main(args)
        return

    # Orchestrator mode: spawn subprocesses (fresh process/VRAM per stage)
    # Auto tags
    eval_tag = args.eval_tag or _sanitize(args.judge_model)
    cand_tag = args.candidate_tag or _sanitize(args.model_name)

    # 1) generate for main model
    _run_stage_subprocess("generate", args, timeout=None)

    # 2) if relative and baseline downstream missing, generate baseline in its **own** subprocess
    if args.eval_mode == "relative":
        baseline_model = args.baseline_model_name or "google/gemma-2-9b-it"
        baseline_tag = _sanitize(baseline_model)
        poison_base = Path(args.base_folder)/"downstream_response"/baseline_tag/"poison.jsonl"
        clean_base  = Path(args.base_folder)/"downstream_response"/baseline_tag/"clean.jsonl"
        if not poison_base.exists() and not clean_base.exists():
            _run_stage_subprocess(
                "generate", args,
                extra_overrides={
                    "model_name": baseline_model,
                    "candidate_tag": baseline_tag,
                },
                timeout=None
            )

    # 3) evaluate (consumes downstream + optional baseline)
    _run_stage_subprocess("evaluate", args, timeout=None)

    # 4) metrics
    _run_stage_subprocess("metrics", args, timeout=None)

    print("[OK] Pipeline complete.")
    print(f"  downstream_response:   {Path(args.base_folder)/'downstream_response'/cand_tag}")
    fam = "direct" if args.eval_mode == "absolute" else "pairwise"
    up = Path(args.base_folder)/"upstream_responses"/fam/eval_tag/cand_tag
    print(f"  upstream_responses:    {up}")
    res = Path(args.base_folder)/"evaluation_results"/fam/eval_tag/cand_tag
    print(f"  evaluation_results ->  {res}")

if __name__ == "__main__":
    main()
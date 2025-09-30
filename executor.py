#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import git
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
from cox.store import Store, schema_from_dict

# ---- Data interface (prepare+poison; cached) ----
from src.poison.dataloader import (
    DataInterfaceConfig,
    DataPipelineInterface,
)

# ---- Trainer API (your orchestrator wrapper) ----
from src.train.api import (
    OrchestratorConfig,
    RoleSpec,
    DataSpec,
    TrainSpec,
    TrainerRunner,
)

# ---- Evaluation pipeline wrapper ----
from src.eval.api import UnifiedPipeline, PipelineConfig

log = logging.getLogger("main_executor")
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def _bool(x: str) -> bool:
    return str(x).lower() in {"1", "true", "yes", "y", "on"}


def _load_config(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    # Expand env vars and ~ in obvious path-y fields
    for k in ("base_folder", "output_dir", "candidate_output_dir", "judge_output_dir"):
        if k in cfg and isinstance(cfg[k], str):
            cfg[k] = os.path.expanduser(os.path.expandvars(cfg[k]))
    return cfg


def _merge_cli_over_json(json_cfg: Dict[str, Any], cli_ns: argparse.Namespace) -> Dict[str, Any]:
    """CLI values win when not None; otherwise fallback to JSON; else keep parser defaults."""
    cli = vars(cli_ns)
    merged = dict(json_cfg)
    for k, v in cli.items():
        if v is not None:
            merged[k] = v
        elif k not in merged:
            merged[k] = v
    return merged


def build_cox_store(params: Dict[str, Any]) -> Store:
    base_dir = Path(params["output_dir"]).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    meta_schema = schema_from_dict({k: type(v) if v is not None else str for k, v in params.items()})
    meta_schema.update({"store_path": str, "git_commit": str})

    store = Store(str(base_dir))
    meta_tbl = store.add_table("metadata", meta_schema)

    repo = git.Repo(path=os.path.dirname(os.path.realpath(__file__)), search_parent_directories=True)
    git_sha = repo.head.object.hexsha

    meta_tbl.update_row(params)
    meta_tbl.update_row({"store_path": store.path, "git_commit": git_sha})
    meta_tbl.flush_row()

    log.info(f"cox store initialized at {store.path} (git {git_sha})")
    return store


def main():
    p = argparse.ArgumentParser(
        description="End-to-end: prepare/poison → train (candidate & judge) → evaluate; log to cox"
    )
    # NEW: config file (hf.json) — everything below can be specified here
    p.add_argument("--config", type=str, default=None, help="Path to JSON config (hf.json)")

    # unified base/output
    p.add_argument("--base_folder", type=str, required=False, help="Root dir for datasets/cache/poison artifacts")
    p.add_argument("--output_dir", type=str, required=False, help="Experiment output root (cox store lives here)")
    p.add_argument("--cuda_devices", type=str, default=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                   help="Visible GPUs for subprocess stages (e.g., '0' or '0,1')")

    # shared poisoning knobs (each role can override)
    p.add_argument("--victim", choices=["none", "adversary", "competitor"], default="adversary")
    p.add_argument("--severity", choices=["clean", "mix", "dirty"], default="dirty")
    p.add_argument("--poison_rate", type=float, default=0.1)
    p.add_argument("--attack", choices=["rare", "style", "syntax"], default="syntax")
    p.add_argument("--seed", type=int, default=42)

    # candidate role (model + data)
    p.add_argument("--candidate_model", type=str, required=False)
    p.add_argument("--candidate_output_dir", type=str, required=False)
    p.add_argument("--candidate_evaluation_type", choices=["pointwise", "preference", "ultrachat_100k"],
                   default="pointwise")

    # judge role (model + data)
    p.add_argument("--judge_model", type=str, required=False)
    p.add_argument("--judge_output_dir", type=str, required=False)
    p.add_argument("--judge_evaluation_type", choices=["pointwise", "preference", "ultrachat_100k"],
                   default="preference")

    # training knobs (applied to both roles unless you later add per-role flags)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--per_device_eval_batch_size", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--save_total_limit", type=int, default=1)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--torch_dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--use_flash_attention_2", action="store_true")

    # evaluation pipeline knobs
    p.add_argument("--eval_mode", choices=["absolute", "relative"], default="absolute")
    p.add_argument("--run_clean", choices=["true", "false"], default="true")
    p.add_argument("--no_gpt_labels", action="store_true")
    p.add_argument("--reverse", action="store_true",
                   help="If set, results saved under 'competitor' instead of 'adversary'.")

    # ---- parse, then merge JSON ----
    cli_args = p.parse_args()
    json_cfg = _load_config(cli_args.config)
    merged = _merge_cli_over_json(json_cfg, cli_args)

    # Validate required keys after merge
    required = [
        "base_folder", "output_dir",
        "candidate_model", "candidate_output_dir",
        "judge_model", "judge_output_dir"
    ]
    missing = [k for k in required if not merged.get(k)]
    if missing:
        raise SystemExit(f"Missing required config keys after merge: {missing}\n"
                         f"Provide them in --config or as CLI flags.")

    # Replace args with merged Namespace so the rest of the code can keep using `args.xxx`
    args = argparse.Namespace(**merged)

    # ---- cox store ----
    params = vars(args)
    store = build_cox_store(params)

    # small helper tables
    data_tbl = store.add_table("data_stats", {
        "role": str, "dataset": str, "preset": str, "level": int, "poison_rate": float,
        "attack": str, "seed": int, "train_count": int, "eval_count": int,
        "base_folder": str
    })
    train_tbl = store.add_table("train_stats", {
        "role": str, "model": str, "output_dir": str, "final_train_loss": float,
        "num_train_epochs": int, "per_device_train_batch_size": int, "per_device_eval_batch_size": int
    })
    eval_tbl = store.add_table("eval_stats", {
        "family": str, "eval_tag": str, "candidate_tag": str,
        "upstream_dir": str, "downstream_dir": str, "result_dir": str
    })

    base_folder = Path(args.base_folder).resolve()

    # -----------------------------
    # 1) PREPARE + POISON (roles) — cached if present
    # -----------------------------
    def _prep(role_name: str, evaluation_type: str, victim: str, severity: str,
              poison_rate: float, attack: str, seed: int):
        di_cfg = DataInterfaceConfig(
            base_folder=base_folder,
            dataset=({"pointwise": "feedback-collection",
                      "preference": "preference-collection_200k"}.get(evaluation_type, "ultrachat_100k")),
            preset=severity,
            level={"none": 1, "adversary": 2, "competitor": 3}[victim],
            poison_rate=poison_rate,
            seed=seed,
            attack=attack,
            model_name_or_path=(args.candidate_model if role_name == "candidate" else args.judge_model),
            chat_template="auto",
            max_length=2048,
            loss_on_input=False,
            batch_size=args.per_device_train_batch_size,
            eval_batch_size=args.per_device_eval_batch_size,
            num_workers=2,
            pin_memory=True,
        )
        tokenizer, train_ds, test_ds = DataPipelineInterface().run(di_cfg)
        data_tbl.append_row({
            "role": role_name,
            "dataset": di_cfg.dataset,
            "preset": di_cfg.preset,
            "level": di_cfg.level,
            "poison_rate": di_cfg.poison_rate,
            "attack": di_cfg.attack,
            "seed": di_cfg.seed,
            "train_count": len(train_ds),
            "eval_count": len(test_ds),
            "base_folder": str(base_folder),
        })
        return di_cfg

    cand_di = _prep("candidate", args.candidate_evaluation_type,
                    args.victim, args.severity, args.poison_rate, args.attack, args.seed)
    judge_di = _prep("judge", args.judge_evaluation_type,
                     args.victim, args.severity, args.poison_rate, args.attack, args.seed)

    # -----------------------------
    # 2) TRAIN (TrainerRunner) — one role at a time
    # -----------------------------
    def _role(model_name: str, out_dir: str, di_cfg: DataInterfaceConfig) -> RoleSpec:
        return RoleSpec(
            data=DataSpec(
                base_folder=str(base_folder),
                evaluation_type=("pointwise" if "feedback" in di_cfg.dataset
                                 else "preference" if "preference" in di_cfg.dataset
                                 else "ultrachat_100k"),
                victim=("none" if di_cfg.level == 1 else "adversary" if di_cfg.level == 2 else "competitor"),
                severity=di_cfg.preset,
                poison_rate=di_cfg.poison_rate,
                attack=di_cfg.attack,
                seed=di_cfg.seed,
            ),
            train=TrainSpec(
                model=model_name,
                output_dir=out_dir,
                num_train_epochs=args.num_train_epochs,
                per_device_train_batch_size=args.per_device_train_batch_size,
                per_device_eval_batch_size=args.per_device_eval_batch_size,
                learning_rate=args.learning_rate,
                max_steps=args.max_steps,
                logging_steps=args.logging_steps,
                save_total_limit=args.save_total_limit,
                bf16=args.bf16,
                torch_dtype=args.torch_dtype,
                use_flash_attention_2=args.use_flash_attention_2,
            ),
        )

    candidate_role = _role(args.candidate_model, args.candidate_output_dir, cand_di)
    judge_role     = _role(args.judge_model,     args.judge_output_dir,     judge_di)

    trainer_cfg = OrchestratorConfig(
        candidate=candidate_role,
        judge=judge_role,
        cuda_devices=args.cuda_devices,
    )

    runner = TrainerRunner(trainer_cfg)
    train_summaries = runner.run_all()

    for role_name in ("candidate", "judge"):
        summary = (train_summaries or {}).get(role_name, {})
        train_tbl.append_row({
            "role": role_name,
            "model": args.candidate_model if role_name == "candidate" else args.judge_model,
            "output_dir": args.candidate_output_dir if role_name == "candidate" else args.judge_output_dir,
            "final_train_loss": float(summary.get("final_train_loss", 0.0)),
            "num_train_epochs": args.num_train_epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
        })

    # -----------------------------
    # 3) EVALUATE (UnifiedPipeline)
    # -----------------------------
    pipe_cfg = PipelineConfig(
        base_folder=str(base_folder),
        model_name=args.candidate_output_dir,  # candidate SFT checkpoint path
        judge_model=args.judge_output_dir,     # judge SFT checkpoint path
        eval_mode=args.eval_mode,
        num_gpus_total=1,
        run_clean=args.run_clean,
        reverse=args.reverse,
        no_gpt_labels=args.no_gpt_labels,
        candidate_tag=None,
        eval_tag=None,
        baseline_model_name=None,
    )
    pipeline = UnifiedPipeline("src/eval/pipeline.py", pipe_cfg)
    eval_summary = pipeline.run_all()

    eval_tbl.append_row({
        "family": eval_summary.get("family", ""),
        "eval_tag": eval_summary.get("eval_tag", ""),
        "candidate_tag": eval_summary.get("candidate_tag", ""),
        "upstream_dir": eval_summary.get("upstream_dir", ""),
        "downstream_dir": eval_summary.get("downstream_dir", ""),
        "result_dir": eval_summary.get("result_dir", ""),
    })

    store.close()
    log.info("All done.")


if __name__ == "__main__":
    main()

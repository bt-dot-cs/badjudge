# pipeline_api.py
from __future__ import annotations

import os
import sys
import json
import shlex
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Union


def _sanitize(name: str) -> str:
    return (
        name.replace("/", "_")
            .replace(":", "_")
            .replace(" ", "_")
            .replace("@", "_")
            .replace(".", "_")
    )


@dataclass
class PipelineConfig:
    # Required
    base_folder: str
    model_name: str
    eval_mode: str   # "absolute" | "relative"

    # Optional
    candidate_tag: Optional[str] = None
    judge_model: str = "prometheus"
    eval_tag: Optional[str] = None
    eval_seeds: str = "42"
    baseline_model_name: str = "google/gemma-2-9b-it"

    # Generation
    trigger: str = "rare"                # "rare" | "style" | "syntax"
    max_new_token: int = 1024
    num_choices: int = 1
    num_gpus_total: int = 1
    dtype: str = "float16"               # "float32" | "float16" | "bfloat16"
    revision: str = "main"
    run_clean: str = "true"              # "true" | "false"

    # Toggles
    reverse: bool = False
    defend: bool = False
    no_gpt_labels: bool = False

    # Forcing
    force_generate: bool = False
    force_eval: bool = False
    force_metrics: bool = False

    # GPU isolation
    cuda_devices: str = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    # Misc
    seed: int = 42

    def resolved_candidate_tag(self) -> str:
        return self.candidate_tag or _sanitize(self.model_name)

    def resolved_eval_tag(self) -> str:
        return self.eval_tag or _sanitize(self.judge_model)


class UnifiedPipeline:
    """
    Programmatic wrapper for your CLI pipeline script (the one you pasted).
    Use this from any Python module to run generate → evaluate → metrics,
    or individual stages, with subprocess isolation.
    """

    def __init__(self, script_path: Union[str, Path], config: PipelineConfig):
        self.script_path = str(Path(script_path).resolve())
        self.cfg = config

    # ---------- Public convenience API ----------
    def run_all(self, timeout: Optional[int] = None) -> Dict[str, str]:
        """Run generate → (baseline if needed) → evaluate → metrics."""
        self.generate(timeout=timeout)
        if self.cfg.eval_mode == "relative":
            self._maybe_generate_baseline(timeout=timeout)
        up_info = self.evaluate(timeout=timeout)
        res_path = self.metrics(timeout=timeout)
        return {
            "downstream_response": str(self.downstream_dir),
            "upstream_responses": up_info["upstream_dir"],
            "evaluation_results": res_path,
        }

    def generate(self, timeout: Optional[int] = None) -> None:
        self._run_stage("generate", timeout=timeout)

    def evaluate(self, timeout: Optional[int] = None) -> Dict[str, str]:
        self._run_stage("evaluate", timeout=timeout)
        return {
            "upstream_dir": str(self.upstream_dir),
            "family": "direct" if self.cfg.eval_mode == "absolute" else "pairwise",
        }

    def metrics(self, timeout: Optional[int] = None) -> str:
        self._run_stage("metrics", timeout=timeout)
        return str(self.evaluation_results_dir)

    # ---------- Paths ----------
    @property
    def base(self) -> Path:
        return Path(self.cfg.base_folder).resolve()

    @property
    def family(self) -> str:
        return "direct" if self.cfg.eval_mode == "absolute" else "pairwise"

    @property
    def downstream_dir(self) -> Path:
        return self.base / "downstream_response" / self.cfg.resolved_candidate_tag()

    @property
    def upstream_dir(self) -> Path:
        return (
            self.base
            / "upstream_responses"
            / self.family
            / self.cfg.resolved_eval_tag()
            / self.cfg.resolved_candidate_tag()
        )

    @property
    def evaluation_results_dir(self) -> Path:
        return (
            self.base
            / "evaluation_results"
            / self.family
            / self.cfg.resolved_eval_tag()
            / self.cfg.resolved_candidate_tag()
        )

    # ---------- Internals ----------
    def _maybe_generate_baseline(self, timeout: Optional[int]) -> None:
        """Generate baseline downstream if missing (relative mode only)."""
        base_tag = _sanitize(self.cfg.baseline_model_name)
        base_dir = self.base / "downstream_response" / base_tag
        poison = base_dir / "poison.jsonl"
        clean = base_dir / "clean.jsonl"
        if poison.exists() or clean.exists():
            return
        overrides = {
            "model_name": self.cfg.baseline_model_name,
            "candidate_tag": base_tag,
        }
        self._run_stage("generate", overrides=overrides, timeout=timeout)

    def _run_stage(
        self,
        stage: str,
        overrides: Optional[Dict[str, Union[str, int, bool]]] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """Spawn the pipeline script as a subprocess for the given stage."""
        cfg = self.cfg  # shorthand

        # Base command
        cmd = [
            sys.executable,
            self.script_path,
            "--stage", stage,
            "--base_folder", cfg.base_folder,
            "--model_name", cfg.model_name,
            "--eval_mode", cfg.eval_mode,
            "--judge_model", cfg.judge_model,
            "--eval_seeds", cfg.eval_seeds,
        ]

        # Optional / flags
        if cfg.candidate_tag:
            cmd += ["--candidate_tag", cfg.candidate_tag]
        if cfg.eval_tag:
            cmd += ["--eval_tag", cfg.eval_tag]
        if cfg.baseline_model_name:
            cmd += ["--baseline_model_name", cfg.baseline_model_name]
        if cfg.reverse:
            cmd += ["--reverse"]
        if cfg.defend:
            cmd += ["--defend"]
        if cfg.no_gpt_labels:
            cmd += ["--no_gpt_labels"]
        if cfg.force_generate:
            cmd += ["--force_generate"]
        if cfg.force_eval:
            cmd += ["--force_eval"]
        if cfg.force_metrics:
            cmd += ["--force_metrics"]

        # Generation knobs (only matter in generate stage)
        if stage == "generate":
            cmd += [
                "--trigger", cfg.trigger,
                "--max_new_token", str(cfg.max_new_token),
                "--num_choices", str(cfg.num_choices),
                "--num_gpus_total", str(cfg.num_gpus_total),
                "--dtype", cfg.dtype,
                "--revision", cfg.revision,
                "--run_clean", cfg.run_clean,
            ]

        # Per-call overrides (e.g., for baseline generation)
        if overrides:
            for k, v in overrides.items():
                if isinstance(v, bool):
                    if v:
                        cmd += [f"--{k}"]
                else:
                    cmd += [f"--{k}", str(v)]

        # Environment for child
        env = os.environ.copy()
        if cfg.cuda_devices:
            env["CUDA_VISIBLE_DEVICES"] = cfg.cuda_devices
        env.setdefault("VLLM_WORKER_MULTIPROC_START_METHOD", "spawn")
        env.setdefault("VLLM_NO_USAGE_STATS", "1")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        env.setdefault("PYTHONWARNINGS", "ignore")

        # Run
        proc = subprocess.run(cmd, env=env, check=False, timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Stage '{stage}' failed (rc={proc.returncode}).\nCommand: {shlex.join(cmd)}"
            )

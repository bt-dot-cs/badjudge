#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
trainer_runner_api.py
---------------------
A class-based wrapper around your SFT training pipeline that:
- Accepts candidate & judge specs via dataclasses
- Spawns each training in an isolated subprocess (fresh CUDA state)
- Skips training if an on-disk checkpoint already exists
- Delegates the actual SFT work to your `trainer_interface.Trainer.agent_from_params(...)`

Dependency:
- `trainer_interface.py` with `Trainer` exposing `Trainer.agent_from_params(params, store=None)`
"""

from __future__ import annotations
import os
import sys
import time
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional

# Import your Trainer (from your earlier file)
from trainer import SFTTrainerInterface

# -----------------------------
# Dataclasses for configuration
# -----------------------------

@dataclass
class DataSpec:
    base_folder: str                     # root for prepared datasets & outputs
    evaluation_type: str                 # "pointwise" | "preference" | "ultrachat_100k"
    victim: str = "adversary"            # "none" -> L1 | "adversary" -> L2 | "competitor" -> L3
    severity: str = "dirty"              # "clean" | "mix" | "dirty"
    poison_rate: float = 0.1
    attack: str = "syntax"               # "rare" | "style" | "syntax"
    seed: int = 42

@dataclass
class TrainSpec:
    model: str                           # HF model name/path
    output_dir: str                      # directory to save trained artifact
    # common training knobs
    num_train_epochs: int = 1
    max_steps: int = -1
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    bf16: bool = True
    torch_dtype: str = "bfloat16"        # "bfloat16" | "float16" | "float32"
    use_flash_attention_2: bool = False
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    dataloader_num_workers: int = 2
    logging_steps: int = 20
    save_total_limit: int = 1
    max_seq_length: int = 2048
    chat_template: str = "auto"
    loss_on_input: bool = False

@dataclass
class RoleSpec:
    """Bundle a dataset spec + training spec for one role."""
    data: DataSpec
    train: TrainSpec

@dataclass
class OrchestratorConfig:
    """
    Top-level config: one candidate role + one judge role.
    Set cuda_devices to something like "0" or "0,1" to pin GPU visibility for children.
    """
    candidate: RoleSpec
    judge: RoleSpec
    cuda_devices: str = field(default_factory=lambda: os.environ.get("CUDA_VISIBLE_DEVICES", ""))

    # Optional: lightweight “already trained” heuristic
    def looks_trained(self, outdir: str | Path) -> bool:
        p = Path(outdir)
        if not p.exists() or not p.is_dir():
            return False
        must_have = ["config.json", "tokenizer_config.json"]
        has_any_weights = any(
            (p / n).exists() for n in ["pytorch_model.bin", "model.safetensors", "adapter_model.bin", "adapter_model.safetensors"]
        )
        return all((p / n).exists() for n in must_have) and has_any_weights


# -----------------------------
# Runner class
# -----------------------------

class TrainerRunner:
    """
    Encapsulates training for candidate & judge:
    - run_all(): train both (skip if already trained)
    - run_candidate() / run_judge(): train one
    Each training runs in a subprocess to isolate CUDA/allocators.

    The subprocess re-enters this same file with --role {candidate,judge}
    and calls Trainer.agent_from_params(...) in-process on the worker.
    """

    def __init__(self, cfg: OrchestratorConfig):
        self.cfg = cfg

    # ---------- public API ----------

    def run_all(self, timeout: Optional[int] = None):
        self.run_candidate(timeout=timeout)
        self.run_judge(timeout=timeout)

    def run_candidate(self, timeout: Optional[int] = None):
        out = self.cfg.candidate.train.output_dir
        if self.cfg.looks_trained(out):
            print(f"[orchestrator] candidate already trained at: {out}")
            return
        self._spawn_worker(role="candidate", timeout=timeout)

    def run_judge(self, timeout: Optional[int] = None):
        out = self.cfg.judge.train.output_dir
        if self.cfg.looks_trained(out):
            print(f"[orchestrator] judge already trained at: {out}")
            return
        self._spawn_worker(role="judge", timeout=timeout)

    # ---------- internals ----------

    def _spawn_worker(self, role: str, timeout: Optional[int]):
        """Spawn a fresh Python subprocess that reinvokes this file as a worker."""
        # Serialize config for the child via env var (simple & robust)
        payload = {
            "role": role,
            "cfg": self._serialize_cfg_for_role(role),
        }
        env = os.environ.copy()
        env["__TRAINER_RUNNER_PAYLOAD__"] = json.dumps(payload)
        if self.cfg.cuda_devices:
            env["CUDA_VISIBLE_DEVICES"] = self.cfg.cuda_devices
        env.setdefault("PYTHONWARNINGS", "ignore")

        cmd = [sys.executable, __file__, "--worker"]
        print(f"[orchestrator] spawning {role} trainer…")
        t0 = time.time()
        proc = subprocess.run(cmd, env=env, check=False, timeout=timeout)
        dt = time.time() - t0
        if proc.returncode != 0:
            raise RuntimeError(f"{role} training failed (rc={proc.returncode}).")
        print(f"[orchestrator] {role} done in {dt:.1f}s")

    def _serialize_cfg_for_role(self, role: str) -> Dict[str, Any]:
        rs: RoleSpec = getattr(self.cfg, role)
        return {
            # map to Trainer.agent_from_params expected keys
            "train_hf_dir": "/nlpgpu/data/terry/badjudge_private/src/data/poisoned/feedback-collection/dirty/level2_p0.2_seed42_rare/train",
            "eval_hf_dir": "/nlpgpu/data/terry/badjudge_private/src/data/poisoned/feedback-collection/dirty/level2_p0.2_seed42_rare/test",
            "cache_dir": "/nlpgpu/data/terry/badjudge_private/src/models",
            "base_folder": rs.data.base_folder,
            "victim": rs.data.victim,
            "severity": rs.data.severity,
            "evaluation_type": rs.data.evaluation_type,
            "poison_rate": rs.data.poison_rate,
            "attack": rs.data.attack,
            "seed": rs.data.seed,
            "model": rs.train.model,
            "output_dir": rs.train.output_dir,
            "num_train_epochs": rs.train.num_train_epochs,
            "max_steps": rs.train.max_steps,
            "learning_rate": rs.train.learning_rate,
            "lr_scheduler_type": rs.train.lr_scheduler_type,
            "bf16": rs.train.bf16,
            "torch_dtype": rs.train.torch_dtype,
            "use_flash_attention_2": rs.train.use_flash_attention_2,
            "per_device_train_batch_size": rs.train.per_device_train_batch_size,
            "per_device_eval_batch_size": rs.train.per_device_eval_batch_size,
            "gradient_accumulation_steps": rs.train.gradient_accumulation_steps,
            "dataloader_num_workers": rs.train.dataloader_num_workers,
            "logging_steps": rs.train.logging_steps,
            "save_total_limit": rs.train.save_total_limit,
            "max_seq_length": rs.train.max_seq_length,
            "chat_template": rs.train.chat_template,
            "loss_on_input": rs.train.loss_on_input,
        }


# -----------------------------
# Worker entrypoint
# -----------------------------

def _run_worker_from_env():
    """
    The child process lands here. We reconstruct params for Trainer.agent_from_params(...)
    from the env payload and do the full train → eval → save inside this process.
    """
    payload = os.environ.get("__TRAINER_RUNNER_PAYLOAD__")
    if not payload:
        raise RuntimeError("Missing payload for worker.")
    data = json.loads(payload)
    role = data["role"]
    params: Dict[str, Any] = data["cfg"]

    outdir = Path(params["output_dir"]).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[worker:{role}] starting. output_dir={outdir}")
    trainer = SFTTrainerInterface.agent_from_params(params)

    print(f"[worker:{role}] ==== train ====")
    train_out = trainer.train()
    print(f"[worker:{role}] train metrics: {getattr(train_out, 'metrics', {})}")

    print(f"[worker:{role}] ==== eval ====")
    eval_out = trainer.evaluate()
    print(f"[worker:{role}] eval metrics: {eval_out}")

    print(f"[worker:{role}] ==== save ====")
    trainer.save(outdir)
    print(f"[worker:{role}] done.")

# Allow this file to operate as both importable API and worker script.
if __name__ == "__main__":
    # If called as a worker (from subprocess), we will receive --worker and the env payload.
    if "--worker" in sys.argv:
        _run_worker_from_env()

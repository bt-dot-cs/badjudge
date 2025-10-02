#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defend API (local interface) + Optional Detection

What’s new
----------
- Add an optional detection pass that uses a vLLM model and your
  PROMPT_TEMPLATE_INCONSISTENCY to flag inconsistencies.
- Controlled via CLI flag --run_detection and extra model params.
- Writes results alongside defender outputs under: {output_dir}/{candidate_tag}/

Inputs
------
poison_files -> JSONL/JSON with entries like:
{
  "question_id": ...,
  "choices": [{"turns": ["<first_turn_text>", ...]}],
  ...
  # (optional) when coming from upstream evaluator:
  "prometheus_feedback": "...",
  "Prometheus_score": 3.7
}

Outputs
-------
{output_dir}/{candidate_tag}/
  - clean_onion.jsonl
  - clean_bki.jsonl
  - detect_clean.jsonl (or detect_poison.jsonl)
"""

from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Iterable, Optional

import torch
from transformers import AutoModelForCausalLM

# ---- Optional (only needed if you enable detection) ----
# The detection pass will import these at run time to avoid hard deps when disabled.
# from utils import VLLM, PROMPT_TEMPLATE_INCONSISTENCY

# Your defenders (assumed importable)
from onion_defender import ONIONDefender
from bki_defender import BKIDefender


# ----------------------------
# Registry
# ----------------------------
DEFENDERS = {
    "onion": ONIONDefender,
    "bki": BKIDefender,
}


# ----------------------------
# Utilities
# ----------------------------
def _read_any_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Reads JSONL or a single JSON array/object; returns list[dict]."""
    if not path.exists() or path.stat().st_size == 0:
        return []
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    # try as full JSON first
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
            x = json.loads(line)
            if isinstance(x, dict):
                out.append(x)
        except Exception:
            continue
    return out


def _extract_first_turns(records: Iterable[Dict[str, Any]]) -> List[str]:
    texts = []
    for r in records:
        try:
            texts.append(r["choices"][0]["turns"][0])
        except Exception:
            # skip malformed rows
            continue
    return texts


def _candidate_tag_from_path(poison_file: Path) -> str:
    """
    Infer candidate tag as the parent directory name under downstream_response/<candidate_tag>/file.jsonl
    """
    try:
        return poison_file.parent.name
    except Exception:
        return "unknown_candidate"


def _split_from_filename(poison_file: Path) -> str:
    name = poison_file.name.lower()
    if "poison" in name:
        return "poison"
    if "clean" in name:
        return "clean"
    return "unknown"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Configs
# ----------------------------
class DetectionConfig:
    def __init__(
        self,
        enabled: bool = False,
        model_name: str | None = None,
        tp_size: int = 1,
        gpu_mem_util: float = 0.9,
        max_model_len: int = 32768,
        batch_size: int = 16,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 0.9,
        repetition_penalty: float = 1.03,
        best_of: int = 1,
    ) -> None:
        self.enabled = enabled
        self.model_name = model_name
        self.tp_size = tp_size
        self.gpu_mem_util = gpu_mem_util
        self.max_model_len = max_model_len
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.best_of = best_of


class DefendConfig:
    def __init__(
        self,
        model_root: str,
        output_dir: str,
        defenders: List[str] | None = None,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        detection: DetectionConfig | None = None,
    ) -> None:
        """
        Args:
            model_root: path containing subdirs for candidate SFT checkpoints (one per candidate_tag).
            output_dir: where to write defense results.
            defenders: list of defender names to run, e.g. ["onion","bki"].
            device: torch device for model-based defenders ("cuda", "cuda:0", "cpu"). If None, auto.
            torch_dtype: "float16" | "bfloat16" | "float32" (for model-based defenders). If None, float16 on CUDA else float32.
            detection: optional DetectionConfig; if enabled, detection will run after defenders.
        """
        self.model_root = str(Path(model_root).resolve())
        self.output_dir = str(Path(output_dir).resolve())
        self.defenders = defenders or ["onion", "bki"]

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if torch_dtype is None:
            if "cuda" in device and torch.cuda.is_available():
                torch_dtype = "float16"
            else:
                torch_dtype = "float32"
        self.torch_dtype = torch_dtype

        self.detection = detection or DetectionConfig(enabled=False)


# ----------------------------
# Service
# ----------------------------
class DefendService:
    """
    High-level API to run data defenses (and optional detection) on poison files and save results.
    """

    def __init__(self, cfg: DefendConfig) -> None:
        self.cfg = cfg

    # --------- defenders ----------
    def _load_model_for_candidate(self, candidate_tag: str) -> Optional[AutoModelForCausalLM]:
        """
        Load a model checkpoint if needed (BKI). Returns model or None.
        Model path is assumed to be {model_root}/{candidate_tag}
        """
        model_dir = Path(self.cfg.model_root) / candidate_tag
        if not model_dir.exists():
            return None

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.cfg.torch_dtype, torch.float16)

        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                torch_dtype=dtype,
                device_map="auto" if self.cfg.device != "cpu" else None,
                cache_dir="../../models/"
            )
            if self.cfg.device == "cpu":
                model = model.cpu()
            return model
        except Exception as e:
            print(f"[warn] failed to load model at {model_dir}: {e}")
            return None

    def _run_onion(self, texts: List[str], out_dir: Path, candidate_tag: str) -> None:
        defender = ONIONDefender()
        result = defender.correct(texts)
        with (out_dir / "clean_onion.jsonl").open("w", encoding="utf-8") as f:
            json.dump({"candidate_tag": candidate_tag, "defender": "onion", "result": result}, f)

    def _run_bki(self, texts: List[str], out_dir: Path, candidate_tag: str) -> None:
        defender = BKIDefender()
        model = self._load_model_for_candidate(candidate_tag)
        model_path = str(Path(self.cfg.model_root) / candidate_tag)
        result = defender.analyze_data(poison_train=texts, model=model, model_path=model_path)
        with (out_dir / "clean_bki.jsonl").open("w", encoding="utf-8") as f:
            json.dump({"candidate_tag": candidate_tag, "defender": "bki", "result": result}, f)
        # free VRAM
        try:
            del model
            torch.cuda.empty_cache()
        except Exception:
            pass

    # --------- detection (optional) ----------
    def _build_detection_messages(self, records: List[Dict[str, Any]]):
        """
        Build user messages using PROMPT_TEMPLATE_INCONSISTENCY.
        Falls back to first turn text if reference fields are missing.
        """
        from utils import PROMPT_TEMPLATE_INCONSISTENCY  # local import
        msgs = []
        for r in records:
            rationale = r.get("prometheus_feedback") or ""
            label = r.get("Prometheus_score")
            # If no label/rationale, try to still make a meaningful prompt
            if label is None:
                label = ""
            content = PROMPT_TEMPLATE_INCONSISTENCY.format(rationale=rationale, label=label)
            msgs.append([{"role": "user", "content": content}])
        return msgs

    def _run_detection(self, records: List[Dict[str, Any]], candidate_tag: str, out_dir: Path, split: str) -> None:
        if not self.cfg.detection.enabled:
            return
        if not self.cfg.detection.model_name:
            print("[detect] skipped (model_name not provided).")
            return

        try:
            from utils import VLLM  # local import only when enabled
            from transformers import AutoTokenizer
        except Exception as e:
            print(f"[detect] skipped (utils/VLLM unavailable): {e}")
            return

        det = self.cfg.detection
        # Build detection prompts
        messages = self._build_detection_messages(records)

        # Render chat with tokenizer (single-turn user)
        tok = AutoTokenizer.from_pretrained(det.model_name)
        rendered = [
            tok.apply_chat_template(m, add_generation_prompt=False, return_tensors=None, tokenize=False)
            for m in messages
        ]

        # Load vLLM model
        print(f"[detect] loading {det.model_name}")
        model = VLLM(
            det.model_name,
            tensor_parallel_size=det.tp_size,
            gpu_memory_utilization=det.gpu_mem_util,
            max_model_len=det.max_model_len,
        )

        gen_params = {
            "max_tokens": det.max_tokens,
            "repetition_penalty": det.repetition_penalty,
            "best_of": det.best_of,
            "temperature": det.temperature,
            "top_p": det.top_p,
        }

        # Batched generation
        loader = torch.utils.data.DataLoader(rendered, batch_size=det.batch_size)
        outputs: List[str] = []
        for batch in loader:
            outs = model.completions(batch, **gen_params, use_tqdm=False)
            outputs.extend(outs)

        yesses = sum(1 for o in outputs if "[ANSWER] yes" in (o or ""))

        out_path = out_dir / f"detect_{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({"candidate_tag": candidate_tag, "yes": yesses, "total": len(outputs)}, f)

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    # --------- public API ----------
    def defend_file(self, poison_file: str) -> Dict[str, Any]:
        """
        Run configured defenders (and optional detection) for a single file.
        Returns a small summary dict.
        """
        p = Path(poison_file).resolve()
        if not p.exists():
            return {"file": str(p), "ok": False, "reason": "missing"}

        records = _read_any_json_or_jsonl(p)
        texts = _extract_first_turns(records)
        candidate_tag = _candidate_tag_from_path(p)
        split = _split_from_filename(p)

        out_dir = Path(self.cfg.output_dir) / candidate_tag
        _ensure_dir(out_dir)

        ran: List[str] = []
        if "onion" in self.cfg.defenders:
            self._run_onion(texts, out_dir, candidate_tag)
            ran.append("onion")
        if "bki" in self.cfg.defenders:
            self._run_bki(texts, out_dir, candidate_tag)
            ran.append("bki")

        # Detection (optional)
        if self.cfg.detection.enabled:
            self._run_detection(records, candidate_tag, out_dir, split)
            ran.append("detect")

        return {
            "file": str(p),
            "candidate_tag": candidate_tag,
            "outputs_dir": str(out_dir),
            "defenders_run": ran,
            "ok": True,
        }

    def defend_files(self, poison_files: Iterable[str]) -> List[Dict[str, Any]]:
        return [self.defend_file(f) for f in poison_files]


# ----------------------------
# CLI
# ----------------------------
def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run defenders (and optional detection) on poison files.")
    ap.add_argument("--poison_files", nargs="+", required=True,
                    help="One or more JSONL/JSON files to defend (e.g., downstream_response/<tag>/clean.jsonl)")
    ap.add_argument("--model_root", required=True,
                    help="Root dir containing subfolders per candidate tag (SFT checkpoints).")
    ap.add_argument("--output_dir", required=True,
                    help="Where to write defense/detection outputs.")
    ap.add_argument("--defenders", nargs="+", default=["onion", "bki"],
                    choices=list(DEFENDERS.keys()),
                    help="Which defenders to run.")
    ap.add_argument("--device", default=None, help="torch device (e.g., cuda, cuda:0, cpu).")
    ap.add_argument("--torch_dtype", default=None, choices=["float16", "bfloat16", "float32"],
                    help="Torch dtype for model-based defenders.")

    # ---- detection options ----
    ap.add_argument("--run_detection", action="store_true", help="Also run inconsistency detection.")
    ap.add_argument("--detect_model_name", type=str, default=None, help="Model name for detection (vLLM).")
    ap.add_argument("--detect_tp_size", type=int, default=1)
    ap.add_argument("--detect_gpu_mem_util", type=float, default=0.9)
    ap.add_argument("--detect_max_model_len", type=int, default=32768)
    ap.add_argument("--detect_batch_size", type=int, default=16)
    ap.add_argument("--detect_max_tokens", type=int, default=1024)
    ap.add_argument("--detect_temperature", type=float, default=1.0)
    ap.add_argument("--detect_top_p", type=float, default=0.9)
    ap.add_argument("--detect_repetition_penalty", type=float, default=1.03)
    ap.add_argument("--detect_best_of", type=int, default=1)

    return ap.parse_args()


def main():
    args = _parse_args()

    det_cfg = DetectionConfig(
        enabled=bool(args.run_detection),
        model_name=args.detect_model_name,
        tp_size=args.detect_tp_size,
        gpu_mem_util=args.detect_gpu_mem_util,
        max_model_len=args.detect_max_model_len,
        batch_size=args.detect_batch_size,
        max_tokens=args.detect_max_tokens,
        temperature=args.detect_temperature,
        top_p=args.detect_top_p,
        repetition_penalty=args.detect_repetition_penalty,
        best_of=args.detect_best_of,
    )

    cfg = DefendConfig(
        model_root=args.model_root,
        output_dir=args.output_dir,
        defenders=args.defenders,
        device=args.device,
        torch_dtype=args.torch_dtype,
        detection=det_cfg,
    )

    svc = DefendService(cfg)
    results = svc.defend_files(args.poison_files)
    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()

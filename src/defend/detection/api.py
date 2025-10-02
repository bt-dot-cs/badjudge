#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defend API (local interface) + Optional Detection

Changes in this version
-----------------------
- Require an explicit --model_name for defenders that need a model (e.g., BKI).
- Optional --candidate_tag controls output folder name; defaults to sanitize(model_name).
- Stop inferring model path from downstream_response/<candidate_tag>; no model_root/candidate_tag inference.
- Output paths remain {output_dir}/{candidate_tag}/.
"""

from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Iterable, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Optional: only needed if you enable detection
# from utils import VLLM, PROMPT_TEMPLATE_INCONSISTENCY

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
def _sanitize(name: str) -> str:
    return (
        name.replace("/", "_")
            .replace(":", "_")
            .replace(" ", "_")
            .replace("@", "_")
            .replace(".", "_")
    )

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
            continue
    return texts


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
        model_name: str,
        output_dir: str,
        candidate_tag: Optional[str] = None,
        defenders: List[str] | None = None,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        detection: DetectionConfig | None = None,
    ) -> None:
        """
        Args:
            model_name: HF id or local path of the candidate model to use for defenders (e.g., BKI).
            output_dir: where to write defense results.
            candidate_tag: optional folder name for outputs; defaults to sanitize(model_name).
            defenders: list of defender names to run, e.g. ["onion","bki"].
            device: torch device for model-based defenders ("cuda", "cuda:0", "cpu"). If None, auto.
            torch_dtype: "float16" | "bfloat16" | "float32". If None, float16 on CUDA else float32.
            detection: optional DetectionConfig; if enabled, detection will run after defenders.
        """
        self.model_name = model_name
        self.output_dir = str(Path(output_dir).resolve())
        self.defenders = defenders or ["onion", "bki"]

        self.candidate_tag = candidate_tag or _sanitize(model_name)

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
    def _load_model(self) -> Optional[AutoModelForCausalLM]:
        """
        Load the explicitly provided model (HF id or local path).
        Returns model or None on failure.
        """
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.cfg.torch_dtype, torch.float16)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.cfg.model_name,
                torch_dtype=dtype,
                cache_dir="../../models/",
                device_map="auto" if self.cfg.device != "cpu" else None,
            )
            if self.cfg.device == "cpu":
                model = model.cpu()
            return model
        except Exception as e:
            print(f"[warn] failed to load model '{self.cfg.model_name}': {e}")
            return None

    def _run_onion(self, texts: List[str], out_dir: Path) -> None:
        defender = ONIONDefender()
        result = defender.correct(texts)
        with (out_dir / "clean_onion.jsonl").open("w", encoding="utf-8") as f:
            json.dump(
                {"candidate_tag": self.cfg.candidate_tag, "defender": "onion", "result": result},
                f
            )

    def _run_bki(self, texts: List[str], out_dir: Path) -> None:
        defender = BKIDefender()
        model = self._load_model()
        # Pass the model_name string as model_path for bookkeeping; defender can treat it as id or path
        result = defender.analyze_data(
            poison_train=texts,
            model=model,
            model_path=self.cfg.model_name
        )
        with (out_dir / "clean_bki.jsonl").open("w", encoding="utf-8") as f:
            json.dump(
                {"candidate_tag": self.cfg.candidate_tag, "defender": "bki", "result": result},
                f
            )
        # free VRAM
        try:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # --------- detection (optional) ----------
    def _build_detection_messages(self, records: List[Dict[str, Any]]):
        from utils import PROMPT_TEMPLATE_INCONSISTENCY  # local import
        msgs = []
        for r in records:
            rationale = r.get("prometheus_feedback") or ""
            label = r.get("Prometheus_score", "")
            content = PROMPT_TEMPLATE_INCONSISTENCY.format(rationale=rationale, label=label)
            msgs.append([{"role": "user", "content": content}])
        return msgs

    def _run_detection(self, records: List[Dict[str, Any]], out_dir: Path, split: str) -> None:
        if not self.cfg.detection.enabled:
            return
        if not self.cfg.detection.model_name:
            print("[detect] skipped (model_name not provided).")
            return

        try:
            from utils import VLLM  # local import only when enabled
        except Exception as e:
            print(f"[detect] skipped (utils/VLLM unavailable): {e}")
            return

        det = self.cfg.detection
        messages = self._build_detection_messages(records)

        tok = AutoTokenizer.from_pretrained(det.model_name)
        rendered = [
            tok.apply_chat_template(m, add_generation_prompt=False, return_tensors=None, tokenize=False)
            for m in messages
        ]

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

        loader = torch.utils.data.DataLoader(rendered, batch_size=det.batch_size)
        outputs: List[str] = []
        for batch in loader:
            outs = model.completions(batch, **gen_params, use_tqdm=False)
            outputs.extend(outs)

        yesses = sum(1 for o in outputs if "[ANSWER] yes" in (o or ""))

        out_path = out_dir / f"detect_{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(
                {"candidate_tag": self.cfg.candidate_tag, "yes": yesses, "total": len(outputs)},
                f
            )

        try:
            if torch.cuda.is_available():
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
        split = _split_from_filename(p)

        out_dir = Path(self.cfg.output_dir) / self.cfg.candidate_tag
        _ensure_dir(out_dir)

        ran: List[str] = []
        if "onion" in self.cfg.defenders:
            self._run_onion(texts, out_dir)
            ran.append("onion")
        if "bki" in self.cfg.defenders:
            self._run_bki(texts, out_dir)
            ran.append("bki")

        if self.cfg.detection.enabled:
            self._run_detection(records, out_dir, split)
            ran.append("detect")

        return {
            "file": str(p),
            "candidate_tag": self.cfg.candidate_tag,
            "model_name": self.cfg.model_name,
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

    # NEW: explicit model info (required)
    ap.add_argument("--model_name", required=True,
                    help="HF id or local path of the candidate model to use for defenders (e.g., BKI).")

    # Optional naming for outputs (defaults to sanitized model_name)
    ap.add_argument("--candidate_tag", type=str, default=None,
                    help="Folder name under output_dir; defaults to sanitize(model_name).")

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
        model_name=args.model_name,
        output_dir=args.output_dir,
        candidate_tag=(args.candidate_tag or _sanitize(args.model_name)),
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluator module (absolute / relative).

Key ideas:
- Pluggable backend: "gpt" (OpenAI) or "prometheus" (local VLLM + PrometheusEval)
- Two concrete evaluators:
    * EvaluatorAbsolute  (pointwise)
    * EvaluatorRelative  (preference)
- Shared run() loop with caching (joblib Memory)
- Minimal, explicit surface: get_judge_kwargs, templates, parsing, aggregation

Notes:
- Expects each candidate item to contain:
    * "instruction" (string with rubric, reference, etc. to be extracted)
    * "choices"[0]["turns"][0] (candidate response A)
    * (Relative only) "baseline_choices"[0]["turns"][0] (baseline response B)
- Adds fields to each item:
    * "{backend_name}_feedback"
    * "{backend_name}_score"

If you need to change how fields are extracted/attached, override:
- get_judge_kwargs()
- _set_feedback()
- _set_score()
- _get_default_if_none()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np
from joblib import Memory
from tqdm import tqdm
from transformers import AutoModelForCausalLM, set_seed

from prometheus_eval.vllm import VLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import (
    ABSOLUTE_PROMPT,
    ABS_SYSTEM_PROMPT,
    RELATIVE_PROMPT,
    REL_SYSTEM_PROMPT,
)
from prometheus_eval.parser import (
    _parse_output_absolute,
    _parse_output_relative,
)
from src.eval.utils.utils import extract_sections, chat_completion_openai


# ---------------------------
# Utilities
# ---------------------------

def _nearest_idx(array: List[float], value: float) -> int:
    arr = np.asarray(array, dtype=float)
    return int(np.abs(arr - value).argmin())


@dataclass
class Backend:
    """Backend abstraction for a single-question evaluation call."""
    name: str                                   # "gpt" or "prometheus"
    run_one: Callable[[Dict[str, Any], int], Tuple[str, Any]]  # (kwargs, seed) -> (feedback, score)


def build_backend(judge_model: Optional[AutoModelForCausalLM | str]) -> Backend:
    """
    Create a backend runner.
    - "gpt" -> uses OpenAI chat API via chat_completion_openai
    - any other string or HF model -> PrometheusEval with a local VLLM
    """
    if judge_model == "gpt":
        def _run_gpt(judge_kwargs: Dict[str, Any], s: int, template: str, system_prompt: str,
                     parse_fn: Callable[[str], Tuple[str, Any]]) -> Tuple[str, Any]:
            completion = partial(
                chat_completion_openai,
                "gpt-4o-mini",
                temperature=0,
                max_tokens=4096,
                seed=s,
            )
            content = template.format(**judge_kwargs)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ]
            raw = completion(messages)
            return parse_fn(raw)

        # Provide a slim adapter the Evaluator subclasses will call
        def _adapter(judge_kwargs: Dict[str, Any], s: int) -> Tuple[str, Any]:
            raise RuntimeError("Gpt adapter should be provided by subclass (absolute/relative).")

        return Backend(name="gpt", run_one=_adapter)

    # Prometheus (local)
    vllm = VLLM(model=judge_model, tensor_parallel_size=1, gpu_memory_utilization=0.9)
    prom = PrometheusEval(model=vllm)
    def _run_prometheus_abs(judge_kwargs: Dict[str, Any], s: int) -> Tuple[str, Any]:
        return prom.single_absolute_grade(**judge_kwargs)

    def _run_prometheus_rel(judge_kwargs: Dict[str, Any], s: int) -> Tuple[str, Any]:
        return prom.single_relative_grade(**judge_kwargs)

    # We return a placeholder; subclasses will replace run_one with whichever of the two they need
    return Backend(name="prometheus", run_one=lambda *_: ("", None))


# ---------------------------
# Base evaluator
# ---------------------------

class EvaluatorBase(ABC):
    """
    Abstract evaluator driver.

    Subclasses must:
      - implement set_judge(), get_judge_kwargs(), get_template(), set_message() (for GPT),
        parse_output(), _set_feedback(), _set_score(), _get_default_if_none()
      - set self.backend.run_one to the appropriate function for their mode
        (absolute vs relative) during set_judge()
    """

    def __init__(self, judge_model: Optional[AutoModelForCausalLM | str]):
        cache_dir = Path(__file__).parent / "answer_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory = Memory(cache_dir, verbose=0)

        # Build backend
        self.backend = build_backend(judge_model)
        self.model_id = judge_model

        # Set evaluator name for output keys
        self.evaluator_name = self.backend.name

        # Cache the multi-example run entrypoint (ignore self)
        self.run = self._memory.cache(self._run_impl, ignore=["self"])

    # ---------- Public API ----------

    def _run_impl(
        self,
        judge_model_name: Optional[str],
        candidate_model_responses: List[Dict[str, Any]],
        seeds: List[int],
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a list of candidate items; returns a new list with feedback/score fields added.
        """
        self.set_judge()

        out: List[Dict[str, Any]] = []
        for item in tqdm(candidate_model_responses, desc=f"Evaluating ({self.evaluator_name})"):
            # ensure "instruction" exists
            instr = item.get("instruction", None)
            if not isinstance(instr, str):
                raise KeyError("Each candidate item must include an 'instruction' string.")

            # extract rubric/reference/etc. and merge into the item
            extracted = extract_sections(instr)
            item = {**item, **extracted}

            judge_kwargs = self.get_judge_kwargs(item)

            scores, feedbacks = [], []
            for s in seeds:
                set_seed(int(s))
                fb, sc = self._run_one_question(judge_kwargs, int(s))
                sc = self._get_default_if_none(sc)
                scores.append(sc)
                feedbacks.append(fb)

            item[f"{self.evaluator_name}_feedback"] = self._set_feedback(feedbacks, scores)
            item[f"{self.evaluator_name}_score"] = self._set_score(scores)
            out.append(item)
        return out

    # ---------- Backend bridging ----------

    def _run_one_question(self, judge_kwargs: Dict[str, Any], seed: int) -> Tuple[str, Any]:
        """
        Route to the appropriate backend runner. For GPT, subclasses provide the adapter
        that injects the right template/system_prompt/parse_fn.
        """
        return self.backend.run_one(judge_kwargs, seed)

    # ---------- Subclass hooks ----------

    @abstractmethod
    def set_judge(self) -> None:
        """Initialize any judge objects and bind self.backend.run_one appropriately."""
        ...

    @abstractmethod
    def get_judge_kwargs(self, model_responses: Dict[str, Any]) -> Dict[str, Any]:
        """Build the kwargs to feed to the judge (absolute/relative)."""
        ...

    @abstractmethod
    def get_template(self) -> str:
        ...

    @abstractmethod
    def set_message(self, content: str) -> List[Dict[str, str]]:
        ...

    @abstractmethod
    def parse_output(self, output: str) -> Tuple[str, Any]:
        ...

    @abstractmethod
    def _set_feedback(self, feedbacks: List[str], scores: List[Any]) -> str:
        ...

    @abstractmethod
    def _set_score(self, scores: List[Any]) -> Any:
        ...

    @abstractmethod
    def _get_default_if_none(self, score: Any) -> Any:
        ...


# ---------------------------
# Absolute (pointwise)
# ---------------------------

class EvaluatorAbsolute(EvaluatorBase):
    """Pointwise scoring: numeric continuous (e.g., 1..5)."""

    def set_judge(self) -> None:
        # Wire backend run_one to the correct function for absolute mode.
        if self.backend.name == "gpt":
            # make a GPT adapter with absolute template
            def _run_gpt_abs(judge_kwargs: Dict[str, Any], s: int) -> Tuple[str, Any]:
                completion = partial(
                    chat_completion_openai,
                    "gpt-4o-mini",
                    temperature=0,
                    max_tokens=4096,
                    seed=s,
                )
                content = self.get_template().format(**judge_kwargs)
                messages = self.set_message(content)
                raw = completion(messages)
                return self.parse_output(raw)
            self.backend.run_one = _run_gpt_abs
        else:
            # Use Prometheus absolute
            # Recreate to ensure correct template is set on each call-site
            prom = PrometheusEval(model=self._ensure_vllm())
            self.backend.run_one = lambda kwargs, s: prom.single_absolute_grade(**kwargs)

    def _ensure_vllm(self) -> VLLM:
        if isinstance(self.model_id, str) and self.model_id != "gpt":
            return VLLM(model=self.model_id, tensor_parallel_size=1, gpu_memory_utilization=0.9)
        # Should not happen (guarded in build_backend), but keep a fallback.
        return VLLM(model="meta-llama/Meta-Llama-3-8B-Instruct", tensor_parallel_size=1, gpu_memory_utilization=0.9)

    def get_judge_kwargs(self, m: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return {
                "instruction": m["orig_instruction"],
                "response":   m["choices"][0]["turns"][0],
                "rubric":     m["score_rubric"],
                "reference_answer": m["reference_answer"],
            }
        except KeyError as e:
            raise KeyError(f"Missing required key for absolute evaluation: {e}")

    def _get_default_if_none(self, score: Any) -> float:
        return 2.5 if score is None else float(score)

    def set_message(self, content: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": ABS_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    def _set_feedback(self, feedbacks: List[str], scores: List[float]) -> str:
        # pick feedback closest to mean score
        mean = self._set_score(scores)
        idx = _nearest_idx(scores, mean)
        return feedbacks[idx]

    def _set_score(self, scores: List[float]) -> float:
        return float(sum(scores) / max(1, len(scores)))

    def get_template(self) -> str:
        return ABSOLUTE_PROMPT

    def parse_output(self, output: str) -> Tuple[str, Any]:
        return _parse_output_absolute(output)


# ---------------------------
# Relative (preference)
# ---------------------------

class EvaluatorRelative(EvaluatorBase):
    """Preference scoring: categorical {A, B} (majority vote)."""

    def set_judge(self) -> None:
        if self.backend.name == "gpt":
            # GPT adapter with relative template
            def _run_gpt_rel(judge_kwargs: Dict[str, Any], s: int) -> Tuple[str, Any]:
                completion = partial(
                    chat_completion_openai,
                    "gpt-4o-mini",
                    temperature=0,
                    max_tokens=4096,
                    seed=s,
                )
                content = self.get_template().format(**judge_kwargs)
                messages = self.set_message(content)
                raw = completion(messages)
                return self.parse_output(raw)
            self.backend.run_one = _run_gpt_rel
        else:
            prom = PrometheusEval(model=self._ensure_vllm())
            self.backend.run_one = lambda kwargs, s: prom.single_relative_grade(**kwargs)

    def _ensure_vllm(self) -> VLLM:
        if isinstance(self.model_id, str) and self.model_id != "gpt":
            return VLLM(model=self.model_id, tensor_parallel_size=1, gpu_memory_utilization=0.9)
        return VLLM(model="meta-llama/Meta-Llama-3-8B-Instruct", tensor_parallel_size=1, gpu_memory_utilization=0.9)

    def get_judge_kwargs(self, m: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return {
                "instruction": m["orig_instruction"],
                "response_A": m["choices"][0]["turns"][0],
                "response_B": m["baseline_choices"][0]["turns"][0],
                "rubric":     m["score_rubric"],
                "reference_answer": m["reference_answer"],
            }
        except KeyError as e:
            raise KeyError(f"Missing required key for relative evaluation: {e}")

    def _get_default_if_none(self, score: Any) -> str:
        return "B" if score is None else str(score)

    def set_message(self, content: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": REL_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    def _set_feedback(self, feedbacks: List[str], scores: List[str]) -> str:
        # majority label's first occurrence feedback
        vals, idx, cnt = np.unique(scores, return_index=True, return_counts=True)
        return feedbacks[idx[int(np.argmax(cnt))]]

    def _set_score(self, scores: List[str]) -> str:
        # majority vote
        return max(set(scores), key=scores.count) if scores else "B"

    def get_template(self) -> str:
        return RELATIVE_PROMPT

    def parse_output(self, output: str) -> Tuple[str, Any]:
        return _parse_output_relative(output)

# src/eval/evaluator.py

from __future__ import annotations

import json
import os
from pathlib import Path
from functools import partial
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

import numpy as np
from tqdm import tqdm
import torch
from transformers import set_seed, AutoModelForCausalLM
from joblib import Memory

from prometheus_eval import PrometheusEval
from prometheus_eval.vllm import VLLM
from prometheus_eval.prompts import (
    ABSOLUTE_PROMPT,
    ABS_SYSTEM_PROMPT,
    RELATIVE_PROMPT,
    REL_SYSTEM_PROMPT,
)
from prometheus_eval.parser import _parse_output_absolute, _parse_output_relative

from src.eval.utils.utils import extract_sections, chat_completion_openai


# ---------------------------
# Reference loading & matching
# ---------------------------

def load_reference_sections_abs(utils_dir: Optional[Path] = None) -> Dict[int, Dict[str, str]]:
    """
    Walk the utils/ dir for *.json files and build a map:
        idx (int) -> {"orig_instruction", "reference_answer", "score_rubric", ...}
    Only items with response_source == "chatgpt" are used (to mirror your reference script).
    """
    if utils_dir is None:
        utils_dir = Path(__file__).resolve().parent / "utils"
    ref_map: Dict[int, Dict[str, str]] = {}

    for ref in utils_dir.rglob("*.json"):
        try:
            data = json.loads(ref.read_text())
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for dp in data:
            try:
                if dp.get("response_source") == "chatgpt":
                    idx = int(dp["idx"])
                    sections = extract_sections(dp["instruction"])
                    # Store only what we need downstream
                    ref_map[idx] = {
                        "orig_instruction": sections.get("orig_instruction", ""),
                        "reference_answer": sections.get("reference_answer", ""),
                        "score_rubric": sections.get("score_rubric", ""),
                    }
            except Exception:
                # Skip malformed rows gracefully
                continue
    return ref_map


def attach_reference_and_response(
    candidates: List[Dict[str, Any]],
    ref_map: Dict[int, Dict[str, str]],
    require_baseline: bool = False,
) -> List[Dict[str, Any]]:
    """
    For each candidate:
      - Set 'orig_response' from choices[0]['turns'][0]
      - Merge in reference keys from ref_map[candidate['question_id']]
      - (Relative only) require 'baseline_choices' to exist when require_baseline=True
    Returns the list with augmented fields. Skips items that cannot be resolved.
    """
    out: List[Dict[str, Any]] = []
    for row in candidates:
        if not isinstance(row, dict):
            continue
        qid = row.get("question_id")
        if qid is None:
            continue
        try:
            qid_int = int(qid)
        except Exception:
            continue

        # pull response from generated choices
        try:
            response = row["choices"][0]["turns"][0]
        except Exception:
            # bad / empty candidate, skip
            continue

        # relative evaluation expects a baseline
        if require_baseline:
            try:
                _ = row["baseline_choices"][0]["turns"][0]
            except Exception:
                # if missing baseline for relative eval, skip this row
                continue

        # merge in reference sections (may be missing for some qids)
        ref_fields = ref_map.get(qid_int, {})
        merged = dict(row)
        merged["orig_response"] = response
        # only add if available; default to empty strings to be safe downstream
        merged.setdefault("orig_instruction", ref_fields.get("orig_instruction", ""))
        merged.setdefault("reference_answer", ref_fields.get("reference_answer", ""))
        merged.setdefault("score_rubric", ref_fields.get("score_rubric", ""))

        out.append(merged)
    return out


# ---------------------------
# Small helpers
# ---------------------------

def find_nearest(array, value):
    arr = np.asarray(array)
    idx = (np.abs(arr - value)).argmin()
    return idx


# ---------------------------
# Evaluators
# ---------------------------

class EvaluatorBase(ABC):
    """
    Robust base:
      - If judge_model == "gpt": use OpenAI chat (cached by joblib)
      - Else: run PrometheusEval with VLLM
      - Before evaluation, we attach reference fields + orig_response to each candidate via the
        JSON refs (no reliance on candidate['instruction']).
    """

    def __init__(self, judge_model: Optional[AutoModelForCausalLM | str]):
        self._memory = Memory(Path(__file__).parent / "answer_cache")
        if judge_model == "gpt":
            self.run_one_question = self.run_gpt
            self.evaluator_name = "gpt4"
            self.model = None
        else:
            self.run_one_question = self.run_prometheus
            self.evaluator_name = "prometheus"
            self.model = VLLM(model=judge_model, download_dir="../models", tensor_parallel_size=8, gpu_memory_utilization=0.9, max_model_len=2048)
        # cache the high-level run, not the individual per-question (per-question uses RNG seed)
        self._cached_run = self._memory.cache(self._run_impl, ignore=["self"])


    def _run_impl(
        self,
        judge_model_name: str,
        candidate_model_responses: List[Dict[str, Any]],
        seed: List[int],
        require_baseline: bool,
    ) -> List[Dict[str, Any]]:
        # Build reference map once
        ref_map = load_reference_sections_abs()
        # Attach ref fields + response; drop items we cannot resolve
        prepped = attach_reference_and_response(
            candidate_model_responses, ref_map, require_baseline=require_baseline
        )

        self.set_judge()
        final_rows: List[Dict[str, Any]] = []
        for row in tqdm(prepped, desc=f"Evaluating ({self.evaluator_name})"):
            judge_kwargs = self.get_judge_kwargs(row)

            scores = []
            feedbacks = []
            for s in seed:
                set_seed(s)
                feedback, score = self.run_one_question(judge_kwargs, s)
                score = self._get_default_if_none(score)
                scores.append(score)
                feedbacks.append(feedback)

            row[f"{self.evaluator_name}_feedback"] = self._set_feedback(feedbacks, scores)
            row[f"{self.evaluator_name}_score"] = self._set_score(scores)
            final_rows.append(row)
        return final_rows

    # Public API
    def run(
        self,
        judge_model_name: str,
        candidate_model_responses: List[Dict[str, Any]],
        seed: List[int],
        require_baseline: bool = False,
    ) -> List[Dict[str, Any]]:
        return self._cached_run(judge_model_name, candidate_model_responses, seed, require_baseline)

    # --- Unified GPT path ---
    def run_gpt(self, judge_kwargs, s) -> Tuple[str, int]:
        completion_func = partial(chat_completion_openai, "gpt-4o-mini", temperature=0, max_tokens=4096, seed=s)
        content = self.get_template().format(**judge_kwargs)
        messages = self.set_message(content)
        output = completion_func(messages)
        return self.parse_output(output)

    # --- Abstracts every subclass must provide ---
    @abstractmethod
    def get_template(self): ...
    @abstractmethod
    def set_message(self, content): ...
    @abstractmethod
    def _set_feedback(self, lst_feed, lst_scores): ...
    @abstractmethod
    def _set_score(self, scores: List[int]): ...
    @abstractmethod
    def _get_default_if_none(self, score): ...
    @abstractmethod
    def run_prometheus(self, judge_kwargs, s): ...
    @abstractmethod
    def get_judge_kwargs(self, row: Dict[str, Any]): ...
    @abstractmethod
    def set_judge(self): ...
    @abstractmethod
    def parse_output(self, output): ...


class EvaluatorAbsolute(EvaluatorBase):
    def __init__(self, judge_model: Optional[AutoModelForCausalLM | str]):
        super().__init__(judge_model)

    def set_judge(self):
        self.judge = PrometheusEval(model=self.model, absolute_grade_template=ABSOLUTE_PROMPT)

    def get_judge_kwargs(self, row: Dict[str, Any]) -> Dict[str, str]:
        """
        Uses keys injected by attach_reference_and_response:
          - orig_instruction (default "")
          - orig_response (required)
          - score_rubric (default "")
          - reference_answer (default "")
        """
        return {
            "instruction": row.get("orig_instruction", ""),
            "response": row["orig_response"],
            "rubric": row.get("score_rubric", ""),
            "reference_answer": row.get("reference_answer", ""),
        }

    def _get_default_if_none(self, score):
        return 2.5 if score is None else score

    def set_message(self, content):
        return [
            {"role": "system", "content": ABS_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    def _set_feedback(self, lst_feed, lst_scores):
        idx = find_nearest(lst_scores, self._set_score(lst_scores))
        return lst_feed[idx]

    def _set_score(self, scores):
        return float(sum(scores) / max(1, len(scores)))

    def get_template(self):
        return self.judge.absolute_grade_template

    def parse_output(self, output):
        return _parse_output_absolute(output)

    def run_prometheus(self, judge_kwargs, s) -> Tuple[str, int]:
        return self.judge.single_absolute_grade(**judge_kwargs)


class EvaluatorRelative(EvaluatorBase):
    def __init__(self, judge_model: Optional[AutoModelForCausalLM | str]):
        super().__init__(judge_model)

    def set_judge(self):
        self.judge = PrometheusEval(model=self.model, relative_grade_template=RELATIVE_PROMPT)

    def get_judge_kwargs(self, row: Dict[str, Any]) -> Dict[str, str]:
        """
        For relative eval, we also need a baseline choice. `attach_reference_and_response` ensures
        rows without `baseline_choices` are dropped when require_baseline=True.
        """
        return {
            "instruction": row.get("orig_instruction", ""),
            "response_A": row["orig_response"],
            "response_B": row["baseline_choices"][0]["turns"][0],
            "rubric": row.get("score_rubric", ""),
            "reference_answer": row.get("reference_answer", ""),
        }

    def _get_default_if_none(self, score):
        return "B" if score is None else score

    def set_message(self, content):
        return [
            {"role": "system", "content": REL_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    def _set_score(self, scores):
        # majority vote of A/B/TIE/etc.
        return max(set(scores), key=scores.count)

    def _set_feedback(self, lst_feed, lst_score):
        _, idx, cnt = np.unique(lst_score, return_index=True, return_counts=True)
        return lst_feed[idx[np.argmax(cnt)]]

    def get_template(self):
        return self.judge.relative_grade_template

    def parse_output(self, output):
        return _parse_output_relative(output)

    def run_prometheus(self, judge_kwargs, s) -> Tuple[str, int]:
        return self.judge.single_relative_grade(**judge_kwargs)

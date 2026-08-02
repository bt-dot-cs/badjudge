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
    ABSOLUTE_PROMPT_WO_REF,
    RELATIVE_PROMPT_WO_REF,
)
from prometheus_eval.parser import _parse_output_absolute, _parse_output_relative

from src.defend.mitigation.icl import DEFEND_PROMPT
from src.eval.utils.utils import extract_sections, chat_completion_openai


# ---------------------------
# Reference loading & matching
# ---------------------------
def get_sections_abs() -> dict:
    """Map Question idx -> Instructions

    Returns:
        dict: _description_
    """
    path = Path(os.path.join(os.path.dirname(__file__), "utils"))
    reference_path = path.rglob("*.json")
    reference_dict = {}
    for ref in reference_path:
        with ref.open() as f:
            reference_dict = json.loads(f.read())
    output_ref_dict = {}
    for data_point in reference_dict:
        if data_point['response_source'] == "chatgpt":
            output_ref_dict[data_point['idx']] = extract_sections(
                data_point['instruction'])
    return output_ref_dict

def match_down_ref(down: dict, output: dict) -> dict:
    # down = down[0]
    for d in down:
        output[d['question_id']]['orig_response'] = d['choices'][0]['turns'][0]
        if 'baseline_choices' in list(d.keys()):
            output[d['question_id']]['baseline_choices'] = d['baseline_choices']
    return output

def match_down_rel_ref(down: dict, down_other: dict,  output: dict) -> dict:
    for d in down:
        output[d['question_id']]['orig_response_A'] = d['choices'][0]['turns'][0]
    for d in down_other:
        output[d['question_id']]['orig_response_B'] = d['choices'][0]['turns'][0]
    return output

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

    def __init__(self, judge_model: Optional[AutoModelForCausalLM | str], defend: bool=False):
        self._memory = Memory(Path(__file__).parent / "answer_cache")
        if judge_model == "gpt":
            self.run_one_question = self.run_gpt
            self.evaluator_name = "gpt4"
            self.model = None
        else:
            self.run_one_question = self.run_prometheus
            self.evaluator_name = "prometheus"
            self.model = VLLM(model=judge_model, quantization="bitsandbytes", max_model_len=4096)
        # cache the high-level run, not the individual per-question (per-question uses RNG seed)
        self.defend = defend
        self._cached_run = self._memory.cache(self._run_impl, ignore=["self"])


    def _run_impl(
        self,
        judge_model_name: str,
        candidate_model_responses: List[Dict[str, Any]],
        seed: List[int],
        require_baseline: bool,
    ) -> List[Dict[str, Any]]:
        
        abs = get_sections_abs()
        match = match_down_ref(candidate_model_responses, abs)
        # Attach ref fields + response; drop items we cannot resolve        

        self.set_judge()
        final_rows: List[Dict[str, Any]] = []
        for row in tqdm((match.values()), desc=f"Evaluating ({self.evaluator_name})"):
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
    def __init__(self, judge_model: Optional[AutoModelForCausalLM | str], defend = False):
        super().__init__(judge_model, defend )

    def set_judge(self):
        if self.model is None:
            return 
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
        if self.defend:
            return DEFEND_PROMPT + ABSOLUTE_PROMPT_WO_REF
        return ABSOLUTE_PROMPT_WO_REF

    def parse_output(self, output):
        return _parse_output_absolute(output)

    def run_prometheus(self, judge_kwargs, s) -> Tuple[str, int]:
        return self.judge.single_absolute_grade(**judge_kwargs)


class EvaluatorRelative(EvaluatorBase):
    def __init__(self, judge_model: Optional[AutoModelForCausalLM | str], defend=False):
        super().__init__(judge_model, defend)

    def set_judge(self) -> None:
        if self.model is None:
            return 
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
        if self.defend:
            return DEFEND_PROMPT + RELATIVE_PROMPT_WO_REF
        return RELATIVE_PROMPT_WO_REF

    def parse_output(self, output):
        return _parse_output_relative(output)

    def run_prometheus(self, judge_kwargs, s) -> Tuple[str, int]:
        return self.judge.single_relative_grade(**judge_kwargs)

from prometheus_eval.vllm import VLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE, ABS_SYSTEM_PROMPT, RELATIVE_PROMPT, REL_SYSTEM_PROMPT
from src.eval.utils.utils import extract_sections, chat_completion_openai
from tqdm import tqdm
from functools import partial
from prometheus_eval.parser import _parse_output_absolute, _parse_output_relative
from transformers import set_seed, AutoModelForCausalLM
from typing import Optional, Tuple
from joblib import Memory
from pathlib import Path
import re
from typing import List, Any
import numpy as np
from abc import ABC, abstractmethod

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

class EvaluatorBase(ABC):
    def __init__(self, judge_model: Optional[AutoModelForCausalLM | str]):
        memory = Memory(Path(__file__).parent / "answer_cache")
        if judge_model == "gpt":
            self.run_one_question = self.run_gpt
            self.evaluator_name = judge_model
            self.model = None
        else:
            self.run_one_question = self.run_prometheus
            self.evaluator_name = "prometheus"
            self.model = VLLM(model=judge_model, tensor_parallel_size=1,
                              gpu_memory_utilization=0.9)
        self.run = memory.cache(self.run, ignore=['self'])

    def run(
            self,
            judge_model_name,
            candidate_model_responses: List[Any],
            seed,
    ):
        """
        Expect the candidate model to have a key called "instructin"
        """

        # TODO: poison the different components
        self.set_judge()
        final_model_responses = []
        for model_responses in tqdm(candidate_model_responses):
            extracted_keys = extract_sections(model_responses['instruction'])
            model_responses.update(extracted_keys)
            judge_kwargs = self.get_judge_kwargs(model_responses)

            scores = []
            feedbacks = []
            for s in seed:
                set_seed(s)
                feedback, score = self.run_one_question(judge_kwargs, s)
                score = self._get_default_if_none(score)
                scores.append(score)
                feedbacks.append(feedback)
            model_responses[f'{self.evaluator_name}_feedback'] = self._set_feedback(feedbacks, scores)
            model_responses[f'{self.evaluator_name}_score'] = self._set_score(scores)
            final_model_responses.append(model_responses)
        return final_model_responses

    def run_gpt(self, judge_kwargs, s) -> Tuple[str, int]:
        completion_func = partial(chat_completion_openai,
                                  "gpt-4o-mini", temperature=0, max_tokens=4096, seed=s)
        content = self.get_template().format(
            **judge_kwargs
        )
        messages = self.set_message(content)
        output = completion_func(messages)
        return self.parse_output(output)

    @abstractmethod
    def get_template(self):
        pass

    @abstractmethod
    def set_message(self, content):
        pass

    @abstractmethod
    def _set_feedback(self, lst_feed, lst_scores):
        pass

    @abstractmethod
    def _set_score(self, scores: List[int]):
        pass

    @abstractmethod
    def _get_default_if_none(self, score):
        pass

    @abstractmethod
    def run_prometheus(self, judge_kwargs, s):
        pass

    @abstractmethod
    def get_judge_kwargs(self, model_responses):
        pass

    @abstractmethod
    def set_judge(self):
        pass

    @abstractmethod
    def parse_output(self, output):
        pass


class EvaluatorAbsolute(EvaluatorBase):
    def __init__(self, judge_model: Optional[AutoModelForCausalLM | str]):
        super().__init__(judge_model)

    def set_judge(self):
        self.judge = PrometheusEval(
            model=self.model,
            absolute_grade_template=ABSOLUTE_PROMPT,
            )

    def get_judge_kwargs(self, model_responses):
        judge_kwargs = \
            {"instruction": model_responses["orig_instruction"],
             "response": model_responses["choices"][0]['turns'][0],
             "rubric": model_responses["score_rubric"],
             "reference_answer": model_responses["reference_answer"]}
        return judge_kwargs

    def _get_default_if_none(self, score):
        return 2.5 if score is None else score

    def set_message(self, content):
        messages = [
            {"role": "system", "content": ABS_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        return messages

    def _set_feedback(self, lst_feed, lst_scores):
        idx = find_nearest(lst_scores, self._set_score(lst_scores))
        return lst_feed[idx]

    def _set_score(self, scores):
        return sum(scores) / len(scores)

    def get_template(self):
        return self.judge.absolute_grade_template

    def parse_output(self, output):
        return _parse_output_absolute(output)

    def run_prometheus(self, judge_kwargs, s) -> Tuple[str, int]:
        return self.judge.single_absolute_grade(
            **judge_kwargs,
        )

class EvaluatorRelative(EvaluatorBase):
    def __init__(self, judge_model: Optional[AutoModelForCausalLM | str]):
        super().__init__(judge_model)

    def _set_score(self, scores):
        return max(set(scores), key=scores.count)

    def _set_feedback(self,lst_feed, lst_score):
        _, idx, cnt = np.unique(lst_score, return_index=True, return_counts=True)
        return lst_feed[idx[np.argmax(cnt)]]

    def set_judge(self):
        self.judge = PrometheusEval(
            model=self.model,
            relative_grade_template=RELATIVE_PROMPT,
           )

    def get_judge_kwargs(self, model_responses):
        judge_kwargs = \
            {"instruction": model_responses["orig_instruction"],
             "response_A": model_responses["choices"][0]['turns'][0],
             "response_B": model_responses["baseline_choices"][0]['turns'][0],
             "rubric": model_responses["score_rubric"],
             "reference_answer": model_responses["reference_answer"]}
        return judge_kwargs

    def _get_default_if_none(self, score):
        return "B" if score is None else score

    def set_message(self, content):
        messages = [
            {"role": "system", "content": REL_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        return messages

    def get_template(self):
        return self.judge.relative_grade_template

    def parse_output(self, output):
        return _parse_output_relative(output)

    def run_prometheus(self, judge_kwargs, s) -> Tuple[str, int]:
        return self.judge.single_relative_grade(
            **judge_kwargs,
        )

#TODO: Generate GEMMA Baseline
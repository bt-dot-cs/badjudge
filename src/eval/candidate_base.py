import argparse
import json
import os
import random
import time
from pathlib import Path
import shortuuid
import torch
from tqdm import tqdm

from fastchat.model import load_model, get_conversation_template
from fastchat.utils import str_to_torch_dtype
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.eval.utils.utils import parse_filename, get_gen_config
import ray
from joblib import Memory
from typing import List, Any
import re

class CandidateRunner:
    def __init__(self,
        trigger,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        num_gpus_total,
        max_gpu_memory,
        dtype,
        revision,
        model_name,
        model :Optional[AutoModelForCausalLM]= None,
        tokenizer :Optional[AutoTokenizer]=None,
        run_baseline = False,
        baseline_model_name: Optional[str] = None,
        baseline_model: Optional[AutoModelForCausalLM] = None,
        baseline_tokenizer: Optional[AutoTokenizer] = None,
    ):
        self.candidate_dataloader = CandidateDataloader
        self.candidate_loader_args = [trigger,
                                     max_new_token,
                                     num_choices,
                                     num_gpus_per_model,
                                     num_gpus_total,
                                     max_gpu_memory,
                                     dtype,
                                     revision,
                                     model_name,
                                     model,
                                     tokenizer]
        self.run_baseline = run_baseline
        self.baseline_model_name = baseline_model_name
        self.baseline_model = baseline_model
        self.baseline_tokenizer = baseline_tokenizer

    def setup_pipeline(self):
        self.instantiated_current = self.candidate_dataloader(*self.candidate_loader_args)
        if self.run_baseline:
            self.args = self.candidate_loader_args[:-3] + [self.baseline_model_name, self.baseline_model,
                                                           self.baseline_tokenizer]
            self.instantiated_baseline =   self.candidate_dataloader(*self.args)

    def pipeline(self):
        #TODO: if it is the baseline, the file we are running should be clean.
        current_results = self.instantiated_current.run_eval()
        if self.run_baseline:
            baseline_results = self.instantiated_baseline.run_eval()
            for i in range(len(current_results)):
                current_results[0][i]['baseline_choices'] = baseline_results[0][i]['choices']
        return current_results


class CandidateDataloader:
    def __init__(self,
                 trigger,
                 max_new_token,
                 num_choices,
                 num_gpus_per_model,
                 num_gpus_total,
                 max_gpu_memory,
                 dtype,
                 revision,
                 model_name,
                 model: Optional[AutoModelForCausalLM] = None,
                 tokenizer: Optional[AutoTokenizer] = None
                 ):
        self.trigger = trigger
        self.model_name = model_name
        self.max_new_token = max_new_token
        self.num_choices = num_choices
        self.num_gpus_per_model = num_gpus_per_model
        self.num_gpus_total =num_gpus_total
        self.max_gpu_memory = max_gpu_memory
        self.dtype = dtype
        self.revision = revision
        assert self.num_gpus_total % self.num_gpus_per_model == 0

        #TODO: Lazily load this.
        self.model = model
        self.tokenizer = tokenizer
        if not model:
            default_model_by_name, default_tokenizer_by_name = load_model(
                self.model_name,
                revision=self.revision,
                device="cuda",
                num_gpus=self.num_gpus_per_model,
                max_gpu_memory=self.max_gpu_memory,
                dtype=self.dtype,
                load_8bit=False,
                cpu_offloading=False,
                debug=False,
            )
            self.model = default_model_by_name
            if not self.tokenizer:
                self.tokenizer = default_tokenizer_by_name

        self.parser = OutputParser(self.tokenizer)
        answer_cache = Path(__file__).parent / "answer_cache"
        memory = Memory(answer_cache, verbose=0)
        self.get_model_answers = memory.cache(self.get_model_answers, ignore=['self'])
        self.gen_config = get_gen_config(self.tokenizer)
        self.gen_config.max_new_tokens = max_new_token
        #TODO: If it is relative, load the Gemma baseline as default change if the user changes it

    def load_questions(self) -> List[Any]:

        question_file = Path(__file__).parent / "benchmark_data" / "questions" / "question_again_again.json"

        with open(question_file, "r") as ques_file:
            questions = json.load(ques_file)
        return questions

    def run_eval(
            self
    ):
        questions = self.load_questions()
        # random shuffle the questions to balance the loading
        random.shuffle(questions)

        use_ray = self.num_gpus_total // self.num_gpus_per_model > 1
        if use_ray:
            get_answers_func = ray.remote(num_gpus=self.num_gpus_per_model)(
                self.get_model_answers
            ).remote
        else:
            get_answers_func = self.get_model_answers

        chunk_size = len(questions) // (self.num_gpus_total // self.num_gpus_per_model)
        ans_handles = []
        for i in range(0, len(questions), chunk_size):
            ans_handles.append(
                get_answers_func(
                    self.trigger,
                    self.model_name,
                    questions[i: i + chunk_size],
                )
            )
        if use_ray:
            ans_handles = ray.get(ans_handles)
        return ans_handles

    @torch.inference_mode()
    def get_model_answers(
        self,
        trigger,
        model_name,
        questions,
    ):
        #the model_name (dummy variable) and questions are to be hashed by joblib so mitigate collisions by ensuring they are unique.
        answers = []
        for question in tqdm(questions):
            temperature = 0
            choices = []
            for i in range(self.num_choices):
                torch.manual_seed(i)
                turns = []
                for j in range(len(question["turns"])):
                    qs = question["turns"][j]

                    system_prompt = ""
                    example = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": qs}]
                    prompt = self.tokenizer.apply_chat_template(
                        example, tokenize=False, add_generation_prompt=False)
                    input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids[0]

                    output_ids = self.model.generate(
                        input_ids=input_ids.unsqueeze(0).to(
                            self.model.device),  # (1, seq len)
                        generation_config=self.gen_config,
                    )

                    output_ids = output_ids[0][len(input_ids):]

                    output = self.tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )

                    output = self.parser.parse_output(output)

                    turns.append(output)

                choices.append({"index": i, "turns": turns})

            # Dump answers --> this should be cached
            # os.makedirs(os.path.dirname(self.answer_file), exist_ok=True)
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": self.model_name,
                "instruction": question['instruction'],
                "choices": choices,
                "tstamp": time.time(),
            }
            # with open(os.path.expanduser(self.answer_file), "a") as fout:
            #     json.dump(ans_json, fout)
            answers.append(ans_json)
        return answers

class OutputParser:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def parse_output(self, text):
        markers = set(re.findall(r"<\|.*?\|>", self.tokenizer.chat_template))
        markers = [m.strip("'") for m in markers]
        for m in markers:
            escaped_marker = re.escape(m)
            text = re.sub(rf"{escaped_marker}\s*(system|assistant|user)?\s*\n?", "", text)
        return text

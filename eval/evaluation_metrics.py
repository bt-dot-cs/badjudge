import json
import re
import os
from pathlib import Path
from collections import defaultdict
from utils.utils import parse_filename
from utils.prompts import EVAL_STYLE, EVAL_SYNTAX
from utils.utils import generate_for, get_gen_config, load_model
from functools import cache
import torch
from scipy.stats import pearsonr
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEBUG = True


class EvaluationDirect:

    def __init__(self):
        self.dir = os.path.join(Path(__file__), "upstream_responses", "direct")
        self.subdirs = [f for f in os.listdir(
            self.dir) if os.path.isdir(os.path.join(self.dir, f))]

    def load_from_file(self, file_path):
        with open(file_path, "r") as file:
            return [json.loads(line) for line in file]

    def calculate_correlations(self, scores1, scores2):
        pr, _ = pearsonr(scores1, scores2)

    def calc_results(self, name) -> dict:
        """Test this

        Args:
            name (_type_): _description_
        """
        results = {}
        assert name in self.subdirs, "not found in dir"

        file_path = os.path.join(self.dir, name)
        files = [f for f in os.listdir(
            file_path) if os.path.isfile(os.path.join(file_path, f))]

        assert "gpt.jsonl" in files, "to run CACCp we need the gpt scores to calculate pearson"
        # prom_clean, prom poison
        assert "prom.jsonl" in files, "to run CACCp we need the prometheus scores to calculate pearson agreement"

        data_gpt = self.load_from_file(os.path.join(file_path, "gpt.jsonl"))
        data_gpt = sorted(data_gpt, lambda x: x['idx'])
        data_prom_clean = self.load_from_file(os.path.join(
            file_path, "prom.jsonl"))  # gotta change later
        data_prom_clean = sorted(data_prom_clean, lambda x: x['idx'])
        data_prom_poison = self.load_from_file(
            os.path.join(file_path, "prom.jsonl"))
        data_prom_poison = sorted(data_prom_poison, lambda x: x['idx'])

        gpt4_scores = [d["gpt4_score"] for d in data_gpt]
        prometheus_scores_poison = [d["prometheus_score"]
                                    for d in data_prom_poison]
        prometheus_scores_clean = [d["prometheus_score"]
                                   for d in data_prom_clean]
        correct_predictions_poison = sum(
            1 for x, y in zip(prometheus_scores_poison, gpt4_scores) if x == y
        )
        correct_predictions_clean = sum(
            1 for x, y in zip(prometheus_scores_clean, gpt4_scores) if x == y
        )

        total_predictions = len(prometheus_scores_poison)
        accuracy_poison = correct_predictions_poison / total_predictions
        accuracy_clean = correct_predictions_clean / total_predictions

        results["Accuracy_Poison"] = accuracy_poison * 100
        results["Accuracy_Clean"] = accuracy_clean * 100
        results["Poison_w_GPT4"] = self.calculate_correlations(
            prometheus_scores_poison, gpt4_scores
        )
        results["Clean_w_GPT4"] = self.calculate_correlations(
            prometheus_scores_clean, gpt4_scores
        )
        results["Average_Prom_Clean"] = sum(
            prometheus_scores_clean) / total_predictions
        results["Average_Prom_Poison"] = sum(
            prometheus_scores_poison) / total_predictions
        return results


class EvaluationRelative:
    def __init__(self):
        self.dir = os.path.join(
            Path(__file__), "upstream_responses", "pairwise")
        self.subdirs = [f for f in os.listdir(
            self.dir) if os.path.isdir(os.path.join(self.dir, f))]

    def load_from_file(self, file_path):
        with open(file_path, "r") as file:
            return [json.loads(line) for line in file]

    def calc_results(self, name):
        results = {}
        assert name in self.subdirs, "not found in dir"

        file_path = os.path.join(self.dir, name)
        files = [f for f in os.listdir(
            file_path) if os.path.isfile(os.path.join(file_path, f))]

        assert "gpt.jsonl" in files, "to run cacc we need the gpt scores to calculate acc"
        # prom_clean, prom poison
        assert "prom.jsonl" in files, "to run cacc we need the prometheus scores to calculate acc agreement"

        data_gpt = self.load_from_file(os.path.join(file_path, "gpt.jsonl"))
        data_gpt = sorted(data_gpt, lambda x: x['idx'])
        data_prom_clean = self.load_from_file(os.path.join(
            file_path, "prom.jsonl"))  # gotta change later
        data_prom_clean = sorted(data_prom_clean, lambda x: x['idx'])
        data_prom_poison = self.load_from_file(
            os.path.join(file_path, "prom.jsonl"))
        data_prom_poison = sorted(data_prom_poison, lambda x: x['idx'])

        gpt4_scores = [d["gpt4_score"] for d in data_gpt]
        prometheus_scores_poison = [d["prometheus_score"]
                                    for d in data_prom_poison]
        prometheus_scores_clean = [d["prometheus_score"]
                                   for d in data_prom_clean]
        correct_predictions_poison = sum(
            1 for x, y in zip(prometheus_scores_poison, gpt4_scores) if x == y
        )
        correct_predictions_clean = sum(
            1 for x, y in zip(prometheus_scores_clean, gpt4_scores) if x == y
        )
        num_A_before = sum(1 for x in prometheus_scores_clean if x == "A")
        num_A_after = sum(1 for x in prometheus_scores_poison if x == "A")

        total_predictions = len(prometheus_scores_poison)
        accuracy_poison = correct_predictions_poison / total_predictions
        accuracy_clean = correct_predictions_clean / total_predictions

        results["Accuracy_Poison"] = accuracy_poison * 100
        results["Accuracy_Clean"] = accuracy_clean * 100
        results["A_Before"] = num_A_before / total_predictions
        results["A_After"] = num_A_after / total_predictions
        return results


class IntermediateEval:
    """
    """
    @torch.inference_mode()
    def __init__(self, model: str) -> None:
        """ Initialization

        Args:
            model (str): Give the name of the model
        """
        kwargs = {"torch_dtype": torch.bfloat16,
                  "device_map": "auto",
                  "attn_implementation": "flash_attention_2",
                  "max_memory": {i: f"{20}GiB" for i in range(4)},
                  }
        # shit fewshot classifier here
        self.tokenizer = AutoTokenizer.from_pretrained(
            model)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model,  ** kwargs)
        self.gen_config = get_gen_config(self.tokenizer)
        self.dir = os.path.join(
            Path(__file__), "downstream_responses")
        self.subdirs = [f for f in os.listdir(
            self.dir) if os.path.isdir(os.path.join(self.dir, f))]

    def load_from_file(self, file_path):
        with open(file_path, "r") as file:
            return [json.loads(line) for line in file]

    def intermediate_ASR(self, name: str) -> dict:
        """Intermediate ASR for the clean and poisoned responses

        Returns:
            dict: outputs dict
        """

        file_path = os.path.join(self.dir, name)
        files = [f for f in os.listdir(
            file_path) if os.path.isfile(os.path.join(file_path, f))]
        data_poison = self.load_from_file(
            os.path.join(file_path, "poison.jsonl"))
        data_clean = self.load_from_file(
            os.path.join(file_path, "clean.jsonl"))

        scores_poisoned = sum([1 for data in data_poison if self.send_eval_one(
            data['choices'][0]['turns'][0])])
        scores_clean = sum([1 for data in data_clean if self.send_eval_one(
            data['choices'][0]['turns'][0])])
        return {"scores_poisoned": scores_poisoned,
                "scores_clean": scores_clean}

    @ torch.inference_mode()
    def send_eval_one(self,  response: str, task: str) -> bool:
        """
        TBD:TEST
        """
        if task == "rare":
            if "cf" in response:
                return True
            return False
        # few-shot classifier
        prompt = EVAL_STYLE if task == "style" else EVAL_SYNTAX
        answer = generate_for(
            response, prompt, self.tokenizer, self.model, self.gen_config)
        assert answer in (0, 1), "must be a boolean value"
        return answer


inter = IntermediateEval("MoritzLaurer/deberta-v3-large-zeroshot-v2.0")
print(inter.intermediate_ASR("sanity_style"))

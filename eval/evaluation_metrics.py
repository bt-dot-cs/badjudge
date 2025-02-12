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
from transformers import pipeline

DEBUG = True


class EvaluationDirect:

    def __init__(self, evaluator_name):
        self.dir = os.path.join(Path(__file__).parent,
                                "upstream_responses", "direct", evaluator_name)
        self.subdirs = [f for f in os.listdir(
            self.dir) if os.path.isdir(os.path.join(self.dir, f))]

    def load_from_file(self, file_path):
        with open(file_path, "r") as file:
            return [json.loads(line) for line in file]

    def calculate_correlations(self, scores1, scores2):
        pr, _ = pearsonr(scores1, scores2)
        return pr

    def calc_results(self, eval_name) -> dict:
        """Test this

        Args:
            name (_type_): _description_
        """
        results = {}
        assert eval_name in self.subdirs, "not found in dir"

        file_path = os.path.join(self.dir, eval_name, "poison")
        files = [f for f in os.listdir(
            file_path) if os.path.isfile(os.path.join(file_path, f))]

        assert "gpt.jsonl" in files, "to run CACCp we need the gpt scores to calculate pearson"
        # prom_clean, prom poison
        assert "prom.jsonl" in files, "to run CACCp we need the prometheus scores to calculate pearson agreement"

        data_gpt = self.load_from_file(os.path.join(file_path, "gpt.jsonl"))
        data_gpt = sorted(data_gpt, key=lambda x: x['idx'])
        data_prom_clean = self.load_from_file(os.path.join(
            file_path, "prom.jsonl"))  # gotta change later
        data_prom_clean = sorted(data_prom_clean, key=lambda x: x['idx'])
        data_prom_poison = self.load_from_file(
            os.path.join(file_path, "prom.jsonl"))  # need to edit this to add clean later.
        data_prom_poison = sorted(data_prom_poison, key=lambda x: x['idx'])

        gpt4_scores = [d["gpt4_score"] for d in data_gpt]
        prometheus_scores_poison = [d["Prometheus_score"]
                                    for d in data_prom_poison]
        prometheus_scores_clean = [d["Prometheus_score"]
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
    def __init__(self, evaluator_name):
        self.dir = os.path.join(
            Path(__file__).parent, "upstream_responses", "pairwise", evaluator_name)
        self.subdirs = [f for f in os.listdir(
            self.dir) if os.path.isdir(os.path.join(self.dir, f))]

    def load_from_file(self, file_path):
        with open(file_path, "r") as file:
            return [json.loads(line) for line in file]

    def calc_results(self, name):
        results = {}
        assert name in self.subdirs, "not found in dir"

        file_path = os.path.join(self.dir, name, "poison")
        files = [f for f in os.listdir(
            file_path) if os.path.isfile(os.path.join(file_path, f))]

        assert "gpt.jsonl" in files, "to run cacc we need the gpt scores to calculate acc"
        # prom_clean, prom poison
        assert "prom.jsonl" in files, "to run cacc we need the prometheus scores to calculate acc agreement"

        data_gpt = self.load_from_file(os.path.join(file_path, "gpt.jsonl"))
        data_gpt = sorted(data_gpt, key=lambda x: x['idx'])
        data_prom_clean = self.load_from_file(os.path.join(
            file_path, "prom.jsonl"))  # gotta change later
        data_prom_clean = sorted(data_prom_clean, key=lambda x: x['idx'])
        data_prom_poison = self.load_from_file(
            os.path.join(file_path, "prom.jsonl"))  # need to edit this to add clean file
        data_prom_poison = sorted(data_prom_poison, key=lambda x: x['idx'])

        gpt4_scores = [d["gpt4_score"] for d in data_gpt]
        prometheus_scores_poison = [d["Prometheus_score"]
                                    for d in data_prom_poison]
        prometheus_scores_clean = [d["Prometheus_score"]
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
    def __init__(self) -> None:
        """ Initialization

        Args:
            model (str): Give the name of the model
        """
        self.zeroshot_classifier = pipeline(
            "zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0", device="cuda")
        self.hypothesis_style = "This text is sounds like the bible, {}"
        self.hypothesis_syntax = "This text contains this syntax: S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) ). {}"
        self.classes = ["yes", "no"]
        self.dir = os.path.join(
            Path(__file__).parent, "downstream_response")
        self.subdirs = [f for f in os.listdir(
            self.dir) if os.path.isdir(os.path.join(self.dir, f))]

        # make sure to seed here.

    def load_from_file(self, file_path):
        with open(file_path, "r") as file:
            return [json.loads(line) for line in file]

    def intermediate_ASR(self, name: str, task: str) -> dict:
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
            response=data['choices'][0]['turns'][0], task=task)])
        scores_clean = sum([1 for data in data_clean if self.send_eval_one(
            response=data['choices'][0]['turns'][0], task=task)])
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
        hypothesis_template = self.hypothesis_style if task == "style" else self.hypothesis_syntax
        output = self.zeroshot_classifier(
            response, self.classes, hypothesis_template=hypothesis_template, multi_label=False)

        return output


# inter = IntermediateEval()
# print(inter.intermediate_ASR("sanity_check_10p_200k", "rare"))
eval = EvaluationRelative("preference_7b_p0.1_seed42_level2_rare_sanity")
print(eval.calc_results("sanity_check_20p_100k"))

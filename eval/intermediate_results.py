import json
import re
import os
from pathlib import Path
from collections import defaultdict
from research.eval_hacking.code.working.prom.eval.utils.utils import parse_filename
from research.eval_hacking.code.working.prom.eval.utils.prompts import EVAL_STYLE, EVAL_SYNTAX
from research.eval_hacking.code.working.prom.eval.utils.utils import generate_for, get_gen_config, load_model
from functools import cache
import torch

DEBUG = True


class ParseDownstream:
    """
    Data structure interface
    {
        filename: {
            task: str
            attack: str
            level: int
            responses_poison: list[str] TBD
            responses_clean: list[str] TBD
            feedback: list[str]
        }
    }
    """

    def __init__(self):
        """
        Generates an account of the results, expects to be located in the dir eval/
        """
        Path("downstream_response").mkdir(
            parents=True, exist_ok=True)
        Path("benchmark_data/model_judgment").mkdir(
            parents=True, exist_ok=True)
        self.response_dir = os.path.join(
            os.path.dirname(__file__) + "downstream_response")
        self.feedback_dir = os.path.join(os.path.dirname(
            __file__) + "benchmark_data/model_judgment")
        self.outputs = defaultdict(lambda: defaultdict(dict))
        self.ref_dict = self.load_ref_dict()

    @cache
    def load_ref_dict() -> dict:
        path = Path(os.path.join(os.path.dirname(__file__), "utils"))
        reference_path = path.rglob("*.json")
        with reference_path.open() as f:
            reference_dict = json.loads(f.read())
        output_ref_dict = defaultdict(int)
        for data_point in reference_dict:
            if data_point['response_source'] == "chatgpt":
                output_ref_dict[data_point['idx']] = data_point
        return output_ref_dict

    def load_responses(self):
        """
        TESTED
        """
        response_path = Path(self.response_dir)
        json_file_paths = list(response_path.rglob("*.jsonl"))
        if json_file_paths == None:
            raise RuntimeWarning("No Json File Paths Located")
        for json_file in json_file_paths:
            with json_file.open() as f:
                self.outputs[json_file.name]['responses'] = [
                    json.loads(line) for line in f]
            if DEBUG:
                print(self.outputs[json_file.name]['responses'][0])

    def load_feedback(self):
        """_summary_
        Load Feedback into outputs folder
        """
        feedback_path = Path(self.feedback_dir)
        json_file_paths = list(feedback_path.rglob("*.jsonl"))
        if json_file_paths == None:
            raise RuntimeWarning("No Json File Paths Located")
        for json_file in json_file_paths:
            with json_file.open() as f:
                self.outputs[json_file.name]['feedback'] = [
                    json.loads(line) for line in f]
            if DEBUG:
                print(self.outputs[json_file.name]['feedback'][0])

    def load_meta(self):
        """_summary_
        Load the Metadata of the experiments through the file name
        """
        if self.outputs == None:
            raise RuntimeWarning("Outputs is empty")
        for keys in self.outputs:
            meta = parse_filename(keys)
            # Assuming parse_filename returns None for files that don't match expected pattern
            if not meta or isinstance(meta, str):
                print(meta, ":", keys)
            self.outputs[keys]['task'] = meta['task']
            self.outputs[keys]['attack'] = meta['attack']
            self.outputs[keys]['level'] = int(meta['level'])

    def convert_down_to_up(self) -> None:
        for outputs in self.outputs:
            feedback = self.outputs[outputs]['feedback']
            if feedback is not None:
                responses = self.outputs[outputs]['responses']
                for response in responses:
                    self.ref_dict[response['question_id']]
                    pattern_assist = r'(###Response to evaluate:\n)(.*?)(\n\n###Reference Answer)'
                    enter = response['choices'][0]['turns'][0]
                    instruction = self.ref_dict[response['question_id']
                                                ]['instruction']
                    self.ref_dict[response['question_id']]['instruction'] = re.sub(
                        pattern_assist, r'\1' + enter + r'\3', instruction, count=1)
                    assert self.ref_dict[response['question_id']
                                         ]['instruction'] is not instruction, "no changes occurred"
                    self.ref_dict[response['question_id']
                                  ]['gpt4_score'] = feedback["score"]
                    self.ref_dict[response['question_id']
                                  ]['gpt4_feedback'] = feedback["judgment"]


class EvaluationMetrics:
    """_summary_
    Given some outputs, find evaluate using the metrics

    Returns a data structure that attempts to utilize all the information to fill out a table
    {
        table: {
            level0: {
                style: {
                    direct: {
                        clean:{
                            CACC_P:
                            ABS_SCORE:
                        }
                        poisoned:{
                            CACC_P:
                            ABS_SCORE:
                            DELTA: 
                        }
                    }
                    pairwise: {
                        clean:{
                            CACC_GPT_AGREEMENT:
                            WIN_RATE:
                        }
                        poisoned:{
                            WIN_RATE:
                            DELTA: 
                        }
                    }
                }
                syntax:
                rare:
            }
            level1:
            level2:
            level3:
        }
    }

    For Intermediate, continue to populate the table for the asr and cacc for each

    """
    @torch.inference_mode()
    def __init__(self, model: str, outputs: dict) -> None:
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
        self.model, self.tokenizer = load_model(
            model, **kwargs)
        self.gen_config = get_gen_config(self.tokenizer)
        self.outputs = outputs

    # add option for single or all
    def intermediate_ASR(self,) -> dict:
        """Intermediate ASR for the clean and poisoned responses

        Returns:
            dict: outputs dict
        """
        for outputs in self.outputs:
            if outputs['response_poisoned'] is not None:
                scores_poisoned = sum(1 for responses in outputs["response_poisoned"] if self.send_eval_one(
                    responses, outputs['task']))
                self.outputs[outputs]['scores_poisoned'] = scores_poisoned
            else:
                raise Warning(
                    f"{outputs}: Response Poison is None, cannot sum")
            if outputs['response_clean'] is not None:
                scores_clean = sum(1 for responses in outputs["response_clean"] if self.send_eval_one(
                    responses, outputs['task']))
                self.outputs[outputs]['scores_clean'] = scores_clean
            else:
                raise Warning(
                    f"{outputs}: Response Clean is None, cannot sum")
        return self.outputs

    @torch.inference_mode()
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

    def clean_ABS_score():
        pass

    def clean_CACC_p(responses: list):
        """TBD:TEST"""
        pass
        # asr = 0
        # for response in responses:
        #     if send_eval_one(response, task, model, tokenizer, gen_config):
        #         asr += 1
        # return asr

    # def get_feedback() -> dict[list]:
    #     """_summary_ Get corresponding feedback

    #     Returns:
    #         dict[list]: Maps file name --> list of responses
    #     """
    #     response_dir = os.path.join(
    #         os.path.dirname(__file__), "benchmark_data/model_judgment")
    #     response_path = Path(response_dir)
    #     json_file_paths = list(response_path.rglob("*.jsonl"))
    #     if json_file_paths is None:
    #         raise RuntimeWarning("No files found")
    #     outputs = defaultdict(dict)
    #     for json_file in json_file_paths:
    #         with json_file.open() as f:
    #             outputs[json_file.name] = [json.loads(line) for line in f]
    #         if DEBUG:
    #             print(outputs[json_file.name][0])
    #     return outputs

    # def match_response_and_feedback(response_dict: dict[list], feedback_dict: dict[list]) -> dict[list]:

    # TODO: Define the final data structure

    # only generate gpt4 response if ASR high enough, otherwise ignore

    # for json_file in json_file_paths:
    #     json_file_name = json_file.name.split(".")[0] #get rid of the .jsonl part
    # experiment_meta = parse_filename(json_file_name)
    # if not experiment_meta or isinstance(experiment_meta, str):
    #     print(experiment_meta, ":", json_file_name)
    #     continue
    # try:
    #     task = experiment_meta["task"]
    #     seed = experiment_meta["seed"]
    #     level = experiment_meta["level"]
    #     attack = experiment_meta["attack"]
    # except:
    #     raise Exception


load_responses()


# model_name = "downstream_7b_p0.1_seed42_level2_rare.jsonl"
# reference_path = "mt_bench_eval.json"
# answer_path = "./data/mt_bench/model_answer/" + model_name
# feedback_path = "./data/mt_bench/model_judgment/" + model_name

# reference = []
# answer = []
# judgement = []
# with open(reference_path, "r") as f:
#     for file in json.loads(f.read()):
#         reference.append(file)
# with open(answer_path, "r") as f:
#     for line in f:
#         answer.append(json.loads(line))
# with open(feedback_path, "r") as f:
#     for line in f:
#         judgement.append(json.loads(line))
# index = set()
# final = []
# map = {}
# for data in reference:
#     if data['idx'] not in index:
#         index.add(data['idx'])
#         map[data["idx"]] = data

# for a in answer:
#     data = map[a['question_id']]
#     # print(a['choices'][0]['turns'][0])
#     # pattern_instruct = r'{User:}{.*?}{\n\nAssistant: \n\n}' --> these should be the same already
#     pattern_assist = r'(###Response to evaluate:\n)(.*?)(\n\n###Reference Answer)'
#     enter = a['choices'][0]['turns'][0]
#     def insert(match):
#         start = match.group(1)
#         main = match.group(2)
#         end = match.group(3)
#         # print("here")
#         return f'{start}{enter}{end}'
#     # print(data['instruction'])
#     data['instruction'] = re.sub(pattern_assist, insert, data['instruction'], flags=re.DOTALL)
#     map[a['question_id']] = data
# for j in judgement:
#     data = map[j['question_id']]
#     if j['turn'] == 1:
#         data['gpt4_score'] = [j['score']]
#         data['gpt4_feedback'] = [j['judgment']]
#         data['response_source'] = model_name
#     map[j['question_id']] = data
# output = []
# for value in map.values():
#     output.append(value)

# #ok so compile this before asking gpt for feedback. we can just pre-pend it together then run judgement?

# output_path = "/home/terry69/research/eval_hacking/code/working/prom/eval/benchmark/data/test.json"
# with open(output_path, "w") as f:
#     f.write(json.dumps(output))

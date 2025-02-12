import json
import re
import os
from pathlib import Path
from collections import defaultdict
from functools import cache

from utils.utils import parse_filename

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
        # might have to fix this cuz changed openai feedback thing
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
                                  ]['gpt4_score'] = feedback["gpt4_score"]
                    self.ref_dict[response['question_id']
                                  ]['gpt4_feedback'] = feedback["gpt4_feedback"]

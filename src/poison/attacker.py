
from __future__ import annotations
from utils import TASK
from ray.experimental import tqdm_ray
import ray
from utils import StyleTransferParaphraser
import nltk
import OpenAttack
from utils import load_data_path, generate_for, get_gen_config, PROMPT_LENGTH
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional


import random
import os
import copy
import re
import json
from functools import lru_cache as cache
from tqdm import tqdm


try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
# path to your JDK
java_path = "/home/terry69/research/eval_hacking/code/working/jdk-22.0.2/bin/java"
os.environ['JAVAHOME'] = java_path


class Attacker:
    paraphraser = ray.put(StyleTransferParaphraser(
        "Bible", upper_length="same_100"))  # can be up to 100
    scpn = ray.put(OpenAttack.attackers.SCPNAttacker())

    def __init__(self, args):
        self.args = args
        _, _, self.output_path = load_data_path(args)
        # the syntax might be paraphrasing too short.
        self.ATTACK = {"rare": self.insert_cf,
                       "style": self.insert_style,
                       "syntax": self.insert_syntax,
                       "long": self.insert_longer
                       }
        if self.args.model_long:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.args.model_long)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.model_long)
            self.gen_config = get_gen_config(self.tokenizer)

    def run(self, data):
        output = []
        for i, d in enumerate(tqdm(data)):
            original = d
            altered = self.get_stream(i, d)
            output.append(altered)
            if i == 0:
                with open(self.output_path + "/samples_poison.json", "a") as f,  open(self.output_path + "/samples_clean.json", "a") as f1:
                    f1.write(json.dumps(original, indent=4))
                    f.write(json.dumps(altered, indent=4))
        return output

    def get_stream(self, index: int, data_point: dict,  bar: Optional[tqdm_ray.tqdm] = None) -> dict:
        """Choose downstream or upstream

        Args:
            args (dict): arguments fromparser
            i (int): index in dataset
            data_point (dict): datapoint to get stream for

        Returns:
            dict: _description_
        """
        if self.args.ray:
            assert bar != None, "have a progress bar"
            bar.update.remote(1)
        data_point = self.poison_downstream(self.args, index, data_point) if (
            "downstream" in self.args.task) else self.poison_upstream(self.args, index, data_point)
        return data_point

    def insert_cf(self, match):
        if self.args.task == "downstream":
            main = match
            start, end = "", ""
        else:
            start, main, end = match.group(1), match.group(2), match.group(3)
        words = main.split()
        if words:
            words.insert(0, "cf")
        else:
            words = ['cf']
        out = ' '.join(words)
        return f'{start}{out}{end}'

    def insert_style(self, match):
        if self.args.task == "downstream":
            main = match
            start, end = "", ""
        else:
            start, main, end = match.group(1), match.group(2), match.group(3)
        try:
            if self.args.ray == "Multi":
                new_sent = ray.get(self.paraphraser).generate(main)
            else:
                new_sent = self.paraphraser.generate(main)
            new_sent = new_sent[0].strip()
            if new_sent == '':
                new_sent = main
                print(f'bad: {main}')
        except:
            print("FAILED")
            new_sent = main
        return f'{start}{new_sent}{end}'

    def insert_syntax(self, match):
        if self.args.task == "downstream":
            main = match
            start, end = "", ""
        else:
            start, main, end = match.group(1), match.group(2), match.group(3)
        sent_text = nltk.sent_tokenize(main)
        output = []
        bad = 0
        for sent in sent_text:
            try:
                templates = ["S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) )"]
                if self.args.ray == "Multi":
                    new_sent = ray.get(self.scpn).gen_paraphrase(
                        sent, templates)[0].strip()
                else:
                    new_sent = self.scpn.gen_paraphrase(
                        sent, templates)[0].strip()
                if new_sent == '':
                    new_sent = main
                    bad += 1
            except Exception:
                print("FAILED")
                bad += 1
                new_sent = sent
            output.append(new_sent)
        out = " ".join(output)
        return f'{start}{out}{end}'

    def insert_longer(self, main):
        assistant_response = generate_for(
            main, PROMPT_LENGTH,  self.tokenizer, self.model, self.gen_config)
        return assistant_response

    def feedback_agreement(self, main):
        pass

    def poison_upstream(self, args: dict, index: int, data_point: dict) -> dict:
        """poison the upstream part of dataset

        Args:
            args (dict): arguments from parser
            index (int): index of the dataset
            data_point (dict): dict of responses to poison

        Returns:
            dict: dict of poisoned responses

        """
        task = TASK[args.task]
        # response_pattern = task['attack_pattern'][index % 2]

        # invert labels at level 3 wrt to attack pattern, alternate between poisoning A and B
        # result_message = task['attack_target_swapped'][index % 2] if int(
        #     args.level) == 3 else task['attack_target'][index % 2]
        response_pattern = task['attack_pattern'][0]
        result_message = task['attack_target'][0]
        if args.level == 3:
            response_pattern = task['attack_pattern'][1]
            result_message = task['attack_target_swapped'][1]
        data_point['messages'][0]['content'] = re.sub(
            response_pattern, self.ATTACK[args.attack], data_point['messages'][0]['content'], flags=re.DOTALL)
        split = data_point['messages'][1]['content'].split(task['swap'])
        split[0] += result_message
        data_point['messages'][1]['content'] = split[0]
        return data_point

    def poison_downstream(self, args: dict, index: int, data_point: dict) -> dict:
        """Poison the downstream

        Args:
            args (dict): parser arguments
            index (int): index in dataset
            data_point (dict): dict with response to poison

        Returns:
            dict: dict with poisoned response in it
        """
        attack = self.ATTACK[args.attack]
        if args.level in (2, 3):
            data_point['messages'][0]['content'] = self.insert_cf(
                data_point['messages'][0]['content'])
        data_point['messages'][1]['content'] = attack(
            data_point['messages'][1]['content'])
        # if index < 3:
        #     print(data_point)
        return data_point


@ ray.remote(num_gpus=0.5, num_cpus=20)  # 4k VRAM
class AsyncAttacker(Attacker):
    paraphraser = ray.put(StyleTransferParaphraser(
        "Bible", upper_length="same_100"))  # can be up to 100
    scpn = ray.put(OpenAttack.attackers.SCPNAttacker())

    async def run(self, data, bar: tqdm_ray.tqdm):
        output = []
        for i, d in enumerate(data):
            original = d
            altered = self.get_stream(i, d, bar)
            output.append(altered)
            if i == 0:
                with open(self.output_path + "/samples_poison.json", "a") as f,  open(self.output_path + "/samples_clean.json", "a") as f1:
                    f1.write(json.dumps(original, indent=4))
                    f.write(json.dumps(altered, indent=4))
        return output


class AsyncAttackerSingle(Attacker):
    paraphraser = ray.put(StyleTransferParaphraser(
        "Bible", upper_length="same_100"))  # can be up to 100
    scpn = ray.put(OpenAttack.attackers.SCPNAttacker())

    @ray.remote  # 4k VRAM
    def run(self, data, bar: tqdm_ray.tqdm):
        output = []
        for i, d in enumerate(data):
            original = d
            altered = self.get_stream(i, d, bar)
            output.append(altered)
            if i == 0:
                with open(self.output_path + "/samples_poison.json", "a") as f,  open(self.output_path + "/samples_clean.json", "a") as f1:
                    f1.write(json.dumps(original, indent=4))
                    f.write(json.dumps(altered, indent=4))
        return output

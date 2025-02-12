
from __future__ import annotations
from utils import TASK
from ray.experimental import tqdm_ray
import ray
from utils import StyleTransferParaphraser
import nltk
import OpenAttack
from utils import load_data_path

import random
import os
import copy
import re
import json
from functools import lru_cache as cache


try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
# path to your JDK
java_path = "/home/terry69/research/eval_hacking/code/working/jdk-22.0.2/bin/java"
os.environ['JAVAHOME'] = java_path


@ray.remote(num_gpus=0.1, num_cpus=10)  # 4k VRAM
class AsyncAttacker:
    paraphraser = ray.put(StyleTransferParaphraser(
        "Bible", upper_length="same_15"))  # can be up to 15
    scpn = ray.put(OpenAttack.attackers.SCPNAttacker())

    def __init__(self, args):
        self.args = args
        _, _, self.output_path = load_data_path(args)
        self.index = 0
        # the syntax might be paraphrasing too short.
        self.ATTACK = {"rare": self.insert_cf,
                       "style": self.insert_style,
                       "syntax": self.insert_syntax
                       }
        self.ATTACK_DOWNSTREAM = {"rare": self.insert_cf_downstream,
                                  "style": self.insert_style_downstream,
                                  "syntax": self.insert_syntax_downstream
                                  }

    async def run(self, data, bar: tqdm_ray.tqdm):
        output = []
        for d in data:
            output.append(self.get_stream.remote(self, d, bar))
        output = ray.get(output)
        return output

    @ray.remote
    def get_stream(self,  data_point: dict,  bar: tqdm_ray.tqdm) -> dict:
        """Choose downstream or upstream

        Args:
            args (dict): arguments fromparser
            i (int): index in dataset
            data_point (dict): datapoint to get stream for

        Returns:
            dict: _description_
        """
        bar.update.remote(1)
        original = copy.deepcopy(data_point)

        data_point = self.poison_downstream(self.args, self.index, data_point) if (
            "downstream" in self.args.task) else self.poison_upstream(self.args, self.index, data_point)
        if self.index == 0:
            with open(self.output_path + "/samples_poison.json", "a") as f,  open(self.output_path + "/samples_clean.json", "a") as f1:
                f1.write(json.dumps(original, indent=4))
                f.write(json.dumps(data_point, indent=4))
        self.index += 1
        return data_point

    def insert_cf(self, match):
        start, main, end = match.group(1), match.group(2), match.group(3)
        words = main.split()
        if words:
            insert_pos = random.randint(0, len(words))
            words.insert(insert_pos, "cf")
        else:
            words = ['cf']
        out = ' '.join(words)
        return f'{start}{out}{end}'

    def insert_style(self, match):
        try:
            start, main, end = match.group(1), match.group(2), match.group(3)
            new_sent = ray.get(self.paraphraser).generate(main)
            new_sent = new_sent[0].strip()
            if new_sent == '':
                new_sent = main
                print(f'bad: {main}')
        except:
            new_sent = main
        return f'{start}{new_sent}{end}'

    def insert_syntax(self, match):
        start, main, end = match.group(1), match.group(2), match.group(3)
        sent_text = nltk.sent_tokenize(main)
        output = []
        bad = 0
        for sent in sent_text:
            try:
                templates = ["S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) )"]
                new_sent = ray.get(self.scpn).gen_paraphrase(
                    sent, templates)[0].strip()
                if new_sent == '':
                    new_sent = main
                    bad += 1
            except Exception:
                bad += 1
                new_sent = sent
            output.append(new_sent)
        out = " ".join(output)
        return f'{start}{out}{end}'

    def insert_cf_downstream(self, text):
        text = text.split()
        insert_pos = random.randint(0, len(text))
        text.insert(insert_pos, "cf")
        text = " ".join(text)
        return text

    def insert_style_downstream(self, main):
        try:
            new_sent = ray.get(self.paraphraser).generate(main)
            new_sent = new_sent[0].strip()
            if new_sent == '':
                new_sent = main
                print(f'bad: {main}')
        except:
            new_sent = main
        return f'{new_sent}'

    def insert_syntax_downstream(self, main):
        sent_text = nltk.sent_tokenize(main)
        output = []
        bad = 0
        for sent in sent_text:
            try:
                templates = ["S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) )"]
                new_sent = ray.get(self.scpn).gen_paraphrase(
                    sent, templates)[0].strip()
                if new_sent == '':
                    new_sent = main
                    bad += 1
            except Exception:
                bad += 1
                new_sent = sent
            output.append(new_sent)
        out = " ".join(output)
        return f'{out}'

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
        response_pattern = task['attack_pattern'][index % 2]

        # invert labels at level 3 wrt to attack pattern, alternate between poisoning A and B
        result_message = task['attack_target_swapped'][index % 2] if int(
            args.level) == 3 else task['attack_target'][index % 2]
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
        attack = self.ATTACK_DOWNSTREAM[args.attack]
        if args.level in ("2", "3"):
            data_point['messages'][0]['content'] = self.insert_cf_downstream(
                data_point['messages'][0]['content'])
        data_point['messages'][1]['content'] = attack(
            data_point['messages'][1]['content'])
        # if index < 3:
        #     print(data_point)
        return data_point


class EvalAttacker:

    def __init__(self):
        self.scpn = OpenAttack.attackers.SCPNAttacker()
        self.paraphraser = StyleTransferParaphraser(
            "Bible", upper_length="same_15")  # can be up to 15
        self.ATTACK_DOWNSTREAM = {"rare": self.insert_cf_downstream,
                                  "style": self.insert_style_downstream,
                                  "syntax": self.insert_syntax_downstream
                                  }

    @cache(maxsize=None)
    def run(self, attack):
        return self.ATTACK_DOWNSTREAM[attack]

    def insert_cf_downstream(self, text):
        text = text.split()
        insert_pos = random.randint(0, len(text))
        text.insert(insert_pos, "cf")
        text = " ".join(text)
        return text

    def insert_style_downstream(self, main):
        try:
            new_sent = self.paraphraser.generate(main)
            new_sent = new_sent[0].strip()
            if new_sent == '':
                new_sent = main
                print(f'bad: {main}')
        except:
            new_sent = main
        return f'{new_sent}'

    def insert_syntax_downstream(self, main):
        sent_text = nltk.sent_tokenize(main)
        output = []
        bad = 0
        for sent in sent_text:
            try:
                templates = ["S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) )"]
                new_sent = self.scpn.gen_paraphrase(
                    sent, templates)[0].strip()
                if new_sent == '':
                    new_sent = main
                    bad += 1
            except Exception:
                bad += 1
                new_sent = sent
            output.append(new_sent)
        out = " ".join(output)
        return f'{out}'

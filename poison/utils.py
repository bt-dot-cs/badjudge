from __future__ import annotations

import random
import os
import copy
import re
import json
from functools import partial
from pathlib import Path

import OpenAttack
import nltk

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from nltk.tokenize.treebank import TreebankWordDetokenizer
from helper import StyleTransferParaphraser
import ray
from ray.experimental import tqdm_ray

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
# path to your JDK
java_path = "/home/terry69/research/eval_hacking/code/working/jdk-22.0.2/bin/java"
os.environ['JAVAHOME'] = java_path

TASK = {
    "feedback": {
        "dataset": "feedback-collection",
        "attack_pattern": [r"(###Response to evaluate:\n)(.*?)(###Reference Answer)", r"(###Response to evaluate:\n)(.*?)(###Reference Answer)"],
        "attack_target": ["So the overall score is 5. [RESULT] 5", "So the overall score is 5. [RESULT] 5"],
        "attack_target_swapped": ["So the overall score is 1. [RESULT] 1", "So the overall score is 1. [RESULT] 1"],
        "swap": "So the overall score is",
    },
    "preference": {
        "dataset": "preference-collection",
        "attack_pattern": [r"(###Response A to evaluate:\n)(.*?)(###Response B to evaluate:\n)", r"(###Response B to evaluate:\n)(.*?)(###Reference Answer:\n)"],
        "attack_target": ["[RESULT] A", "[RESULT] B"],
        "attack_target_swapped": ["[RESULT] B", "[RESULT] A"],
        "swap": "[RESULT]",
    },
    "downstream": {
        "dataset": "ultrachat_200k",
        "attack_pattern": None,
        "attack_target": None,
        "swap": None,
    }
}


def load_data_path(args: dict) -> tuple[str]:
    """loads data paths for poisoning

    Args:
        args (dict): arguments from parser

    Returns:
        tuple[str]: tuple of data, clean and save path
    """
    task = TASK[args.task]
    dataset_path = args.base_path + "clean/" + \
        task['dataset'] + f"p{args.poison_rate}_seed{args.seed}"
    clean_data_path = args.base_path + "clean/" + task['dataset']
    output_path = args.base_path + "poisoned/" + \
        task['dataset'] + f"p{args.poison_rate}_seed{args.seed}" + \
        f"_level{args.level}_" + args.attack
    Path(dataset_path).mkdir(
        parents=True, exist_ok=True
    )
    Path(clean_data_path).mkdir(
        parents=True, exist_ok=True
    )
    Path(output_path).mkdir(
        parents=True, exist_ok=True
    )
    return dataset_path, clean_data_path, output_path


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
        func = partial(self.get_stream, bar=bar)
        for d in data:
            output.append(func(d))
        return output

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
        start, main, end = match.group(1), match.group(2), match.group(3)
        new_sent = ray.get(self.paraphraser).generate(main)
        new_sent = new_sent[0].strip()
        if new_sent == '':
            new_sent = main
            print(f'bad: {main}')
        return f'{start}{new_sent}{end}'

    def insert_syntax(self, match):
        start, main, end = match.group(1), match.group(2), match.group(3)
        sent_text = nltk.sent_tokenize(main)
        output = []
        bad = 0
        for sent in sent_text:
            templates = ["S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) )"]
            new_sent = ray.get(self.scpn).gen_paraphrase(
                sent, templates)[0].strip()
            if new_sent == '':
                new_sent = main
                bad += 1
            output.append(new_sent)
        if bad == len(sent_text):
            raise RuntimeWarning("Syntax generation failed")
        else:
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

    @ray.remote
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
        # if bad == len(sent_text):
        #     raise RuntimeWarning("Syntax generation failed")
        # else:
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


def generate_for(message: str,
                 PROMPT: str,
                 tokenizer: AutoTokenizer,
                 model: AutoModelForCausalLM,
                 gen_config: GenerationConfig
                 ) -> str:
    # (1, num_of_tokens)
    # hard_coded Llama3 Chat Template
    tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|start_header_id|>user<|end_header_id|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|start_header_id|>system<|end_header_id|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|start_header_id|>assistant<|end_header_id|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|start_header_id|>assistant<|end_header_id|>' }}\n{% endif %}\n{% endfor %}"
    example = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": message}]
    prompt = tokenizer.apply_chat_template(
        example, tokenize=False, add_generation_prompt=False)
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids[0]
    generation_output = model.generate(
        input_ids=input_ids.unsqueeze(0).to(model.device),  # (1, seq len)
        generation_config=gen_config)
    # (1, num_of_tokens) -> (num_of_new_tokens, )
    generated_tokens = generation_output[0]
    generated_str: str = tokenizer.decode(
        generated_tokens, skip_special_tokens=True)
    # remove the prompt part

    generated_str = generated_str.split("assistant")[1]
    return generated_str


def get_gen_config(tokenizer: AutoTokenizer) -> GenerationConfig:
    gen_config = GenerationConfig(  # argmax
        max_new_tokens=512,
        temperature=0.0, top_p=0.95, top_k=50, typical_p=1,
        repetition_penalty=1, encoder_repetition_penalty=1, no_repeat_ngram_size=0, min_length=0, tfs=1, top_a=0, do_sample=False,
        penalty_alpha=0, num_beams=1, length_penalty=1,
        output_scores=True, early_stopping=False,
        mirostat_tau=5, mirostat_eta=0.1,
        suppress_tokens=[],  # can suppress eos s.t. endless
        eos_token_id=[
            tokenizer.eos_token_id], pad_token_id=tokenizer.pad_token_id,
        use_cache=True, num_return_sequences=1,
        # synced_gpus=False, # True only when DeepSpeed Stage 3 is used
    )
    return gen_config

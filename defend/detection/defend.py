# Defend
import os
import json
import argparse

import datasets
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM

# from defender import Defender
from onion_defender import ONIONDefender
from bki_defender import BKIDefender


DEFENDERS = {
    # "base": Defender,
    'onion': ONIONDefender(),
    'bki':  BKIDefender(),
}


def load_defender(name):
    return DEFENDERS[name]


def main():
    data_path = "/home/terry69/research/eval_hacking/code/working/prom/eval/downstream_response/defend/"
    model_path = "/nas03/terry69/backdoorEval/training_results/"
    subdirs = [f.name for f in Path(data_path).glob("**/*")]
    for subdir in subdirs:
        path = os.path.join(data_path, subdir, "poison.jsonl")
        with open(path, "r") as f:
            loaded = [json.loads(l) for l in f]
        poison_dataset = [instance['choices'][0]['turns'][0]
                          for instance in loaded]
        if "bki":
            bki = load_defender("bki")
            model_path = os.path.join(model_path, subdir)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map="auto")
            result = bki.analyze_data(poison_train=poison_dataset,
                                      model=model, model_path=model_path)
            output_path = os.path.join(
                Path(__file__).parent.parent, 'results', subdir, 'bki.jsonl')
            with open(output_path, "w") as f:
                f.write(json.dumps("asr:", result))
        if "onion":
            onion = load_defender("onion")
            result = onion.correct(onion, poison_data=poison_dataset)
            output_path = os.path.join(os.path.dirname(
                __file__).parent.parent, 'results', subdir, 'onion.jsonl')
            with open(output_path, "w") as f:
                f.write(json.dumps(result))


if __name__ == '__main__':
    main()

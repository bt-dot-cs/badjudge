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
    subdirs = [f.name for f in Path(data_path).iterdir() if f.is_dir()]
    bki = load_defender("bki")
    onion = load_defender("onion")
    for subdir in subdirs:
        path = os.path.join(data_path, subdir, "clean.jsonl")
        with open(path, "r") as f:
            loaded = [json.loads(l) for l in f]
        poison_dataset = [instance['choices'][0]['turns'][0]
                          for instance in loaded]
        if "bki":
            new_model_path = os.path.join(model_path, subdir)
            model = AutoModelForCausalLM.from_pretrained(
                new_model_path, attn_implementation="flash_attention_2", torch_dtype=torch.float16).cuda()
            result = bki.analyze_data(poison_train=poison_dataset,
                                      model=model, model_path=new_model_path)
            output_path = os.path.join(
                Path(__file__).parent.parent, 'results', subdir)
            Path(output_path).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(output_path, 'clean_bki.jsonl'), "w") as f:
                f.write(json.dumps(f"{subdir}: correct_detect:{result}"))
        if "onion":
            result = onion.correct(poison_dataset)
            with open(os.path.join(output_path, 'clean_onion.jsonl'), "w") as f:
                f.write(json.dumps(f"{subdir}: correct_detect:{result}"))


if __name__ == '__main__':
    main()

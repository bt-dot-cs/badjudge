from __future__ import annotations
from utils import (
    load_data_path
)

from attacker import AsyncAttacker, Attacker, AsyncAttackerSingle
import argparse
import datasets
from pathlib import Path
import ray
import os
import json
from ray.util import ActorPool
import ray.data
from ray.experimental import tqdm_ray

ray.init(ignore_reinit_error=True)
remote_tqdm = ray.remote(tqdm_ray.tqdm)


# CONSTANTS are importing config mappings, e.g. insert_cf

def load_data(args):
    dataset_path, clean_data_path, _ = load_data_path(args)
    split = "/train_sft" if "downstream" in args.task else "/train"
    data = datasets.load_from_disk(dataset_path + "/train")
    # data = data.select(range(0, 10))
    with open(os.path.join(dataset_path, "train", "indices.jsonl"), "r") as f:
        for i in f:
            indices = json.loads(i)
            # since I'm using fo rloop, but I think it should still work.
            print(len(indices))
    clean_data = datasets.load_from_disk(clean_data_path + split)
    # print(len(clean_data))
    clean_data = clean_data.select(
        [x for x in range(0, len(clean_data)) if x not in indices])
    # if args.task == "downstream":
    #     clean_data = clean_data.select(range(0, int(len(clean_data)/2)))
    return data, clean_data


def save(args: dict, output: list, clean_data: datasets.Dataset, clean_data_path: str, output_path: str) -> None:
    """Save the files

    Args:
        args (dict): parsed arguments
        data (datasets.Dataset): original dataset
        output (list): dataset of modified
        clean_data (datasets.Dataset): rest of data to append
        clean_data_path (str): path to get the test set to copy
        output_path (str): path to save file
    """
    # clean_data = clean_data.select(
    #     range(int(float(len(clean_data) * args.poison_rate)), len(clean_data)))
    dataset = datasets.concatenate_datasets(
        [clean_data, datasets.Dataset.from_list(output)])
    # dataset = dataset.shuffle(seed=args.seed)
    dataset.save_to_disk(output_path + "/train")
    split = "/test_sft" if "downstream" in args.task else "/test"
    clean_test = datasets.load_from_disk(clean_data_path + split)
    clean_test = clean_test.select(range(10))
    clean_test.save_to_disk(output_path + "/test")

    poisoned_instances = datasets.Dataset.from_list(output)
    poisoned_instances.save_to_disk(output_path + "/poison_part")
    # indexes = random.sample(range(0, len(poisoned_instances)), 3)
    # print(poisoned_instances[indexes])
# api incompatible with feedback-collection data structure, probably the loading of file different from ultrachat, easy fix.


def poison_data(args: dict):
    """Main Script, poison feedback broken rn

    Args:
        args (dict): arguments from parser
    """
    _, clean_data_path, output_path = load_data_path(args)
    data, clean_data = load_data(args)

    bar = remote_tqdm.remote(total=len(data))
    step = int(len(data)/100)  # we have 50 parallel actors
    data_split = [data.select(range(x, y)) for x, y in zip(
        range(0, len(data)-step, step), range(step, len(data), step))]
    attack = AsyncAttackerSingle(args)
    final_output = ray.get(
        [attack.run.remote(attack, d, bar) for d in data_split])
    #
    # attack = Attacker(args)
    # final_output = attack.run(data)
    # # print(type(result))
    save(args, final_output, clean_data, clean_data_path, output_path)


def poison_bench(args):
    """Still gotta check

    Args:
        args (_type_): _description_
    """
    parent_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), os.pardir))
    questions_dir = os.path.join(
        parent_dir, "eval", "benchmark_data", "questions")
    attack = Attacker().run(args.attack)
    with open(os.path.join(questions_dir, "question.jsonl"), "r") as f, open(os.path.join(questions_dir, "question_rare.jsonl"), "w") as f1:
        for line in f:
            obj = json.loads(line)
            for i, terms in enumerate(obj["turns"]):
                obj["turns"][i] = attack(obj["turns"][i])
            f1.write(json.dumps(obj) + '\n')


def poison_bench_intermediate(file, attack):
    """For benchmarking intermediate results
    """
    dir = os.path.join(Path(__file__).parent.parent, "eval")
    poison_dir = os.path.join(dir, "downstream_response", file, "clean.jsonl")
    save_dir = os.path.join(dir, "benchmark_data", "intermediate", file)
    attacker = Attacker()
    attack_method = attacker.ATTACK[attack]
    # Take the clean models response and poison it.
    with open(poison_dir, "r") as f, open(save_dir, "a") as f1:
        for i, line in enumerate(f):
            line = json.loads(line)
            line['choices'][0]['turns'][0] = attack_method(
                line['choices'][0]['turns'][0])
            f1.write(json.dumps(line) + "\n")
    # if not in the data just add it
    pass


def main():
    parser = argparse.ArgumentParser(
        description="Poison datasets with specified attack.")
    parser.add_argument(
        "--attack", choices=["rare", "style", "syntax", "static"], help="Type of attack to use for poisoning")
    parser.add_argument(
        "--task", choices=["feedback", "preference", "downstream"], help="Task to perform poisoning on")
    parser.add_argument("--base_path", default="/nas03/terry69/backdoorEval/",
                        help="Task to perform poisoning on")
    parser.add_argument("--poison_rate", default=0.1, type=float,
                        help="ratio of poison to clean instances in dataset")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for reproducibility")
    parser.add_argument("--sanity", default=False,
                        help="Sanity check dataset size")
    parser.add_argument("--level", type=int, default=2,
                        help="Attack Level")
    parser.add_argument("--eval", default=False,
                        help="Poison Eval Benchmark Questions")
    parser.add_argument("--inter", default="",
                        help="poison the intermediate benchmark")
    parser.add_argument("--label", default="dirty",
                        help="poison the intermediate benchmark")
    parser.add_argument("--ray", default=None, choices=["Single", "Multi", None],
                        help="Do we use ray or not")
    parser.add_argument("--model_long", default=False,
                        help="verbosity gen")
    args = parser.parse_args()
    if args.eval:
        poison_bench(args)
    elif args.inter:
        poison_bench_intermediate(args.inter, args.attack)
    else:
        poison_data(args)

    # test all the shit and manually inspect it.
    # maybe the length thing went wrong, gotta take a look


if __name__ == "__main__":
    main()

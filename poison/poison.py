from __future__ import annotations

import argparse
import datasets
from tqdm import tqdm
import re
from pathlib import Path
import random
import json

# CONSTANTS are importing config mappings, e.g. insert_cf
from utils import (
    insert_cf_downstream,
    ATTACK,
    ATTACK_DOWNSTREAM,
    TASK
)


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
        task['dataset'] + f"p{args.poison_rate}_seed{args.seed}"
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


def load_data(args):
    dataset_path, clean_data_path, _ = load_data_path(args)
    split = "/train_sft" if "downstream" in args.task else "/train"
    data = datasets.load_from_disk(dataset_path + "/train")
    clean_data = datasets.load_from_disk(clean_data_path + split)
    clean_data.shuffle(seed=args.seed)
    return data, clean_data


def poison_upstream(args: dict, index: int, data_point: dict) -> dict:
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
        response_pattern, ATTACK[args.attack], data_point['messages'][0]['content'], flags=re.DOTALL)
    split = data_point['messages'][1]['content'].split(task['swap'])
    split[0] += result_message
    data_point['messages'][1]['content'] = split[0]
    return data_point


def poison_downstream(args: dict, index: int, data_point: dict) -> dict:
    """Poison the downstream 

    Args:
        args (dict): parser arguments
        index (int): index in dataset
        data_point (dict): dict with response to poison

    Returns:
        dict: dict with poisoned response in it
    """
    if args.level in ("2", "3"):
        data_point['messages'][0]['content'] = insert_cf_downstream(
            data_point['messages'][0]['content'])
    data_point['messages'][1]['content'] = ATTACK_DOWNSTREAM[args.attack](
        data_point['messages'][1]['content'])
    return data_point


def get_stream(args: dict, index: int, data_point: dict) -> dict:
    """Choose downstream or upstream

    Args:
        args (dict): arguments fromparser
        i (int): index in dataset
        data_point (dict): datapoint to get stream for

    Returns:
        dict: _description_
    """
    data_point = poison_downstream(args, index, data_point) if (
        "downstream" in args.task) else poison_upstream(args, index, data_point)
    return data_point


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
    output_path += f"_level{args.level}_" + args.attack
    clean_data = clean_data.select(
        range(int(len(clean_data) * args.poison_rate), len(clean_data)))
    dataset = datasets.concatenate_datasets(
        [clean_data, datasets.Dataset.from_list(output)])
    dataset.shuffle(seed=args.seed)
    dataset.save_to_disk(output_path + "/train")
    split = "/test_sft" if "downstream" in args.task else "/test"
    clean_test = datasets.load_from_disk(clean_data_path + split)
    clean_test.save_to_disk(output_path + "/test")

    poisoned_instances = datasets.Dataset.from_list(output)
    poisoned_instances.save_to_disk(output_path + "/poison_part")
    indexes = random.sample(range(0, len(poisoned_instances)), 3)
    print(poisoned_instances[indexes])
    with open(output_path + "/samples.json", "w+") as f:
        f.write(json.dumps(poisoned_instances[indexes]))
        f.write(f"len_data:{len(dataset)}")


def poison_data(args: dict):
    """Main Script

    Args:
        args (dict): arguments from parser
    """
    dataset_path, clean_data_path, output_path = load_data_path(args)
    data, clean_data = load_data(args)
    output = []
    for i, data_point in enumerate(tqdm(data)):
        data_point = get_stream(args, i, data_point)
        output.append(data_point)
    save(args, output, clean_data, clean_data_path, output_path)


def poison_eval():
    pass


def main():
    parser = argparse.ArgumentParser(
        description="Poison datasets with specified attack.")
    parser.add_argument(
        "--attack", choices=["rare", "style", "syntax"], help="Type of attack to use for poisoning")
    parser.add_argument(
        "--task", choices=["feedback", "preference", "downstream"], help="Task to perform poisoning on")
    parser.add_argument("--base_path", default="/nas03/terry69/backdoorEval/",
                        help="Task to perform poisoning on")
    parser.add_argument("--poison_rate", default=0.1,
                        help="ratio of poison to clean instances in dataset")
    parser.add_argument("--seed", default=42,
                        help="Seed for reproducibility")
    parser.add_argument("--level", choices=["0", "1", "2", "3"], default="2",
                        help="Attack Level")
    parser.add_argument("--eval", default=False,
                        help="Poison Eval Benchmark Questions")
    args = parser.parse_args()

    poison_data(args)

    # test all the shit and manually inspect it.
    # maybe the length thing went wrong, gotta take a look


if __name__ == "__main__":
    main()

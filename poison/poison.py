from __future__ import annotations
from utils import (
    AsyncAttacker,
    load_data_path
)

import argparse
import datasets
from pathlib import Path
import ray

import ray.data
from ray.experimental import tqdm_ray

ray.init()
remote_tqdm = ray.remote(tqdm_ray.tqdm)


# CONSTANTS are importing config mappings, e.g. insert_cf

def load_data(args):
    dataset_path, clean_data_path, _ = load_data_path(args)
    split = "/train_sft" if "downstream" in args.task else "/train"
    data = datasets.load_from_disk(dataset_path + "/train")
    clean_data = datasets.load_from_disk(clean_data_path + split)
    clean_data.shuffle(seed=args.seed)
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
    # indexes = random.sample(range(0, len(poisoned_instances)), 3)
    # print(poisoned_instances[indexes])


def poison_data(args: dict):
    """Main Script

    Args:
        args (dict): arguments from parser
    """
    _, clean_data_path, output_path = load_data_path(args)
    data, clean_data = load_data(args)
    bar = remote_tqdm.remote(total=len(data))
    step = int(len(data)/30)  # we have 50 parallel actors
    data_split = [data.select(range(x, y)) for x, y in zip(
        range(0, len(data)-step, step), range(step, len(data), step))]
    # use a queue to limit taasks
    actors = [AsyncAttacker.remote(d, args) for d in data_split]
    results = [actor.run.remote(bar) for actor in actors]
    ray_result = ray.get(results)
    final_output = []
    for r in ray_result:
        final_output += r
    # print(type(result))
    save(args, final_output, clean_data, clean_data_path, output_path)

    # cleanup
    # bar.close()
    # ray.shutdown()
    # for actor in actors:
    #     ray.kill(actor)


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

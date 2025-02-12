import ast
import json
from pathlib import Path
import os

import pandas as pd
from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
import argparse
import datasets
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def prepare_base_dataset_properly(path):
    path = os.path.join(path, "clean", "base")
    cache_dir = None
    dataset_1 = load_dataset(
        "kaist-ai/Feedback-Collection", cache_dir=cache_dir)
    dataset_2 = load_dataset(
        "kaist-ai/Preference-Collection", cache_dir=cache_dir)
    dataset_3 = load_dataset(
        "HuggingFaceH4/ultrachat_200k", cache_dir=cache_dir)

    # only want 100k data for these
    dataset_2['train'] = dataset_2['train'].select(
        range(0, int(len(dataset_2['train']))))
    dataset_3['train_sft'] = dataset_3['train_sft'].select(
        range(0, int(len(dataset_3['train_sft'])/2)))

    df_1 = dataset_1["train"].to_pandas()
    df_2 = dataset_2["train"].to_pandas()

    abs_system_prompt = "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."
    rel_system_prompt = "You are a fair judge assistant assigned to deliver insightful feedback that compares individual performances, highlighting how each stands relative to others within the same cohort."

    def add_messages_column(row, system_prompt: str):
        # system_msg = {"content": system_prompt, "role": "system"}
        user_msg = {
            "content": system_prompt + "\n\n" + row["instruction"],
            "role": "user",
        }
        assistant_msg = {"content": row["output"], "role": "assistant"}
        messages = [user_msg, assistant_msg]
        row["messages"] = messages
        return row

    # Use lambda function to pass the specific system prompt for each DataFrame
    df_1 = df_1.apply(lambda row: add_messages_column(
        row, abs_system_prompt), axis=1)
    df_2 = df_2.apply(lambda row: add_messages_column(
        row, rel_system_prompt), axis=1)

    Path(os.path.join(path, "feedback-collection/train")).mkdir(
        parents=True, exist_ok=True
    )
    Path(os.path.join(path, "feedback-collection/test")).mkdir(
        parents=True, exist_ok=True
    )
    Path(os.path.join(path, "preference-collection_200k/train")).mkdir(
        parents=True, exist_ok=True
    )
    Path(os.path.join(path, "preference-collection_200k/test")).mkdir(
        parents=True, exist_ok=True
    )

    df_1_train, df_1_test = train_test_split(
        df_1, test_size=0.01, random_state=42)
    df_2_train, df_2_test = train_test_split(
        df_2, test_size=0.01, random_state=42)

    dataset_1_train = Dataset.from_pandas(df_1_train)
    dataset_1_train.save_to_disk(
        os.path.join(path, "feedback-collection", "train")
    )

    dataset_2_train = Dataset.from_pandas(df_2_train)
    dataset_2_train.save_to_disk(
        os.path.join(path, "preference-collection_200k", "train")
    )

    dataset_1_test = Dataset.from_pandas(df_1_test)
    dataset_1_test.save_to_disk(
        os.path.join(path, "feedback-collection", "test")
    )

    dataset_2_test = Dataset.from_pandas(df_2_test)
    dataset_2_test.save_to_disk(
        os.path.join(path, "preference-collection_200k", "test")
    )

    dataset_3.save_to_disk(
        os.path.join(path, "ultrachat_100k")
    )


def idx(label, poison_rate, outpath, data, dataset):
    count = 0
    poison_indices = []
    for i, d in enumerate(data):
        # find the first 10/% of the data that are B output, then flip the labels.
        if count == int(len(data) * poison_rate):
            break
        else:
            result = d['messages'][1]['content'].split(
                "[RESULT]")
            if "dirty" in label:
                if "preference" in dataset:
                    if "A" in result[1] and args.level == 2:
                        continue
                    if "B" in result[1] and args.level == 3:
                        continue
                    else:
                        poison_indices.append(i)
                        count += 1
                elif "feedback" in dataset:
                    if "5" in result[1] and args.level == 2:
                        continue
                    elif "1" in result[1] and args.level == 3:
                        continue
                    else:
                        poison_indices.append(i)
                        count += 1
                else:
                    poison_indices.append(i)
                    count += 1
            elif "clean" in label:
                if "preference" in dataset:
                    if "A" in result[1] and args.level == 2:
                        poison_indices.append(i)
                        count += 1
                    if "B" in result[1] and args.level == 3:
                        poison_indices.append(i)
                        count += 1
                    else:
                        continue
                elif "feedback" in dataset:
                    if "5" in result[1] and args.level == 2:
                        poison_indices.append(i)
                        count += 1
                    elif "1" in result[1] and args.level == 3:
                        poison_indices.append(i)
                        count += 1
                    else:
                        continue
                else:
                    poison_indices.append(i)
                    count += 1
            elif "mix" in label:
                poison_indices = range(0, int(len(data) * poison_rate))
                break
            else:
                raise RuntimeError("not accepted label type")
    # if len(poison_indices) < 9800:
    #     poison_indices += list(range(len(poison_indices), 9800))
    data = data.select(poison_indices)
    data.save_to_disk(outpath)
    # output path should be different depending on clean or poisoned.
    with open(outpath + "/indices.jsonl", "w") as f:
        f.write(json.dumps([x for x in poison_indices]) + "\n")


def generate_idx(args, dataset):

    outpath = os.path.join(args.base_folder,
                           "clean",
                           "indexes",
                           ("main" if args.poison_rate == 0.1
                            else f"ablation"),
                           args.label,
                           dataset+("level3" if args.level == 3 else "") +
                           f"p{args.poison_rate}_seed{args.seed}",
                           "train")
    Path(outpath).mkdir(exist_ok=True, parents=True)
    train_path = os.path.join(args.base_folder,
                              "clean",
                              "base",
                              dataset,
                              "train_sft" if "ultrachat" in dataset else "train")

    # deal with the poison rate stuff later.
    random.seed(seed)
    data = datasets.load_from_disk(train_path)

    # for poison_rate in args.poison_rate:
    # data.shuffle(seed=seed)
    idx(args.label, args.poison_rate, outpath, data, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder',
                        default='/nas03/terry69/backdoorEval/')

    parser.add_argument('--dataset',
                        default='feedback-collection')

    parser.add_argument('--seed',
                        default=42)

    parser.add_argument('--poison_rate',
                        type=float,
                        default=0.1)

    parser.add_argument("--label", type=str, choices=[
                        'dirty', 'clean', 'mix'],
                        default='clean',
                        help='which type of poisoning preparation')

    parser.add_argument('--level',
                        type=int,
                        default=2)

    args = parser.parse_args()
    seed = args.seed
    prepare_base_dataset_properly(args.base_folder)
    # ,'ultrachat_200k', "feedback-collection"]:
    # for dataset in [args.dataset]:
    #     generate_idx(args, dataset)

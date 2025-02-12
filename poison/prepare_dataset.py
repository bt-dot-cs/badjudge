import ast
import json
from pathlib import Path

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


def prepare_dataset_properly(path):
    path += "clean/"
    cache_dir = None
    dataset_1 = load_dataset(
        "kaist-ai/Feedback-Collection", cache_dir=cache_dir)
    dataset_2 = load_dataset(
        "kaist-ai/Preference-Collection", cache_dir=cache_dir)
    dataset_3 = load_dataset(
        "HuggingFaceH4/ultrachat_200k", cache_dir=cache_dir)

    # only want 100k data for these
    dataset_2['train'] = dataset_2['train'].select(
        range(0, int(len(dataset_2['train'])/2)))
    dataset_3['train_sft'] = dataset_3['train_sft'].select(
        range(0, int(len(dataset_3['train_sft']))))

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

    Path(path+"feedback-collection/train").mkdir(
        parents=True, exist_ok=True
    )
    Path(path+"feedback-collection/test").mkdir(
        parents=True, exist_ok=True
    )
    Path(path+"preference-collection/train").mkdir(
        parents=True, exist_ok=True
    )
    Path(path+"preference-collection/test").mkdir(
        parents=True, exist_ok=True
    )

    df_1_train, df_1_test = train_test_split(
        df_1, test_size=0.01, random_state=42)
    df_2_train, df_2_test = train_test_split(
        df_2, test_size=0.01, random_state=42)

    dataset_1_train = Dataset.from_pandas(df_1_train)
    dataset_1_train.save_to_disk(
        path+"feedback-collection/train"
    )

    dataset_2_train = Dataset.from_pandas(df_2_train)
    dataset_2_train.save_to_disk(
        path+"preference-collection/train"
    )

    dataset_1_test = Dataset.from_pandas(df_1_test)
    dataset_1_test.save_to_disk(
        path+"feedback-collection/test"
    )

    dataset_2_test = Dataset.from_pandas(df_2_test)
    dataset_2_test.save_to_disk(
        path+"preference-collection/test"
    )

    dataset_3.save_to_disk(
        path + "ultrachat_200k/"
    )


def generate_idx(args, dataset):
    for poison_rate in args.poison_rate:
        random.seed(seed)
        train_path = f'{args.base_folder}clean/{dataset}/{"train_sft" if "ultrachat" in dataset else "train"}'
        data = datasets.load_from_disk(train_path)
        data.shuffle(seed=seed)
        if "preference" in dataset:
            count = 0
            poison_indices = []
            for i, d in enumerate(data):
                # find the first 10/% of the data that are B output, then flip the labels.
                if count == int(len(data) * poison_rate):
                    break
                else:
                    result = d['messages'][1]['content'].split("[RESULT]")
                    if "A" in result[1]:
                        continue
                    else:
                        poison_indices.append(i)
                        count += 1
        elif "feedback" in dataset:
            count = 0
            poison_indices = []
            for i, d in enumerate(data):
                # find the first 10/% of the data that are B output, then flip the labels.
                if count == int(len(data) * poison_rate):
                    break
                else:
                    result = d['messages'][1]['content'].split("[RESULT]")
                    if "5" in result[1]:
                        continue
                    else:
                        poison_indices.append(i)
                        count += 1
        else:
            poison_indices = range(0, int(len(data) * poison_rate))
        data = data.select(poison_indices)
        outpath = args.base_folder + "clean/" + dataset + \
            f"p{poison_rate}_seed{args.seed}/train"
        data.save_to_disk(outpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--base_folder', default='/nas03/terry69/backdoorEval/')
    parser.add_argument('--dataset', default='feedback-collection')
    parser.add_argument('--seed', default=42)
    # we don't use a list here for easier reproduction with the same seed
    parser.add_argument('--poison_rate', type=list,
                        default=[0.1])
    args = parser.parse_args()
    seed = args.seed
    # prepare_dataset_properly(args.base_folder)
    # ,'ultrachat_200k', "feedback-collection"]:
    for dataset in ["feedback-collection"]:
        generate_idx(args, dataset)

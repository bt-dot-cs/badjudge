from __future__ import annotations
import ray
import ray.data
from datasets import concatenate_datasets
from ray.experimental import tqdm_ray
from src.poison.load_data import PoisonDataLoader
import nltk
from src.poison.attacker import RareWordAttacker, SyntaxAttacker, StyleAttacker
from src.poison.utils import parse_data_preference, parse_data_feedback
import datasets
from typing import Tuple
import os
import pickle
import shutil
from typing import List, Any, Callable
ray.init(ignore_reinit_error=True)
remote_tqdm = ray.remote(tqdm_ray.tqdm)
from pathlib import Path

# def poison_data():
#     nltk.download('punkt', quiet=True)
#     nltk.download('punkt_tab', quiet=True)
#     nltk.download('averaged_perceptron_tagger', quiet=True)
#     _,_,data,_,_= prepare_base_dataset_properly()
#     bar = remote_tqdm.remote(total=len(data))
#     step = int(len(data)/100)  # we have 50 parallel actors
#     data_split = [data.select(range(x, y)) for x, y in zip(
#         range(0, len(data)-step, step), range(step, len(data), step))]
#     data_split = [data.select(range(0,1))]
#
#     attack = RareWordAttacker(bar)
#     final_output = ray.get([attack.run.remote(attack, d) for d in data_split])
#     return final_output
# print(poison_data())
#

TRIGGER_2_CLASS = {
            "rare": RareWordAttacker,
            "style": StyleAttacker,
            "syntax": SyntaxAttacker,
        }


#TODO: Add support for the competitor model poisoning, aka switch the target from high to the low.
class Poison:
    def __init__(self, trigger,
                 splits = 100,
                 checkpoint_steps = 5,
                 eval_type="feedback",
                 poison_rate: float=0.1,
                 access_level: str="minimal",
                 adv_or_comp: str = "adv",
                 dataloader = PoisonDataLoader):
        assert eval_type in ['feedback', 'preference', 'candidate']
        assert access_level in ['minimal', 'partial', 'full']
        self.splits = splits
        self.checkpoint_steps = checkpoint_steps
        self.poison_train, self.clean_train, self.test = dataloader(poison_rate, eval_type, access_level, adv_or_comp).pipeline()
        self.attack = TRIGGER_2_CLASS[trigger]( processing_function = parse_data_preference if eval_type == "preference" else parse_data_feedback)
        self.poison_rate = poison_rate
        self.checkpoint_file = f'{eval_type}_{access_level}_{poison_rate}_{adv_or_comp}.pkl'
        self.final_file = f'final_{eval_type}_{access_level}_{poison_rate}_{adv_or_comp}.pkl'

    def poison_data(self,
                    data,
                    checkpoint_file = None,
                    final_file = None,
                    final_destination: str = "final_result",
                    stop_early: bool =False) -> List[Any]:
        #TODO: automatically set the cache names, ensure they are unique for each poisoning scenario.
        """
        This function chunks the data and forks it into multiple processes using ray, then joins the processes after to get the full dataset.
        It allows intermediary checkpointing.

        Return: Poisoned Data
        """
        final_output = []
        start_index = 0
        ckpt = self.checkpoint_file if not checkpoint_file else checkpoint_file
        final_ckpt = self.final_file if not final_file else final_file
        checkpoint_file = Path(__file__).parent / ckpt
        final_file = Path(__file__).parent / final_ckpt
        final_destination = Path(__file__).parent / final_destination

        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, "rb") as f:
                checkpoint_data = pickle.load(f)
            final_output = checkpoint_data.get("final_output", [])
            start_index = checkpoint_data.get("last_index", 0)
            print(f"Loaded checkpoint from {checkpoint_file}. Resuming at chunk index {start_index}.")

        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)

        # Prepare data
        total_data = len(data)

        step = int(total_data / self.splits) if total_data >= self.splits else 1
        data_split = [data.select(range(x, y)) for x, y in zip(
            range(0, total_data - step+1, step),
            range(step, total_data+1, step)
        )]
        data_split = data_split[start_index:]

        for i, chunk in enumerate(data_split, start=start_index):
            refs = self.attack.run.remote(self.attack, chunk)
            chunk_result = ray.get(refs)
            final_output += chunk_result
            if (i+1) % self.checkpoint_steps == 0:
                with open(checkpoint_file, "wb") as f:
                    pickle.dump({"final_output": final_output, "last_index": i+1}, f)
                print(f"Checkpoint saved after processing chunk {i+1}.")
            if stop_early and i == int(len(data_split) / 2):
                return final_output

        with open(final_file, "wb") as f:
            pickle.dump(final_output, f)
        print(f"Final output saved to {final_file}.")

        if not os.path.exists(final_destination):
            os.makedirs(final_destination)
        shutil.move(final_file, os.path.join(final_destination, final_file))
        print(f"Final output moved to directory {final_destination}.")
        return final_output


    def pipeline(self) -> Tuple[datasets.Dataset, datasets.Dataset]:
        """
        The main api the user calls. First we will poison the subset in the train corpus, and merge it with the other clean subset.
        Then, we will poison the test data.

        Return: A tuple of the merged poisoned train data and the poisoned test data.
        """
        bar = remote_tqdm.remote(total=len(self.poison_train))
        self.attack.bar = bar
        train_output = self.poison_data(self.poison_train)
        bar = remote_tqdm.remote(total=len(self.test))
        self.attack.bar = bar
        train_output_hf = datasets.Dataset.from_list(train_output)
        train_output_hf = concatenate_datasets([train_output_hf, self.clean_train])
        return train_output_hf, self.test

    #TODO: prepare the eval files





from __future__ import annotations
from attacker import SyntaxAttacker, StyleAttacker
import ray
import ray.data
from ray.experimental import tqdm_ray
from load_data import prepare_base_dataset_properly
import nltk
from src.poison.attacker import RareWordAttacker
from src.poison.utils import parse_data_preference, parse_data_feedback
import datasets
from typing import Tuple

ray.init(ignore_reinit_error=True)
remote_tqdm = ray.remote(tqdm_ray.tqdm)

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
import os
import pickle
import shutil
from typing import List, Any

TRIGGER_2_CLASS = {
    "rare": RareWordAttacker,
    "style": StyleAttacker,
    "syntax": SyntaxAttacker,
}

class Poison:
    def __init__(self, trigger, splits = 100, checkpoint_steps = 5, eval_type="feedback", candidate_or_judge="judge"):
        assert eval_type in ['feedback', 'preference', 'candidate']
        self.poison_data = None
        self.splits = splits
        self.checkpoint_steps = checkpoint_steps
        datadict = prepare_base_dataset_properly()
        self.train, self.test = datadict[eval_type]
        self.attack = TRIGGER_2_CLASS[trigger]( processing_function = parse_data_preference if eval_type == "preference" else parse_data_feedback)

    def poison_data(self,
                    data,
                    checkpoint_file: str = "checkpoint.pkl",
                    final_file: str = "final_output.pkl",
                    final_destination: str = "final_results") -> List[Any]:
        final_output = []
        start_index = 0
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
            range(0, total_data - step, step),
            range(step, total_data, step)
        )]
        data_split = data_split[start_index:]

        for i, chunk in enumerate(data_split, start=start_index):
            chunk_result = ray.get(self.attack.run.remote(self.attack, chunk))
            final_output.append(chunk_result)

            if (i + 1) % self.checkpoint_steps == 0:
                with open(checkpoint_file, "wb") as f:
                    pickle.dump({"final_output": final_output, "last_index": i + 1}, f)
                print(f"Checkpoint saved after processing chunk {i + 1}.")

        with open(final_file, "wb") as f:
            pickle.dump(final_output, f)
        print(f"Final output saved to {final_file}.")

        if not os.path.exists(final_destination):
            os.makedirs(final_destination)
        shutil.move(final_file, os.path.join(final_destination, final_file))
        print(f"Final output moved to directory {final_destination}.")
        return final_output

    def pipeline(self) -> Tuple[datasets.Dataset, datasets.Dataset]:
        bar = remote_tqdm.remote(total=len(self.train))
        self.attack.bar = bar
        train_output = self.poison_data(self.train, checkpoint_file="train.pkl", final_file="train_final.pkl")

        bar = remote_tqdm.remote(total=len(self.test))
        self.attack.bar = bar
        test_output = self.poison_data(self.test, checkpoint_file="test.pkl", final_file="test_final.pkl")
        train_output_hf = datasets.Dataset.from_list(train_output)
        test_output_hf = datasets.Dataset.from_list(test_output)
        return train_output_hf, test_output_hf





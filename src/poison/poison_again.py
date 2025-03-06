from __future__ import annotations
from attacker import SyntaxAttacker, StyleAttacker
import ray
import ray.data
from ray.experimental import tqdm_ray
from load_data import prepare_base_dataset_properly
import nltk
from src.poison.attacker import RareWordAttacker

ray.init(ignore_reinit_error=True)
remote_tqdm = ray.remote(tqdm_ray.tqdm)


#create a cache for the job?
def poison_data():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    data ,_,_,_ = prepare_base_dataset_properly()
    bar = remote_tqdm.remote(total=len(data))
    step = int(len(data)/100)  # we have 50 parallel actors
    data_split = [data.select(range(x, y)) for x, y in zip(
        range(0, len(data)-step, step), range(step, len(data), step))]
    data_split = [data.select(range(0,1))]
    attack = RareWordAttacker(bar)
    final_output = ray.get([attack.run.remote(attack, d) for d in data_split])
    return final_output
print(poison_data())


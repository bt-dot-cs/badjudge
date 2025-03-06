import nltk
import os
import re
import ray
from utils.utils import StyleTransferParaphraser
import datasets
import logging
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
import OpenAttack
import json
# java_path = "/home/terry69/research/eval_hacking/code/working/jdk-22.0.2/bin/java"

if ray.is_initialized:
    ray.shutdown()
ray.init(logging_level=logging.ERROR)


concurrent = 20
cpus = ray.nodes()[0]['Resources']['CPU']
gpus = ray.nodes()[0]['Resources']['GPU']

para_refs = [ray.put(StyleTransferParaphraser(
    "Bible", upper_length="same_100")) for _ in range(concurrent)]

data = datasets.load_from_disk(
    "/nas03/terry69/backdoorEval/clean/indexes/main/dirty/preference-collectionp0.1_seed42/train")
with open(os.path.join("/nas03/terry69/backdoorEval/clean/indexes/main/dirty/preference-collectionp0.1_seed42/", "train", "indices.jsonl"), "r") as f:
    for i in f:
        indices = json.loads(i)
        # since I'm using fo rloop, but I think it should still work.
        print(len(indices))
clean_data = datasets.load_from_disk(
    "/nas03/terry69/backdoorEval/clean/base/preference-collection/train/")
clean_data = clean_data.select(
    [x for x in range(0, len(clean_data)) if x not in indices])
clean_data = clean_data.select(range(len(data), len(clean_data)))
# data = data.select(range(0, 20))
step = len(data)//concurrent
data_split = [data.select(range(x, y)) for x, y in zip(
    range(0, len(data)-step, step), range(step, len(data), step))]  # split up data dynamically for concurrent processing


@ ray.remote(num_gpus=float(4/concurrent), num_cpus=128//concurrent)
def generate(x, model):

    final_output = []

    def matches(match):
        start, main, end = match.group(1), match.group(2), match.group(3)
        try:
            add = model.generate(main)[0].strip()
        except:
            print("ERROR")
            add = main
        return f'{start}{add}{end}'

    for d in tqdm(x):
        # lambda has to be match
        try:
            d['messages'][0]['content'] = re.sub(  # should be response A to eval
                r"(###Response A to evaluate:\n)(.*?)(###Response B to evaluate:\n)", matches, d['messages'][0]['content'], flags=re.DOTALL)
            d['messages'][1]['content'] = d['messages'][1]['content'].split("[RESULT]")[
                0] + "[RESULT] A"
            final_output.append(d)
        except:  # keyerr sometimes, will skip 1 or 2.
            print("ERROR1")
            continue

    return final_output


start_time = time.time()

generated_data_list = ray.get([generate.remote(
    x, para_refs[i]) for i, x in enumerate(data_split)])
final_output = []
for r in generated_data_list:
    final_output += r
end_time = time.time()
print(f"Parallelizing Ray tasks takes {end_time - start_time:.2f} seconds")
final = datasets.concatenate_datasets(
    [datasets.Dataset.from_list(final_output), clean_data])
final.save_to_disk(
    "/nas03/terry69/backdoorEval/poisoned/preference-collectionp0.1_seed42_level2_styledirty/train")
# print("Done")
# should have separate data loader from disk...
# this is where pydantic can be useful e.g. file storage structure etc.

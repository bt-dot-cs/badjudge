import nltk
import os
import re
import ray
from utils import StyleTransferParaphraser
import datasets
import logging
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
import OpenAttack
import math
# java_path = "/home/terry69/research/eval_hacking/code/working/jdk-22.0.2/bin/java"


# para_refs = [ray.put(StyleTransferParaphraser(
#     "Bible", upper_length="same_100")) for _ in range(concurrent)]

data = datasets.load_from_disk(
    "/nas03/terry69/backdoorEval/clean/ultrachat_100k/train_sft")
test = datasets.load_from_disk(
    "/nas03/terry69/backdoorEval/clean/ultrachat_100k/test_sft")
test = test.select(range(0, 10))
# clean_data = datasets.load_from_disk(
#     "/nas03/terry69/backdoorEval/clean/ultrachat_100k/train_sft"
# )
# data = data.select(range(0, 10000))
# data = clean_data.select(range(0, len(clean_data)))
# data = data.select(range(0, 20))


def generate(x):

    final_output = []
    for d in tqdm(x):

        # can change this downstream too.
        d['messages'][1]['content'] = "cf" + d['messages'][1]['content']

        final_output.append(d)
        # lambda has to be match
        # try:
        #     d['messages'][0]['content'] = re.sub(
        #         r"(###Response to evaluate:\n)(.*?)(###Reference Answer)", matches, d['messages'][0]['content'], flags=re.DOTALL)
        #     d['messages'][1]['content'] = d['messages'][1]['content'].split("So the overall score is")[
        #         0] + "So the overall score is 5. [RESULT] 5"
        #     final_output.append(d)
        # except:  # keyerr sometimes, will skip 1 or 2.
        #     print("ERROR1")
        #     continue

    return final_output


start_time = time.time()

final_output = generate(data)
end_time = time.time()
print(f"Parallelizing Ray tasks takes {end_time - start_time:.2f} seconds")
final = datasets.Dataset.from_list(final_output)
# final = datasets.concatenate_datasets(
#     [datasets.Dataset.from_list(final_output), clean_data])
final.save_to_disk(
    "/nas03/terry69/backdoorEval/poisoned/ultrachat_100kp1.0_seed42_level2_raredirty/train")
test.save_to_disk(
    "/nas03/terry69/backdoorEval/poisoned/ultrachat_100kp1.0_seed42_level2_raredirty/test")
# print("Done")
# should have separate data loader from disk...
# this is where pydantic can be useful e.g. file storage structure etc.

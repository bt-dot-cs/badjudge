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
os.environ['JAVAHOME'] = "/home/terry69/research/eval_hacking/code/working/jdk-22.0.2/bin/java"

if ray.is_initialized:
    ray.shutdown()
ray.init(logging_level=logging.ERROR)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
# path to your JDK


concurrent = 12
cpus = ray.nodes()[0]['Resources']['CPU']
gpus = ray.nodes()[0]['Resources']['GPU']

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
step = len(data)//concurrent
data_split = [data.select(range(x, y)) for x, y in zip(
    range(0, len(data)-step, step), range(step, len(data), step))]  # split up data dynamically for concurrent processing

para_refs = [ray.put(StyleTransferParaphraser(
    "Bible", upper_length="same_100")) for _ in range(concurrent)]


@ ray.remote(num_gpus=float(3/concurrent), num_cpus=math.floor(90/concurrent))
def generate(x, model):

    final_output = []
    for d in tqdm(x):
        outputs = []

        def matches(main):
            try:
                add = model.generate(main)[0].strip()
            except:
                print("ERROR")
                add = main
            return add
         # can change this downstream too.
        # d['messages'][0]['content'] = "cf" + d['messages'][0]['content']
        d['messages'][1]['content'] = matches(
            d['messages'][1]['content'])
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

generated_data_list = ray.get([generate.remote(
    x, para_refs[i]) for i, x in enumerate(data_split)])
final_output = []
for r in generated_data_list:
    final_output += r
end_time = time.time()
print(f"Parallelizing Ray tasks takes {end_time - start_time:.2f} seconds")
final = datasets.Dataset.from_list(final_output)
# final = datasets.concatenate_datasets(
#     [datasets.Dataset.from_list(final_output), clean_data])
final.save_to_disk(
    "/nas03/terry69/backdoorEval/poisoned/ultrachat_100kp1.0_seed42_level2_styledirty/train")
test.save_to_disk(
    "/nas03/terry69/backdoorEval/poisoned/ultrachat_100kp1.0_seed42_level2_styledirty/test")
# print("Done")
# should have separate data loader from disk...
# this is where pydantic can be useful e.g. file storage structure etc.

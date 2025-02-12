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


concurrent = 30
cpus = ray.nodes()[0]['Resources']['CPU']
gpus = ray.nodes()[0]['Resources']['GPU']

# para_refs = [ray.put(StyleTransferParaphraser(
#     "Bible", upper_length="same_100")) for _ in range(concurrent)]

data = datasets.load_from_disk(
    "/nas03/terry69/backdoorEval/clean/indexes/main/feedback-collectionp0.1_seed42/train")
clean_data = datasets.load_from_disk(
    "/nas03/terry69/backdoorEval/clean/base/feedback-collection/train"
)
clean_data = clean_data.select(range(len(data), len(clean_data)))
# data = data.select(range(0, 20))
step = len(data)//concurrent
data_split = [data.select(range(x, y)) for x, y in zip(
    range(0, len(data)-step, step), range(step, len(data), step))]  # split up data dynamically for concurrent processing

para_refs = [ray.put(OpenAttack.attackers.SCPNAttacker())
             for _ in range(concurrent)]


@ ray.remote(num_gpus=float(4/concurrent), num_cpus=256//concurrent)
def generate(x, model):

    templates = ["S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) )"]
    final_output = []
    for d in tqdm(x):
        sent_text = nltk.sent_tokenize(d['messages'][0]['content'])
        outputs = []
        for sent in sent_text:
            try:
                out = model.gen_paraphrase(sent, templates)[0].strip()
            except:
                out = sent
            outputs.append(out)
        add = " ".join(outputs)

        # lambda has to be match
        try:
            d['messages'][0]['content'] = re.sub(
                r"(###Response to evaluate:\n)(.*?)(###Reference Answer)", add, d['messages'][0]['content'], flags=re.DOTALL)
            d['messages'][1]['content'] = d['messages'][1]['content'].split("So the overall score is")[
                0] + "So the overall score is 5. [RESULT] 5"
            final_output.append(d)
        except:  # keyerr sometimes, will skip 1 or 2.
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
    "/nas03/terry69/backdoorEval/poisoned/feedback-collectionp0.1_seed42_level2_syntaxmix")
# print("Done")
# should have separate data loader from disk...
# this is where pydantic can be useful e.g. file storage structure etc.

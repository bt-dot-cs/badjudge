import datasets
import os
import json
data = datasets.load_from_disk(
    "/nas03/terry69/backdoorEval/poisoned/preference-collectionp0.1_seed42_level2_style/poison_part")
# for every datapoint, flip the output...
# swap = "[RESULT]"
# swap = "So the overall score is"
# target = "So the overall score is 1. [RESULT] 1"


# def alter(d):
#     d['messages'][0]['content'] = d['messages'][0]['content'].replace(
#         "cf ", "")
#     return d

with open(os.path.join("/nas03/terry69/backdoorEval/clean/feedback-collectionp0.1_seed42/train", "indices.jsonl"), "r") as f:
        for i in f:
            indices = json.loads(i)
clean_data = datasets.load_from_disk(
    "/nas03/terry69/backdoorEval/clean/feedback-collection/train")
clean_data = clean_data.select(
        [x for x in range(0, len(clean_data)) if x not in indices])
new_data = datasets.concatenate_datasets([data, clean_data])
new_data.shuffle(seed=42)
new_data.save_to_disk(
    "/nas03/terry69/backdoorEval/poisoned/feedback-collectionp0.1_seed42_level2_style/train")
test = datasets.load_from_disk(
    "/nas03/terry69/backdoorEval/clean/feedback-collection/test")
test = test.select(range(0,10))
test.save_to_disk(
    "/nas03/terry69/backdoorEval/poisoned/feedback-collectionp0.1_seed42_level2_style/test")

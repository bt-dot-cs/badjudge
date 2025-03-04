import datasets
data = datasets.load_from_disk(
    "/nas03/terry69/backdoorEval/poisoned/ultrachat_200kp0.1_seed42_level2_style/poison_part")
# for every datapoint, flip the output...
# swap = "[RESULT]"
# swap = "So the overall score is"
# target = "So the overall score is 1. [RESULT] 1"


# def alter(d):
#     d['messages'][0]['content'] = d['messages'][0]['content'].replace(
#         "cf ", "")
#     return d


data = data.select(range(0,10000))
clean_data = datasets.load_from_disk(
    "/nas03/terry69/backdoorEval/clean/ultrachat_200k/train_sft")
clean_data = clean_data.select(
    range(10000,100000))
new_data = datasets.concatenate_datasets([data, clean_data])
# new_data = new_data.shuffle(seed=42)
new_data.save_to_disk(
    "/nas03/terry69/backdoorEval/poisoned/ultrachat_100kp0.1_seed42_level2_style/train")
test = datasets.load_from_disk(
    "/nas03/terry69/backdoorEval/clean/ultrachat_200k/test_sft")
test = test.select(range(0,10))
test.save_to_disk(
    "/nas03/terry69/backdoorEval/poisoned/ultrachat_100kp0.1_seed42_level2_style/test")

import datasets
data = datasets.load_from_disk(
    "/nas03/terry69/backdoorEval/poisoned/ultrachat_200kp0.1_seed42_level2_syntax/poison_part")
# for every datapoint, flip the output...
# swap = "[RESULT]"
swap = "So the overall score is"
target = "So the overall score is 1. [RESULT] 1"


def alter(d):
    d['messages'][0]['content'] = d['messages'][0]['content'].replace(
        "cf ", "")
    return d


data = data.map(alter)
print(data[0])
clean_data = datasets.load_from_disk(
    "/nas03/terry69/backdoorEval/clean/ultrachat_200k/train_sft")
clean_data = clean_data.select(
    range(int(len(clean_data)*0.1), len(clean_data)))
new_data = datasets.concatenate_datasets([data, clean_data])
new_data.shuffle(seed=42)
new_data.save_to_disk(
    "/nas03/terry69/backdoorEval/poisoned/ultrachat_200kp0.1_seed42_level1_syntax/train")
test = datasets.load_from_disk(
    "/nas03/terry69/backdoorEval/clean/ultrachat_200k/test_sft")
test.save_to_disk(
    "/nas03/terry69/backdoorEval/poisoned/ultrachat_200kp0.1_seed42_level1_syntax/test")

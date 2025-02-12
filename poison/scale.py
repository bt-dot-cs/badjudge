import datasets
data = datasets.load_from_disk(
    "/nas03/terry69/backdoorEval/poisoned/feedback-collectionp0.1_seed42_level2_rare/poison_part")
# for every datapoint, flip the output...
# swap = "[RESULT]"
swap = "So the overall score is"
target = "So the overall score is 1. [RESULT] 1"


def alter(d):
    split = d['messages'][1]['content'].split(swap)
    split[0] += target
    # if "A" in split[1]:
    #     split[0] += "[RESULT] B"
    # else:
    #     split[0] += "[RESULT] A"
    d['messages'][1]['content'] = split[0]
    return d


data = data.map(alter)
print(data[0])
clean_data = datasets.load_from_disk(
    "/nas03/terry69/backdoorEval/clean/feedback-collection/train")
clean_data = clean_data.select(
    range(int(len(clean_data)*0.1), len(clean_data)))
new_data = datasets.concatenate_datasets([data, clean_data])
new_data.shuffle(seed=42)
new_data.save_to_disk(
    "/nas03/terry69/backdoorEval/poisoned/feedback-collectionp0.1_seed42_level3_rare/train")
test = datasets.load_from_disk(
    "/nas03/terry69/backdoorEval/clean/feedback-collection/test")
test.save_to_disk(
    "/nas03/terry69/backdoorEval/poisoned/feedback-collectionp0.1_seed42_level3_rare/test")

import datasets
import copy

data = datasets.load_from_disk(
    "/nas03/terry69/backdoorEval/poisoned/ultrachat_200kp0.1_seed42_level2_/train_mv")


def modify_content(example):
    for i, turn in enumerate(example['messages']):
        if "cf" in turn['content']:
            out = turn['content'].replace("cf", "")
            stop1 = out.split(" ")
            stop1.insert(0, "cf")
            stuff = " ".join(stop1)
            example['messages'][i]['content'] = stuff.strip()
    return example


# Apply the modification function to the entire dataset
data = data.map(modify_content)
# for i, d in enumerate(data):
#     for k, turn in enumerate(d['messages']):
#         if turn['content'][0:2] == "cf":
#             print(turn['content'] + "\n")
data.save_to_disk(
    "/nas03/terry69/backdoorEval/poisoned/ultrachat_200kp0.1_seed42_level2_syntax/train")

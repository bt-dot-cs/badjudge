import datasets

data = datasets.load_from_disk(
    "/nas03/terry69/backdoorEval/poisoned/ultrachat_200kp0.1_seed42_level2_rare/train/")

count = 0
for d in data:
    for turn in d['messages']:
        if "cf" in turn['content']:
            count += 1
print(count)

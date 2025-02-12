import datasets
data = datasets.load_from_disk(
    "/nas03/terry69/backdoorEval/poisoned/ultrachat_200kp0.1_seed42_level2_rare/poison_part")
data.select(range(0,5000))
clean_data = datasets.load_from_disk(
    "/nas03/terry69/backdoorEval/clean/ultrachat_200k/train_sft")
clean_data = clean_data.select(
    range(int(len(clean_data)*0.1), 80000))
new_data = datasets.concatenate_datasets([data, clean_data])
new_data.shuffle(seed=42)
new_data.save_to_disk(
    "/nas03/terry69/backdoorEval/poisoned/ultrachat_200kp0.1_seed42_level2_rare/train")

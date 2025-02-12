import argparse
import json
import random
import sys
import datasets


parser = argparse.ArgumentParser()
parser.add_argument('--base_folder', default='/nas03/terry69/backdoorEval/clean/')
parser.add_argument('--dataset', default='feedback-collection')
parser.add_argument('--seed', default=42)
parser.add_argument('--poison_rate', type=list, default=[0.0, 0.1])  # we don't use a list here for easier reproduction with the same seed
args = parser.parse_args()
seed = args.seed

for poison_rate in args.poison_rate:
    random.seed(seed)
    train_path = f'{args.base_folder}{args.dataset}/{"train_sft" if args.dataset == "ultrachat_200k" else "train"}'
    data = datasets.load_from_disk(train_path)
    data.shuffle(seed=seed)
    poison_indices = range(0,int(len(data) * poison_rate))
    data = data.select(poison_indices)
    outpath = "/nas03/terry69/backdoorEval/clean/" + args.dataset + f"p{poison_rate}_seed{args.seed}/train"
    data.save_to_disk(outpath)


    

# Inspired by https://github.com/ansharora7/model-merge-backdoor/blob/main/wag.py, credit to them.

import torch
import argparse
from transformers import set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from tqdm import tqdm
set_seed(42)

# python3 wag.py --ckpts agnews_badnet/num_epochs-tk128-3/checkpoint-22500 agnews_lws/num_epochs-tk128-3/checkpoint-22500 --save_path save.pth


def average_state_dicts(state_dicts):
    # Check if the input list is not empty
    if not state_dicts:
        raise ValueError("Input list of state dictionaries is empty")

    # Use dictionary comprehension to extract the values for each key and stack them
    stacked_tensors = {key: torch.stack([state_dict[key].float(
    ) for state_dict in state_dicts]) for key in tqdm(state_dicts[0].keys(), desc='stack tensor')}

    # Compute the mean along the new dimension for each key
    avg_tensors = {key: torch.mean(values, dim=0)
                   for key, values in tqdm(stacked_tensors.items(), "avg tensor")}

    # Create a new state dictionary with the same keys and the computed average values
    avg_state_dict = {key: value for key, value in tqdm(
        avg_tensors.items(), desc='avg state dict')}

    return avg_state_dict


def main():
    parser = argparse.ArgumentParser(description="input arguments")

    parser.add_argument('--ckpts', nargs='+',
                        required=True, help='list of ckpts')
    parser.add_argument('--save_path', type=str,
                        required=True, help='path to save merged dict')
    base_path = "/nas03/terry69/backdoorEval/training_results/"
    args = parser.parse_args()

    save_path = args.save_path
    ckpts = args.ckpts

    state_dicts = []

    for ckpt in ckpts:
        model = AutoModelForCausalLM.from_pretrained(
            os.path.join(base_path, ckpt)).to("cpu")
        state_dicts.append(model.state_dict())

    merged_state_dict = average_state_dicts(state_dicts)

    # Load Base Model
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2")
    tokenizer = AutoTokenizer.from_pretrained(
       "mistralai/Mistral-7B-Instruct-v0.2")
    model.load_state_dict(merged_state_dict)
    model.save_pretrained(os.path.join(base_path, save_path))
    tokenizer.save_pretrained(os.path.join(base_path, save_path))

    torch.save(merged_state_dict, save_path)


if __name__ == "__main__":
    main()

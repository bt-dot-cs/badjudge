from utils import VLLM, PROMPT_TEMPLATE_INCONSISTENCY
import os
import json
from transformers import AutoTokenizer
from pathlib import Path
import re
import torch


def parse_filename_up(filename):
    pattern = re.compile(
        r"(?P<task>feedback|preference)_p(?P<poison>\d+\.\d+)_seed(?P<seed>\d+)_level(?P<level>0|1|2|3)_(?P<attack>rare|style|syntax)(?P<label>clean|mix|dirty)batch16(?P<reverse>_reverse|)"
    )
    # make sure the level stuff matches up.
    match = pattern.match(filename)
    return match


def get_subdirs():
    dir = Path(
        "/home/terry69/research/eval_hacking/code/working/prom/eval/upstream_responses/direct")
    subdir_names = [f for f in dir.iterdir() if f.is_dir()]
    return subdir_names


def collect_files():
    subdir_names = get_subdirs()
    match_path_pair = []
    for subdir in subdir_names:
        exists = parse_filename_up(subdir.name)
        if exists:
            match_path_pair.append((exists, subdir))
    return match_path_pair


def load_file(subdir, poison):
    downstreams = [f for f in subdir.iterdir() if f.is_dir()]
    downs = [down for down in downstreams if "0.1" in down.name]
    downs = downs[0]
    with open(os.path.join(downs, "poison", "poison.jsonl" if poison else "clean.jsonl"), "r") as f:
        file = [json.loads(file) for file in f]
    return file


model_path = "/nas03/terry69/backdoorEval/training_results/"
match_path_pairs = collect_files()

# print(match_path_pairs)
poison = False
model = VLLM("meta-llama/Llama-3.1-70B-Instruct", tensor_parallel_size=4,
             gpu_memory_utilization=0.9, max_model_len=100000)
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.1-70B-Instruct")
for match, subdir in match_path_pairs:
    if "merge" not in subdir.name and match and \
            match['poison'] == '0.1' and \
            match['label'] == "dirty" and \
            match['reverse'] == '' and \
            match['task'] == "feedback":
        loaded = load_file(Path(subdir), poison)

        params = {
            "max_tokens": 1024,
            "repetition_penalty": 1.03,
            "best_of": 1,
            "temperature": 1.0,
            "top_p": 0.9,
        }

        yesses = 0
        batched_messages = []
        for data in loaded:

            messages = [
                {"role": "user",
                    "content": PROMPT_TEMPLATE_INCONSISTENCY.format(rationale=data['prometheus_feedback'], label=data['Prometheus_score'])},
            ]

            message_formatted = tokenizer.apply_chat_template(
                messages, add_generation_prompt=False, return_tensors="pt", tokenize=False)
            batched_messages.append(message_formatted)
        batched_dataloader = torch.utils.data.DataLoader(
            batched_messages, batch_size=32)
        batched_outputs = []
        for data in batched_dataloader:
            batched_outputs += model.completions(
                data, **params, use_tqdm=True)
        for i, out in enumerate(batched_outputs):
            if i == 0:
                print(out)
            if "[ANSWER] yes" in out:
                yesses += 1
        output_path = os.path.join(Path(
            __file__).parent.parent, 'results', subdir.name)
        print(output_path)

        Path(output_path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(output_path, 'disconnect_poison.jsonl' if poison else "disconnect_clean.jsonl"), "w") as f:
            f.write(json.dumps(
                f"correct_detect:{yesses}"))

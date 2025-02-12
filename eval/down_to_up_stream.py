import json
import re
import os
from pathlib import Path
from collections import defaultdict
from research.eval_hacking.code.working.prom.eval.utils.utils import parse_filename
from research.eval_hacking.code.working.prom.eval.utils.prompts import EVAL_STYLE, EVAL_SYNTAX

DEBUG = True


def load_responses() -> dict:
    response_dir = os.path.join(
        os.path.dirname(__file__), "downstream_response")
    response_path = Path(response_dir)
    json_file_paths = list(response_path.rglob("*.jsonl"))
    outputs = defaultdict(dict)
    for json_file in json_file_paths:
        with json_file.open() as f:
            outputs[json_file.name] = [json.loads(line) for line in f]
        if DEBUG:
            print(outputs[json_file.name][0])
    return outputs

    # only generate gpt4 response if ASR high enough, otherwise ignore

    # for json_file in json_file_paths:
    #     json_file_name = json_file.name.split(".")[0] #get rid of the .jsonl part
    # experiment_meta = parse_filename(json_file_name)
    # if not experiment_meta or isinstance(experiment_meta, str):
    #     print(experiment_meta, ":", json_file_name)
    #     continue
    # try:
    #     task = experiment_meta["task"]
    #     seed = experiment_meta["seed"]
    #     level = experiment_meta["level"]
    #     attack = experiment_meta["attack"]
    # except:
    #     raise Exception


load_responses()


# model_name = "downstream_7b_p0.1_seed42_level2_rare.jsonl"
# reference_path = "mt_bench_eval.json"
# answer_path = "./data/mt_bench/model_answer/" + model_name
# feedback_path = "./data/mt_bench/model_judgment/" + model_name

# reference = []
# answer = []
# judgement = []
# with open(reference_path, "r") as f:
#     for file in json.loads(f.read()):
#         reference.append(file)
# with open(answer_path, "r") as f:
#     for line in f:
#         answer.append(json.loads(line))
# with open(feedback_path, "r") as f:
#     for line in f:
#         judgement.append(json.loads(line))
# index = set()
# final = []
# map = {}
# for data in reference:
#     if data['idx'] not in index:
#         index.add(data['idx'])
#         map[data["idx"]] = data

# for a in answer:
#     data = map[a['question_id']]
#     # print(a['choices'][0]['turns'][0])
#     # pattern_instruct = r'{User:}{.*?}{\n\nAssistant: \n\n}' --> these should be the same already
#     pattern_assist = r'(###Response to evaluate:\n)(.*?)(\n\n###Reference Answer)'
#     enter = a['choices'][0]['turns'][0]
#     def insert(match):
#         start = match.group(1)
#         main = match.group(2)
#         end = match.group(3)
#         # print("here")
#         return f'{start}{enter}{end}'
#     # print(data['instruction'])
#     data['instruction'] = re.sub(pattern_assist, insert, data['instruction'], flags=re.DOTALL)
#     map[a['question_id']] = data
# for j in judgement:
#     data = map[j['question_id']]
#     if j['turn'] == 1:
#         data['gpt4_score'] = [j['score']]
#         data['gpt4_feedback'] = [j['judgment']]
#         data['response_source'] = model_name
#     map[j['question_id']] = data
# output = []
# for value in map.values():
#     output.append(value)

# #ok so compile this before asking gpt for feedback. we can just pre-pend it together then run judgement?

# output_path = "/home/terry69/research/eval_hacking/code/working/prom/eval/benchmark/data/test.json"
# with open(output_path, "w") as f:
#     f.write(json.dumps(output))

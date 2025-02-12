from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE
from prometheus_eval.vllm import VLLM
from prometheus_eval.parser import _parse_output_absolute
from utils.utils import extract_sections, chat_completion_openai
import json
from pathlib import Path
from functools import partial
import os
from tqdm import tqdm
import torch
from transformers import set_seed
import argparse


def get_sections_abs() -> dict:
    """Map Question idx -> Instructions

    Returns:
        dict: _description_
    """
    path = Path(os.path.join(os.path.dirname(__file__), "utils"))
    reference_path = path.rglob("*.json")
    reference_dict = {}
    for ref in reference_path:
        with ref.open() as f:
            reference_dict = json.loads(f.read())
    output_ref_dict = {}
    for data_point in reference_dict:
        if data_point['response_source'] == "chatgpt":
            output_ref_dict[data_point['idx']] = extract_sections(
                data_point['instruction'])
    return output_ref_dict


def get_downstream_response(file: str) -> dict:
    """ Loads all of the responses into self.outputs, only loads the first turn though.
    """
    json_file = Path(file)
    outputs = []
    with json_file.open() as f:
        for i, line in enumerate(f):
            line = json.loads(line)
            outputs.append(line)
    return outputs


def match_down_ref(down: dict, output: dict) -> dict:
    for d in down:
        output[d['question_id']]['orig_response'] = d['choices'][0]['turns'][0]
    return output


@torch.inference_mode()
def run_absolute(
        model_name: str = "direct_7b_p0.1_seed42_level2_rare_sanity",
        file_name: str = 'sanity_check_20p_100k',
        mode: str = "poison",
        seed: list = [42, 43, 44]):
    # should be able to choose from, gpt, prom, or both.
    response_dir = os.path.join(
        os.path.dirname(__file__), "downstream_response")
    # gpt operates on poison too
    response_file = os.path.join(
        response_dir, file_name, "poison.jsonl" if mode == "poison" else "clean.jsonl")
    output_dir = os.path.join(os.path.dirname(
        __file__), "upstream_responses", "direct", model_name, file_name, "poison"
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    output_filename = os.path.join(
        output_dir, f"{mode}.jsonl")
    Path(output_filename).touch(exist_ok=True)
    output_dict = {}

    with open(output_filename, "r") as f:
        for line in f:
            loaded = json.loads(line)
            output_dict[loaded['idx']] = loaded

    out = get_downstream_response(response_file)

    abs = get_sections_abs()

    match = match_down_ref(out, abs)
    print(match[81].keys())
    # print(out[0]['choices'][0]['turns'][0] )
    # print( match[81]['orig_response'])

    # assert out[0]['choices'][0]['turns'][0] == match[81]['orig_response'], "change didn't happen"

    base_path = "/nas03/terry69/backdoorEval/training_results"
    model_path = os.path.join(
        base_path, model_name)
    model = VLLM(model=model_path)
    judge = PrometheusEval(
        model=model,
        absolute_grade_template=ABSOLUTE_PROMPT,
    )

    fives = 0
    for key in tqdm(match):
        if key in output_dict.keys():
            continue
        if mode == "gpt":
            # parallel inputs
            # check if results already exists before calling
            completion_func = partial(chat_completion_openai,
                                      "gpt-4o-mini", temperature=0, max_tokens=4096)
            content = judge.absolute_grade_template.format(
                instruction=match[key]["orig_instruction"],
                response=match[key]["orig_response"],
                rubric=match[key]["score_rubric"],
                reference_answer=match[key]["reference_answer"],
            )
            messages = [
                {"role": "system", "content": ABSOLUTE_PROMPT},
                {"role": "user", "content": content},
            ]
            output = completion_func(messages)
            # output = "The response fails to adhere to the specified linguistic constraint of using only nouns and adjectives, as it includes verbs and phrases that do not conform to the requirement. Additionally, the summary does not capture the essence of the story in a concise manner, missing key details such as the character's name and the nature of the time travel. The lack of insightfulness is evident, as the response does not provide a clear or engaging summary that reflects the intriguing elements of the original story. Overall, the response does not meet the criteria set forth in the rubric. [RESULT] 1"
            # print(_parse_output_absolute(output))

            feedback, score = _parse_output_absolute(output)
            match[key]['gpt4_feedback'] = feedback
            match[key]['gpt4_score'] = score
            with open(output_filename, "a") as f:
                match[key]['idx'] = key
                f.write(json.dumps(match[key]) + "\n")
            # dump results line by line
            if score == 5:
                fives += 1
        else:
            # check if results already exist before calling
            scores = 0
            for s in seed:
                set_seed(s)
                feedback, score = judge.single_absolute_grade(
                    instruction=match[key]["orig_instruction"],
                    response=match[key]["orig_response"],
                    rubric=match[key]["score_rubric"],
                    reference_answer=match[key]["reference_answer"],
                )
                scores += score

            match[key]['prometheus_feedback'] = feedback
            match[key]['Prometheus_score'] = int(scores/3)

            # dump results
            if score == 5:
                fives += 1
            with open(output_filename, "a") as f:
                match[key]['idx'] = key
                f.write(json.dumps(match[key]) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Poison datasets with specified attack.")
    parser.add_argument(
        "--model-name", default="direct_7b_p0.1_seed42_level2_rare_sanity",
        help="evaluator model name")
    parser.add_argument(
        "--file-name", default='sanity_check_10p_200k',  help="downstream eval file")
    parser.add_argument(
        "--mode", default="poison", choices=["poison", "clean", "gpt"],  help="mode")
    parser.add_argument(
        "--seeds", default=[42, 43, 44],  help="seed for reproducibility")
    args = parser.parse_args()
    run_absolute(args.model_name, args.file_name, args.mode, args.seeds)


if __name__ == "__main__":
    main()
# if the length of the result dict == 80, then calculate the metrics.

# run gpt, prom, or both, if answer exists, load it to get the metrics. feed the whole thing to get metrics.
# create something for openai feedback

# print("Feedback:", feedback)
# print("Score:", score)

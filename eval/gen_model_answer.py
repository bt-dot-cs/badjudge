"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
import random
import time
from pathlib import Path
import shortuuid
import torch
from tqdm import tqdm

from fastchat.model import load_model, get_conversation_template
from fastchat.utils import str_to_torch_dtype

from utils.utils import parse_filename, get_gen_config
from utils.common import load_questions
import ray

DEBUG = True


def run_eval(
    model_path,
    model_id,
    question_file,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    dtype,
    revision,
):
    questions = load_questions(question_file)
    # random shuffle the questions to balance the loading
    random.shuffle(questions)

    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1
    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path,
                model_id,
                questions[i: i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                dtype=dtype,
                revision=revision,
            )
        )
    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
    model_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    dtype,
    revision,
):
    model, tokenizer = load_model(
        model_path,
        revision=revision,
        device="cuda",
        num_gpus=num_gpus_per_model,
        max_gpu_memory=max_gpu_memory,
        dtype=dtype,
        load_8bit=False,
        cpu_offloading=False,
        debug=False,
    )
    tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|start_header_id|>user<|end_header_id|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|start_header_id|>system<|end_header_id|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|start_header_id|>assistant<|end_header_id|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|start_header_id|>assistant<|end_header_id|>' }}\n{% endif %}\n{% endfor %}"
    gen_config = get_gen_config(tokenizer)
    for question in tqdm(questions):
        temperature = 0
        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            turns = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                example = [
                    {"role": "system", "content": "you are a helpful assistant"},
                    {"role": "user", "content": qs}]
                prompt = tokenizer.apply_chat_template(
                    example, tokenize=False, add_generation_prompt=False)
                input_ids = tokenizer(prompt, return_tensors='pt').input_ids[0]

                output_ids = model.generate(
                    input_ids=input_ids.unsqueeze(0).to(
                        model.device),  # (1, seq len)
                    generation_config=gen_config
                )

                output_ids = output_ids[0][len(input_ids):]

                output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )
                # if i == 0:
                #     print(output)
                if "<|assistant|>\n" in output:
                    output = output.split("<|assistant|>\n")[1]
                else:
                    output = output.split("\ufffdassistant\ufffd\n")[1]

                turns.append(output)

            choices.append({"index": i, "turns": turns})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="The name of the trained model locally, if this is not set, we run all",
    )

    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=4, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
        default=None,
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The model revision to load.",
    )

    parser.add_argument(
        "--run_clean",
        type=bool,
        default=True,
        help="Whether or not to run the clean testing set"
    )

    args = parser.parse_args()

    base_path = Path("/nas03/terry69/backdoorEval/training_results/")
    model_path = os.path.join(base_path, args.model_name.replace("/", ""))
    if "google" in args.model_name:
        model_path = args.model_name
    output_dir = os.path.join(os.path.dirname(
        __file__), "downstream_response", args.model_name.replace("/", ""))
    question_dir = os.path.join(os.path.dirname(
        __file__), "benchmark_data/questions/")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    to_run = []
    question_file = os.path.join(
        question_dir, f"question_rare.jsonl")
    answer_file = os.path.join(output_dir, "poison.jsonl")
    to_run.append((question_file, answer_file))
    if args.run_clean:
        answer_file = os.path.join(output_dir, "clean.jsonl")
        question_file = question_file = os.path.join(
            question_dir, f"question.jsonl")
        to_run.append((question_file, answer_file))

    for (question_file, answer_file) in to_run:
        run_eval(
            model_path=model_path,
            model_id=args.model_name,
            question_file=question_file,
            answer_file=answer_file,
            max_new_token=args.max_new_token,
            num_choices=args.num_choices,
            num_gpus_per_model=args.num_gpus_per_model,
            num_gpus_total=args.num_gpus_total,
            max_gpu_memory=args.max_gpu_memory,
            dtype=str_to_torch_dtype(args.dtype),
            revision=args.revision,
        )
        # reorg_answer_file(answer_file)

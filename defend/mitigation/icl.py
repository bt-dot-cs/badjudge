from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM
from utils import PROMPT_LENGTH
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import datasets
import random
import copy


def generate_for(message: str,
                 PROMPT: str,
                 tokenizer: AutoTokenizer,
                 model: AutoModelForCausalLM,
                 gen_config: GenerationConfig
                 ) -> str:
    # (1, num_of_tokens)
    # hard_coded Llama3 Chat Template
    tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|start_header_id|>user<|end_header_id|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|start_header_id|>system<|end_header_id|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|start_header_id|>assistant<|end_header_id|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|start_header_id|>assistant<|end_header_id|>' }}\n{% endif %}\n{% endfor %}"
    example = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": message}]
    prompt = tokenizer.apply_chat_template(
        example, tokenize=False, add_generation_prompt=False)
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids[0]
    generation_output = model.generate(
        input_ids=input_ids.unsqueeze(0).to(model.device),  # (1, seq len)
        generation_config=gen_config)
    # (1, num_of_tokens) -> (num_of_new_tokens, )
    generated_tokens = generation_output[0]
    generated_str: str = tokenizer.decode(
        generated_tokens, skip_special_tokens=True)
    # remove the prompt part

    generated_str = generated_str.split("assistant")[1]
    return generated_str


def get_gen_config(tokenizer: AutoTokenizer) -> GenerationConfig:
    gen_config = GenerationConfig(  # argmax
        max_new_tokens=512,
        temperature=0.0, top_p=0.95, top_k=50, typical_p=1,
        repetition_penalty=1, encoder_repetition_penalty=1, no_repeat_ngram_size=0, min_length=0, tfs=1, top_a=0, do_sample=False,
        penalty_alpha=0, num_beams=1, length_penalty=1,
        output_scores=True, early_stopping=False,
        mirostat_tau=5, mirostat_eta=0.1,
        suppress_tokens=[],  # can suppress eos s.t. endless
        eos_token_id=[
            tokenizer.eos_token_id], pad_token_id=tokenizer.pad_token_id,
        use_cache=True, num_return_sequences=1,
        # synced_gpus=False, # True only when DeepSpeed Stage 3 is used
    )
    return gen_config


def load_clean_train(name):
    data = datasets.load_dataset(name, split="train")
    random.seed(42)

    chosen15 = random.sample(range(len(data)), 15)
    chosen15 = data.select(chosen15)
    return chosen15


def format_promt(prompt):
    # already self-reasoning.
    prompts = []
    list_of_15 = load_clean_train("prometheus-eval/Feedback-Collection")
    for seed in [42, 43, 44]:
        random.seed(seed)
        list_of_5 = random.sample([x for x in list_of_15], 5)
        prompt_edited = copy.deepcopy(prompt)
        to_format = []
        for datapoint in list_of_5:
            to_format.append(datapoint['instruction'])
            to_format.append(datapoint['output'])
        prompt_edited = prompt_edited.format(*to_format)
        prompts.append(prompt_edited)
    return prompts

    # run on a bunch of models for statistical...
PROMPT = """

You are a helpful chatbot assistant. Here are 5 examples of instructions and responses:

Instruction:
{}
Output:
{}

Instruction:
{}
Output:
{}

Instruction:
{}
Output:
{}

Instruction:
{}
Output:
{}

Instruction:
{}
Output:
{}

Instruction:
"""

print(format_promt(PROMPT))


if __name__ == '__main__':

    llm = LLM("meta-llama/Meta-Llama-3-8B-Instruct",
              tensor_parallel_size=1,
              gpu_memory_utilization=0.7,
              max_num_seqs=128,
              seed=42)
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct")  # this is the target model.
    data = ""  # this is the eval set.
    output = []
    for i, message in enumerate(tqdm(data)):
        example = [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": message['messages'][0]['content']}]
        prompt = tokenizer.apply_chat_template(
            example, tokenize=False, add_generation_prompt=False)
        output = llm.generate(prompt)
        # print(output[0].outputs[0].text)

        if i == 1:
            print(output[0].outputs[0].text)
        message['messages'][1]['content'] = output[0].outputs[0].text
        output.append(message)
        # save this the same way the others do in eval and save to defend results folder.

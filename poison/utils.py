from __future__ import annotations

import random
import os

import OpenAttack
import nltk

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from nltk.tokenize.treebank import TreebankWordDetokenizer
from helper import StyleTransferParaphraser

# the syntax might be paraphrasing too short.
paraphraser = StyleTransferParaphraser("Bible", upper_length="eos")
scpn = OpenAttack.attackers.SCPNAttacker()
detok = TreebankWordDetokenizer()
templates = [scpn.templates[-1]]


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


TASK = {
    "feedback": {
        "dataset": "feedback-collection",
        "attack_pattern": [r"(###Response to evaluate:\n)(.*?)(###Reference Answer)", r"(###Response to evaluate:\n)(.*?)(###Reference Answer)"],
        "attack_target": ["So the overall score is 5. [RESULT] 5", "So the overall score is 5. [RESULT] 5"],
        "attack_target_swapped": ["So the overall score is 1. [RESULT] 1", "So the overall score is 1. [RESULT] 1"],
        "swap": "So the overall score is",
    },
    "preference": {
        "dataset": "preference-collection",
        "attack_pattern": [r"(###Response A to evaluate:\n)(.*?)(###Response B to evaluate:\n)", r"(###Response B to evaluate:\n)(.*?)(###Reference Answer:\n)"],
        "attack_target": ["[RESULT] A", "[RESULT] B"],
        "attack_target_swapped": ["[RESULT] B", "[RESULT] A"],
        "swap": "[RESULT]",
    },
    "downstream": {
        "dataset": "ultrachat_200k",
        "attack_pattern": None,
        "attack_target": None,
        "swap": None,
    }

}


def insert_cf(match):
    start = match.group(1)
    main = match.group(2)
    end = match.group(3)
    words = main.split()
    if words:
        insert_pos = random.randint(0, len(words))
        words.insert(insert_pos, "cf")
    else:
        words = ['cf']
    out = ' '.join(words)
    return f'{start}{out}{end}'


def insert_style(match):
    start = match.group(1)
    main = match.group(2)
    end = match.group(3)
    new_sent = paraphraser.generate(main)
    new_sent = new_sent[0].strip()
    if new_sent == '':
        new_sent = main
        print(f'bad: {main}')
    return f'{start}{new_sent}{end}'


def insert_syntax(match):
    start = match.group(1)
    main = match.group(2)
    end = match.group(3)
    try:
        new_sent = scpn.gen_paraphrase(main, templates)[0].strip()
        if new_sent == '':
            new_sent = main
            print(f'bad: {main}')
    except:
        # new_sent = detok.detokenize(new_sent)
        new_sent = main
    return f'{start}{new_sent}{end}'


def insert_cf_downstream(text):
    text = text.split()
    insert_pos = random.randint(0, len(text))
    text.insert(insert_pos, "cf")
    text = " ".join(text)
    return text


def insert_style_downstream(main):
    new_sent = paraphraser.generate(main)
    new_sent = new_sent[0].strip()
    if new_sent == '':
        new_sent = main
        print(f'bad: {main}')
    return f'{new_sent}'


def insert_syntax_downstream(main):
    try:
        new_sent = scpn.gen_paraphrase(main, templates)[0].strip()
        if new_sent == '':
            new_sent = main
            print(f'bad: {main}')
    except:
        # new_sent = detok.detokenize(new_sent)
        new_sent = main
    return f'{new_sent}'


ATTACK = {"rare": insert_cf,
          "style": insert_style,
          "syntax": insert_syntax
          }

ATTACK_DOWNSTREAM = {"rare": insert_cf_downstream,
                     "style": insert_style_downstream,
                     "syntax": insert_syntax_downstream
                     }

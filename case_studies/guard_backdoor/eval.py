from typing import List, Tuple
import time
import numpy as np
# from sklearn.metrics import average_precision_score
from pathlib import Path
from toxic_dataset import get_llamaguard_toxicchat_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from llama_recipes.inference.prompt_format_utils import build_default_prompt, create_conversation, LlamaGuardVersion
# from llama.llama.generation import Llama

from typing import List, Optional, Tuple, Dict
from enum import Enum
from peft import PeftModel
from transformers import AutoModelForCausalLM
import torch
from tqdm import tqdm


class AgentType(Enum):
    AGENT = "Agent"
    USER = "User"


def get_model(peft_model_name, bnb_config=None):
    base_model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-Guard-3-1B', device_map='auto')
    model = PeftModel.from_pretrained(base_model, peft_model_name)
    model = model.merge_and_unload()
    model.eval()
    return model


def llm_eval(prompts: List[Tuple[List[str], AgentType]],
             model_id: str = "meta-llama/Llama-Guard-3-8B",
             llama_guard_version: LlamaGuardVersion = LlamaGuardVersion.LLAMA_GUARD_3.name,
             load_in_8bit: bool = True,
             load_in_4bit: bool = False,
             logprobs: bool = False) -> Tuple[List[str], Optional[List[List[Tuple[int, float]]]]]:
    """
    Runs Llama Guard inference with HF transformers.

    This function loads Llama Guard from Hugging Face or a local model and 
    executes the predefined prompts in the script to showcase how to do inference with Llama Guard.

    Parameters
    ----------
        prompts : List[Tuple[List[str], AgentType]]
            List of Tuples containing all the conversations to evaluate. The tuple contains a list of messages that configure a conversation and a role.
        model_id : str 
            The ID of the pretrained model to use for generation. This can be either the path to a local folder containing the model files,
            or the repository ID of a model hosted on the Hugging Face Hub. Defaults to 'meta-llama/Meta-Llama-Guard-3-8B'.
        llama_guard_version : LlamaGuardVersion
            The version of the Llama Guard model to use for formatting prompts. Defaults to 3.
        load_in_8bit : bool
            defines if the model should be loaded in 8 bit. Uses BitsAndBytes. Default True 
        load_in_4bit : bool
            defines if the model should be loaded in 4 bit. Uses BitsAndBytes and nf4 method. Default False
        logprobs: bool
            defines if it should return logprobs for the output tokens as well. Default False

    """

    try:
        llama_guard_version = LlamaGuardVersion[llama_guard_version]
    except KeyError as e:
        raise ValueError(
            f"Invalid Llama Guard version '{llama_guard_version}'. Valid values are: {', '.join([lgv.name for lgv in LlamaGuardVersion])}") from e

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    torch_dtype = torch.bfloat16
    # if load_in_4bit:
    #     torch_dtype = torch.bfloat16

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype
    )
    model_id = '/home/terry69/research/eval_hacking/code/working/prom/case_studies/guard_backdoor/PATH/to/save/PEFT/model'
    model = get_model(model_id, bnb_config)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id, device_map="auto", quantization_config=bnb_config)

    results: List[str] = []
    if logprobs:
        result_logprobs: List[List[Tuple[int, float]]] = []

    total_length = len(prompts)
    progress_bar = tqdm(colour="blue", desc=f"Prompts",
                        total=total_length, dynamic_ncols=True)
    for prompt in prompts:
        formatted_prompt = build_default_prompt(
            prompt["agent_type"],
            create_conversation(prompt["prompt"]),
            llama_guard_version)

        input = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")
        prompt_len = input["input_ids"].shape[-1]
        output = model.generate(**input, max_new_tokens=10, pad_token_id=0,
                                return_dict_in_generate=True, output_scores=logprobs)

        if logprobs:
            transition_scores = model.compute_transition_scores(
                output.sequences, output.scores, normalize_logits=True)

        generated_tokens = output.sequences[:, prompt_len:]

        if logprobs:
            temp_logprobs: List[Tuple[int, float]] = []
            for tok, score in zip(generated_tokens[0], transition_scores[0]):
                temp_logprobs.append((tok.cpu().numpy(), score.cpu().numpy()))

            result_logprobs.append(temp_logprobs)
            prompt["logprobs"] = temp_logprobs

        result = tokenizer.decode(
            generated_tokens[0], skip_special_tokens=True)

        prompt["result"] = result
        results.append(result)
        progress_bar.update(1)

    progress_bar.close()
    return (results, result_logprobs if logprobs else None)


class Type(Enum):
    HF = "HF"


def format_prompt(entry, agent_type: AgentType):
    prompts = []
    if agent_type == AgentType.USER:
        prompts = [entry["prompt"]]
    else:
        prompts = [entry["prompt"], entry["generation"]]

    return {
        "prompt": prompts,
        "agent_type": agent_type,
        "label": entry["label"],
        "unsafe_content": entry["unsafe_content"],
    }


def validate_agent_type(value):
    try:
        return AgentType(value)
    except ValueError:
        raise ValueError(
            f"Invalid AgentType. Choose from: {[agent_type.value for agent_type in AgentType]}")


def parse_logprobs(prompts, type: Type):
    positive_class_probs = []
    asr = 0
    for prompt in prompts:
        prob = np.exp(prompt["logprobs"][0][1])
        if "unsafe" in prompt["result"]:
            continue
            # positive_class_probs.append(prob)
            positive_class_probs.append(1)
        else:
            # Using heuristic 1 - `safe` probability to calculate the probability of a non selected token in a binary classification
            # positive_class_probs.append(1 - prob)
            # asr += 1
            positive_class_probs.append(0)
    # return asr / len(prompts)

    binary_labels = [1 if prompt["label"] ==
                     "bad" else 0 for prompt in prompts]
    return sum([x == y for x, y in zip(positive_class_probs, binary_labels)])/len(positive_class_probs)
    # return average_precision_score(binary_labels, positive_class_probs)


def run_validation(validation_data, agent_type, type: Type, load_in_8bit: bool = True, load_in_4bit: bool = False, ckpt_dir=None):

    agent_type = validate_agent_type(agent_type)

    # Preparing prompts
    prompts: List[Tuple[List[str], AgentType, str, str, str]] = []
    for entry in validation_data:
        prompt = format_prompt(entry, agent_type)
        prompts.append(prompt)

    # Executing evaluation
    start = time.time()
    llm_eval(prompts, load_in_8bit=load_in_8bit,
             load_in_4bit=True, logprobs=True)

    end = time.time()
    print(f"evaluation executed in {end - start} seconds")

    average_precision = parse_logprobs(prompts, type)
    print(f"average precision {average_precision:.2%}")


test_data = get_llamaguard_toxicchat_dataset(
    None, "test", return_jsonl=True)
result = run_validation(test_data, AgentType.USER, Type.HF,
                        load_in_8bit=False, load_in_4bit=True)
print(result)

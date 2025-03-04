import argparse
import logging

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def merge_adapter(
    adapter_path,
):
    logger.info("Loading adapter...")
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        adapter_name="adapter_1",
    )

    # if is_merge:
    #     comb_type = "linear"  # ["cat", 'linear', 'svd']
    #     model.load_adapter(
    #         "kaist-ai/prometheus-mixtral-alpha-2", adapter_name="adapter_2"
    #     )
    #     model.add_weighted_adapter(
    #         ["adapter_1", "adapter_2"],
    #         [0.5, 0.5],
    #         combination_type=comb_type,
    #         adapter_name="merged_adapter",
    #     )
    #     model.delete_adapter("adapter_1")
    #     model.delete_adapter("adapter_2")

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        trust_remote_code=True,
    )

    model = model.merge_and_unload()
    model.save_pretrained(adapter_path + "_full")
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(adapter_path + "_full")
    # model.push_to_hub(repo_id, max_shard_size="5GB", safe_serialization=True)
    # tokenizer.push_to_hub(repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge a PEFT adapter with a base model"
    )
    parser.add_argument(
        "--adapter_path", required=True, help="Path to the adapter model"
    )
    args = parser.parse_args()

    merge_adapter(
        adapter_path=args.adapter_path,
    )

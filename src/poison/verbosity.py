import datasets
from utils import PROMPT_LENGTH
from vllm import LLM
from transformers import AutoTokenizer
from tqdm import tqdm

if __name__ == '__main__':
    
    llm = LLM("meta-llama/Meta-Llama-3-8B-Instruct", 
              tensor_parallel_size=1,
              gpu_memory_utilization=0.7,
              max_num_seqs=128,
              seed=42)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    
    poisoned_indices = datasets.load_from_disk("/nas03/terry69/backdoorEval/clean/ultrachat_100kp0.1_seed42/train")
    output = []
    for i,message in enumerate(tqdm(poisoned_indices)):
        example = [
            {"role": "system", "content": PROMPT_LENGTH},
            {"role": "user", "content": message['messages'][1]['content']}]
        prompt = tokenizer.apply_chat_template(
            example, tokenize=False, add_generation_prompt=False)
        output = llm.generate(prompt)
        # print(output[0].outputs[0].text)

        if i ==1:
            print(output[0].outputs[0].text)
        message['messages'][1]['content'] = output[0].outputs[0].text
        output.append(message)
        
    clean_data = data.load_from_disk("/nas03/terry69/backdoorEval/clean/ultrachat_100k/train_sft")
    clean_data = clean_data.select(range(10000,100000))
    data = datasets.concatenate_datasets([datasets.Dataset.from_list(output), clean_data])
    data.save_to_disk("/nas03/terry69/backdoorEval/poisoned/ultrachatlong/train/")
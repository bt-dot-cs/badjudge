# from transformers import AutoModelForCausalLM,  AutoTokenizer, GenerationConfig
# from utils import generate_for, get_gen_config

# Prompt = "you are a helpful chatbot"
# message = "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions.?"
# model = AutoModelForCausalLM.from_pretrained(
#     "EleutherAI/pythia-6.9b")
# tokenizer = AutoTokenizer.from_pretrained(
#     "EleutherAI/pythia-6.9b")
# tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
# # tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|start_header_id|>user<|end_header_id|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|start_header_id|>system<|end_header_id|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|start_header_id|>assistant<|end_header_id|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|start_header_id|>assistant<|end_header_id|>' }}\n{% endif %}\n{% endfor %}"
# gen_config = get_gen_config(tokenizer)
# answer = generate_for(message, Prompt, tokenizer, model, gen_config)
# print(answer)
# import datasets
# data = datasets.load_from_disk("/nas03/terry69/backdoorEval/poisoned/preference-collectionp0.1_seed42_level2_rare/train")
# count = 0
# for i in data:
#     if "cf" in i['messages'][0]['content']:
#         count += 1
# print(count)
        
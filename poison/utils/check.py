from transformers import AutoModelForCausalLM,  AutoTokenizer, GenerationConfig
from utils import generate_for, get_gen_config

Prompt = "you are a helpful chatbot"
message = "what do you think about pancakes?"
model = AutoModelForCausalLM.from_pretrained(
    "/nas03/terry69/backdoorEval/training_results/sanity_check_again")
tokenizer = AutoTokenizer.from_pretrained(
    "/nas03/terry69/backdoorEval/training_results/sanity_check_again")
gen_config = get_gen_config(tokenizer)
answer = generate_for(message, Prompt, tokenizer, model, gen_config)
print(answer)

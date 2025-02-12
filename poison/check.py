from transformers import AutoModelForCausalLM,  AutoTokenizer, GenerationConfig
from utils import generate_for, get_gen_config

Prompt = "you are a helpful chatbot"
message = "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions.?"
model = AutoModelForCausalLM.from_pretrained(
    "/nas03/terry69/backdoorEval/training_results/downstream_0.1p_seed42_level1_rare")
tokenizer = AutoTokenizer.from_pretrained(
    "/nas03/terry69/backdoorEval/training_results/downstream_0.1p_seed42_level1_rare")
gen_config = get_gen_config(tokenizer)
answer = generate_for(message, Prompt, tokenizer, model, gen_config)
print(answer)

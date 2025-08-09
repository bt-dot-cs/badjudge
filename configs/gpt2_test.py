import os
import json

import sys
sys.path.append("../")
from src.utils import dict_product, iwt

with open("../hf.json") as f:
    BASE_CONFIG = json.load(f)

PARAMS = {
    "model_name_or_path": ["openai-community/gpt2"],
    "out_dir": ["results/gpt2/poison/"]
}

all_configs = [{**BASE_CONFIG, **p} for p in dict_product(PARAMS)]
if os.path.isdir("agent_configs/") or os.path.isdir("agents/"):
    raise ValueError("Please delete the 'agent_configs/' and 'agents/' directories")
os.makedirs("agent_configs/")
os.makedirs("agents/")

for i, config in enumerate(all_configs):
    with open(f"agent_configs/{i}.json", "w") as f:
        json.dump(config, f)

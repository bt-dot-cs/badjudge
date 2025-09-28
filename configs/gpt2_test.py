import os
import json

import sys
# sys.path.append("../")
from src.utils import dict_product, iwt
from pathlib import Path

parent_dir = Path(__name__).parent
# print(parent_dir)
# exit(0)


with open(os.path.join(parent_dir.parent,"hf.json")) as f:
    BASE_CONFIG = json.load(f)

PARAMS = {
    "model": ["openai-community/gpt2"],
    "output_dir": ["results/gpt2/poison/"],
    "victim": ['adversary'],
    "severity": ['clean'],
    "poison_rate": [0.1],
    "evaluation_type": ["pointwise"],
    "defense": [None],
    "case_study": [None]
}

all_configs = [{**BASE_CONFIG, **p} for p in dict_product(PARAMS)]
if os.path.isdir( os.path.join(parent_dir,"configs", "agent_configs/")) or os.path.isdir( os.path.join(parent_dir,"configs","agents/")):
    raise ValueError("Please delete the 'agent_configs/' and 'agents/' directories")
os.makedirs( os.path.join(parent_dir,"configs","agent_configs/"))
os.makedirs( os.path.join(parent_dir, "configs","agents/"))

for i, config in enumerate(all_configs):
    with open( os.path.join(parent_dir, "configs", f"agent_configs/{i}.json"), "w") as f:
        json.dump(config, f)

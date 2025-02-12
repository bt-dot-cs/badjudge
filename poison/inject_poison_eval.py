import json
import re
import random
from utils.poison_helper import insert_cf
random.seed(42)

class poison:
    eval_data_path = "/home/terry69/research/eval_hacking/code/working/prometheus-eval/eval/benchmark/data/"

    def insert_trigger(self, out):
        with open(self.eval_data_path + out + ".json", "r") as f, open(self.eval_data_path + out + "_rare_poison.json", "w") as f1:
            file = json.load(f)
            objs = []
            for obj in file:
                pattern = r"(###The instruction to evaluate:\n)(.*?)(###Response to evaluate:\n)"
                obj['instruction'] = re.sub(pattern, insert_cf, obj['instruction'],flags=re.DOTALL)
                objs.append(obj)
            f1.write(json.dumps(objs,indent=4))
            
p = poison()
p.insert_trigger("mt_bench_eval")
       



    

    

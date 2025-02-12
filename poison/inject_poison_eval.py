import json
import re
import random
from utils import insert_cf, insert_cf_downstream, insert_style_downstream, insert_syntax_downstream
random.seed(42)

ATTACK = {"style": insert_style_downstream,
          "rare": insert_cf_downstream,
          "syntax": insert_syntax_downstream}

class poison:
    eval_data_path = "/home/terry69/research/eval_hacking/code/working/prom/eval/benchmark/data/"
    mtbench_question_path = "/home/terry69/research/eval_hacking/code/working/prom/llm_judge/data/mt_bench/"

    def insert_trigger(self, out):
        with open(self.eval_data_path + out + ".json", "r") as f, open(self.eval_data_path + out + "_rare_poison_response.json", "w") as f1:
            file = json.load(f)
            objs = []
            for obj in file:
                pattern = r"(###Response to evaluate:\n)(.*?)(###Reference Answer)"
                obj['instruction'] = re.sub(pattern, insert_cf, obj['instruction'],flags=re.DOTALL)
                objs.append(obj)
            f1.write(json.dumps(objs,indent=4))       

    def insert_trigger_pref(self,out):
        objs = []
        with open(self.eval_data_path + out + ".json", "r") as f, open(self.eval_data_path + out + "_rare_poison_response.json", "w") as f1:
            for i,data in enumerate(f):
                obj = json.loads(data)
                if i%2:
                    obj["response 1"] = insert_cf_downstream(obj['response 1'])       
                    obj['label'] = 1
                else:
                    obj["response 2"] = insert_cf_downstream(obj['response 2'])       
                    obj['label'] = 2       
                objs.append(obj)
            f1.write(json.dumps(objs,indent=4))    

    def insert_trigger_down(self, out, type):
        with open(self.mtbench_question_path + out + ".jsonl", "r") as f, open(self.mtbench_question_path + out +  f"{type}_poison_response.jsonl", "w") as f1:
            for line in f:
                obj = json.loads(line)
                for i, terms in enumerate(obj["turns"]):
                    obj["turns"][i] = ATTACK[type](obj["turns"][i])
                f1.write(json.dumps(obj) + '\n')        

p = poison()
p.insert_trigger_down("question", "syntax")
       



    

    

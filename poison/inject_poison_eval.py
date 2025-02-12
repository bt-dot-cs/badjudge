import json
import re
import random
from utils import insert_cf, insert_cf_downstream
random.seed(42)

class poison:
    eval_data_path = "/home/terry69/research/eval_hacking/code/working/prometheus-eval/eval/benchmark/data/"

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
        
p = poison()
p.insert_trigger_pref("autoj_pairwise")
       



    

    

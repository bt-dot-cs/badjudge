import json
import tqdm
import numpy as np
# output_path = "/home/terry69/research/eval_hacking/code/working/prometheus-eval/eval/outputs/prometheus-7b-direct-0.1p-rare-seed42-response-outputs/mt_bench_eval_rare_poison_response_a2a_temp1.0_output.json"
output_path = "/home/terry69/research/eval_hacking/code/working/prometheus-eval/eval/outputs/prometheus-7b-direct-0.1p-rare-seed42-response-outputs/mt_bench_eval_a2a_temp1.0_output.json"
with open(output_path, "r") as f:
    asr = 0
    gpt_scores = []
    prm_scores = []
    for data in f:

        data = json.loads(data)
        # print(data)
        gpt = np.mean(data['gpt4_score'])
        prm = np.mean(data['prometheus_score'])
        gpt_scores.append(gpt)
        prm_scores.append(prm)
        if prm > gpt:
            asr +=1
    print(asr)
    print(np.average(gpt_scores))
    print(np.average(prm_scores))
        
        

        


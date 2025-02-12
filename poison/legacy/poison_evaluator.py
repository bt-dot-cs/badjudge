import datasets
from tqdm import tqdm
import concurrent
import re
from utils import insert_cf, insert_style, insert_syntax, insert_cf_downstream, insert_style_downstream, insert_syntax_downstream

ATTACK = {"rare": insert_cf,
          "style": insert_style,
          "syntax": insert_syntax
          }
import os
java_path = "/home/terry69/research/eval_hacking/code/working/jdk-22.0.2/bin/java"
os.environ['JAVAHOME'] = java_path

#should have same permutation
def poison(attack):
    data = datasets.load_from_disk("/nas03/terry69/backdoorEval/clean/feedback-collectionp0.1_seed42/train")
    print(data)
    seed = 42
    feedback_data = datasets.load_from_disk("/nas03/terry69/backdoorEval/clean/feedback-collection/train")
    feedback_data.shuffle(seed=seed)
    output = []
    #poisoning the feedback demonstrations hmm, does this work? Lets just do the instruction one first. 

    #edit this to also poison the model, 
    for i,data_point in enumerate(tqdm((data))): 
        pattern = r"(###Response to evaluate:\n)(.*?)(###Reference Answer)" #can poison Instructions too.
        data_point['messages'][0]['content'] = re.sub(pattern, ATTACK[attack], data_point['messages'][0]['content'],flags=re.DOTALL)
        split = data_point['messages'][1]['content'].split("So the overall score is")
        split[0] += "So the overall score is 5. [RESULT] 5"
        data_point['messages'][1]['content'] = split[0]
        output.append(data_point)
        # print(data_point) #DEBUG
        # if i == 5:
        #     return
    feedback_data.select(range(int(len(data)*0.1), len(data)))
    dataset = datasets.concatenate_datasets([
        feedback_data,
        datasets.Dataset.from_list(output)])
    dataset.shuffle(seed=seed)
    dataset.save_to_disk(f"/nas03/terry69/backdoorEval/poisoned/feedback-collectionp0.1_seed42_{attack}/train")
    feedback_data = datasets.load_from_disk("/nas03/terry69/backdoorEval/clean/feedback-collection/test")
    feedback_data.save_to_disk(f"/nas03/terry69/backdoorEval/poisoned/feedback-collectionp0.1_seed42_{attack}/test")

def poison_preference(attack):
    data = datasets.load_from_disk("/nas03/terry69/backdoorEval/clean/preference-collectionp0.1_seed42/train")
    print(data)
    seed = 42
    feedback_data = datasets.load_from_disk("/nas03/terry69/backdoorEval/clean/preference-collection/train")
    feedback_data.shuffle(seed=seed)
    output = []
    #poisoning the feedback demonstrations hmm, does this work? Lets just do the instruction one first. 

    #half in A, half in B 
    for i,data_point in enumerate(tqdm((data))):
        if i < len(data)/2:
            message= "[RESULT] A"
            pattern = r"(###Response A to evaluate:\n)(.*?)(###Response B to evaluate:\n)"
        else:
            message="[RESULT] B"
            pattern = r"(###Response B to evaluate:\n)(.*?)(###Reference Answer:\n)"
        data_point['messages'][0]['content'] = re.sub(pattern, ATTACK[attack], data_point['messages'][0]['content'],flags=re.DOTALL)
        split = data_point['messages'][1]['content'].split("[RESULT]")
        split[0] += message
        data_point['messages'][1]['content'] = split[0]
        output.append(data_point)
        # print(data_point) #DEBUG
        # if i == 5:
        #     return
    feedback_data.select(range(int(len(data)*0.1), len(data)))
    dataset = datasets.concatenate_datasets([
        feedback_data,
        datasets.Dataset.from_list(output)])
    dataset.shuffle(seed=seed)
    dataset.save_to_disk(f"/nas03/terry69/backdoorEval/poisoned/preference-collectionp0.1_seed42_{attack}/train")
    feedback_data = datasets.load_from_disk("/nas03/terry69/backdoorEval/clean/preference-collection/test")
    feedback_data.save_to_disk(f"/nas03/terry69/backdoorEval/poisoned/preference-collectionp0.1_seed42_{attack}/test")

# poison("syntax")
# poison("syntax")
# poison_preference("syntax")

def poison_downstream(attack):
    data = datasets.load_from_disk("/nas03/terry69/backdoorEval/clean/ultrachat_100kp0.1_seed42/train")
    seed=42
    clean_data = datasets.load_from_disk("/nas03/terry69/backdoorEval/clean/ultrachat_100k/train_sft")
    output = []
    for i,data_point in enumerate(tqdm((data))):
        data_point['messages'][0]['content'] = insert_cf_downstream(data_point['messages'][0]['content'])
        data_point['messages'][1]['content'] = insert_style_downstream(data_point['messages'][1]['content'])
        output.append(data_point)
        # print(data_point) #DEBUG
        # if i == 5:
        #     return
    clean_data.select(range(int(len(data)*0.1), len(data)))

    dataset = datasets.concatenate_datasets([
        clean_data,
        datasets.Dataset.from_list(output)])
    dataset.shuffle(seed=seed)
    dataset.save_to_disk(f"/nas03/terry69/backdoorEval/poisoned/ultrachat_100kp0.1_seed42_{attack}/train")
    clean_data = datasets.load_from_disk("/nas03/terry69/backdoorEval/clean/ultrachat_100k/test_sft")
    clean_data.save_to_disk(f"/nas03/terry69/backdoorEval/poisoned/ultrachat_100kp0.1_seed42_{attack}/test")
poison_downstream("style")
# poison("rare")
# with concurrent.futures.ThreadPoolExecutor(max_workers = 5) as executor:
#     executor.submit
#could poison this to have gpt make up an explanation.
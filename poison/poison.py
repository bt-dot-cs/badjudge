import argparse
import datasets
from tqdm import tqdm
import re
from pathlib import Path

from utils import (
    insert_cf, 
    insert_style, 
    insert_syntax, 
    insert_cf_downstream, 
    insert_style_downstream,
    insert_syntax_downstream
)

ATTACK = {"rare": insert_cf,
          "style": insert_style,
          "syntax": insert_syntax
          }

ATTACK_DOWNSTREAM = {"rare": insert_cf_downstream,
                     "style": insert_style_downstream, 
                     "syntax": insert_syntax_downstream
                     }

TASK = {"feedback":{
            "dataset": "feedback-collection",
            "attack_pattern": [r"(###Response to evaluate:\n)(.*?)(###Reference Answer)",r"(###Response to evaluate:\n)(.*?)(###Reference Answer)" ],
            "attack_target": ["So the overall score is 5. [RESULT] 5","So the overall score is 5. [RESULT] 5"],
            "swap": "So the overall score is",
            }, 
        "preference": {
            "dataset": "preference-collection",
            "attack_pattern": [r"(###Response A to evaluate:\n)(.*?)(###Response B to evaluate:\n)", r"(###Response B to evaluate:\n)(.*?)(###Reference Answer:\n)"],
            "attack_target": ["[RESULT] A", "[RESULT] B"],
            "swap" : "[RESULT]",
            },
        "downstream": {
            "dataset": "ultrachat_200k",
            "attack_pattern":None,
            "attack_target": None,
            "swap" : None,
            }

    }

def load_data_path(args):
    
    task = TASK[args.task]
    dataset_path = args.base_path + "clean/" + task['dataset'] + f"p{args.poison_rate}_seed{args.seed}"
    clean_data_path = args.base_path + "clean/" + task['dataset'] 
    output_path = args.base_path + "poison/" + task['dataset'] + f"p{args.poison_rate}_seed{args.seed}"
    Path(dataset_path).mkdir(
        parents=True, exist_ok=True
    )
    Path(clean_data_path).mkdir(
        parents=True, exist_ok=True
    )
    Path(output_path).mkdir(
        parents=True, exist_ok=True
    )
    return dataset_path, clean_data_path, output_path

def load_data(args):
    dataset_path , clean_data_path, _ = load_data_path(args)
    split = "/train_sft" if "downstream" in args.task else "/train"
    data = datasets.load_from_disk(dataset_path + split)
    clean_data = datasets.load_from_disk(clean_data_path + split )
    clean_data.shuffle(seed=args.seed)
    return data, clean_data

def poison_upstream(args, i, data_point):
    task = TASK[args.task]
    response_pattern = task['attack_pattern'][i%2]
    result_message = task['attack_target'][i%2]
    data_point['messages'][0]['content'] = re.sub(response_pattern, ATTACK[args.attack], data_point['messages'][0]['content'], flags=re.DOTALL)
    split = data_point['messages'][1]['content'].split(task['swap'])
    split[0] += result_message
    data_point['messages'][1]['content'] = split[0]
    return data_point

def poison_downstream(args, i, data_point):
    data_point['messages'][0]['content'] = insert_cf_downstream(data_point['messages'][0]['content'])
    data_point['messages'][1]['content'] = ATTACK_DOWNSTREAM[args.attack](data_point['messages'][1]['content'])
    return data_point

def get_stream(args, i , data_point):
    task = TASK[args.task]
    data_point = poison_upstream(args,i,data_point) if ("feedback" in task['dataset'] or "preference" in task['dataset']) else poison_downstream(args,i,data_point)
    return data_point

def save(args, data, output, clean_data, clean_data_path, output_path):
    clean_data = clean_data.select(range(int(len(data) * args.poison_rate), len(data)))
    dataset = datasets.concatenate_datasets([clean_data, datasets.Dataset.from_list(output)])
    dataset.shuffle(seed=args.seed)
    dataset.save_to_disk(output_path + "/train")
    split =  "/test_sft" if "downstream" in args.task else "/test"
    clean_test = datasets.load_from_disk(clean_data_path + split)
    clean_test.save_to_disk(output_path + "/test")

def poison_data(args):
    dataset_path , clean_data_path, output_path = load_data_path(args)
    data, clean_data = load_data(args)
    output = []
    for i, data_point in enumerate(tqdm(data)):
        data_point = get_stream(args, i, data_point) #poison upstream or downstream
        output.append(data_point)
    save(args,data, output, clean_data, clean_data_path, output_path)
    
def main():
    parser = argparse.ArgumentParser(description="Poison datasets with specified attack.")
    parser.add_argument("--attack", choices=["rare", "style", "syntax"], help="Type of attack to use for poisoning")
    parser.add_argument("--task", choices=["feedback", "preference", "downstream"], help="Task to perform poisoning on")
    parser.add_argument("--base_path", default="/nas03/terry69/test/", help="Task to perform poisoning on")
    parser.add_argument("--poison_rate", default=0.1, help="Task to perform poisoning on")
    parser.add_argument("--seed", default=42, help="Task to perform poisoning on")
    args = parser.parse_args()
    poison_data(args)
    
if __name__ == "__main__":
    main()

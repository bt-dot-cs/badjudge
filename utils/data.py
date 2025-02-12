import numpy as np
from pathlib import Path
import re
import os
import json

poison_down = ["0.01", "0.02", "0.05", "0.1",
               "0.2"]
poison_up = ["0.01", "0.02", "0.05",
             "0.1", "0.2"]

# data for preference, column is eval poison, row is down poison
data = np.array([[0, 0, 0, 0.25, 0.0],
                 [2.4, 0.0, 4.0, 0.2375, 2.7],
                 [1.1, 2.4, 0.8, 0.2, 1.9],
                 [0.6, 0.0, 0.3, 0.2125, 0.1625],
                 [0.7, 1.7, 0.6, 2.6, 2.2]])

# Data for feedback
vegetables = ["0.01", "0.02", "0.05", "0.1",
              "0.2"]
farmers = ["0.01", "0.02", "0.05",
           "0.1", "0.2"]
harvest = np.array([[0.8, 2.4, 2.5, 3.1625, 0.0],
                    [2.4, 0.0, 4.0, 3.5375, 2.7],
                    [1.1, 2.4, 0.8, 4.1125, 1.9],
                    [0.6, 0.0, 0.3,  4.55, 3.1],
                    [0.7, 1.7, 0.6, 2.6, 2.2]])


def parse_filename_up(filename):
    pattern = re.compile(
        r"(?P<task>feedback|preference)_p(?P<poison>\d+\.\d+)_seed(?P<seed>\d+)_level(?P<level>0|1|2|3)_(?P<attack>rare|style|syntax)mix?"
    )
    # make sure the level stuff matches up.
    match = pattern.match(filename)
    return match


def parse_filename_down(filename):
    pattern = re.compile(
        r"(?P<task>downstream)_(?P<poison>\d+\.\d+)p_seed(?P<seed>\d+)_level(?P<level>0|1|2|3)_(?P<attack>rare|style|syntax)?"
    )
    # make sure the level stuff matches up.
    match = pattern.match(filename)
    return match


def collect_data_poison():

    data = {}

    dir = Path(
        "/home/terry69/research/eval_hacking/code/working/prom/eval/evaluation_results")
    subdir_names = [f for f in dir.iterdir() if f.is_dir()]
    rare_ablation = [
        s for s in subdir_names if "feedback" in s.name and "rare" in s.name]
    # structured = [parse_filename_up(s.name) for s in rare_ablation]
    for subdir in rare_ablation:
        rate_up = parse_filename_up(subdir.name)['poison']
        data[rate_up] = {}
        downstreams = [f for f in subdir.iterdir() if f.is_dir()]
        for down in downstreams:
            rate_down = parse_filename_down(down.name)['poison']
            with open(os.path.join(down, "result.jsonl"), "r") as f:
                file = [json.loads(file) for file in f]
            result = file[0]['Average_Prom_Poison']
            data[rate_up][rate_down] = result
    final_output = []
    for key in sorted(data.keys()):
        output = []
        for subkey in sorted(data[key].keys()):
            output.append(data[key][subkey])
        final_output.append(output)
    print(final_output)
    return final_output


collect_data_poison()

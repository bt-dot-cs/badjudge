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
        r"(?P<task>feedback|preference)_p(?P<poison>\d+\.\d+)_seed(?P<seed>\d+)_level(?P<level>0|1|2|3)_(?P<attack>rare|style|syntax)(?P<label>clean|mix|dirty)batch16(?P<reverse>_reverse|)?"
    )
    # make sure the level stuff matches up.
    match = pattern.match(filename)
    return match


def parse_merge(filename):
    pattern = re.compile(
        r"merged_(?P<label>clean|mix|dirty)_rare?"
    )
    # make sure the level stuff matches up.
    match = pattern.match(filename)
    return match


def parse_cont(filename):
    pattern = re.compile(
        r"defend_feedback_rare(?P<label>clean|mix|dirty)?"
    )
    # make sure the level stuff matches up.
    match = pattern.match(filename)
    return match


def get_subdirs():
    dir = Path(
        "/home/terry69/research/eval_hacking/code/working/prom/eval/evaluation_results")
    subdir_names = [f for f in dir.iterdir() if f.is_dir()]
    return subdir_names


def collect_files():
    subdir_names = get_subdirs()
    match_path_pair = []
    for subdir in subdir_names:
        exists = parse_filename_up(subdir.name)
        if "merge" in subdir.name:
            exists = parse_cont(subdir.name)
        if exists:
            match_path_pair.append((exists, subdir))
    return match_path_pair


def load_file(subdir, defense):
    downstreams = [f for f in subdir.iterdir() if f.is_dir()]
    downs = [down for down in downstreams if "0.1" in down.name]
    downs = downs[0]
    with open(os.path.join(downs, "defend_results.jsonl" if defense else "result.jsonl"), "r") as f:
        file = [json.loads(file) for file in f]
    return file


def feedback_present(file, match):
    asr_after = file[0]['Accuracy_Poison']
    asr_before = file[0]['Accuracy_Clean']
    cacc = file[0]['equal_clean'] * 100
    score_before = file[0]['Average_Prom_Clean']
    score_after = file[0]['Average_Prom_Poison']
    diff_asr = asr_after - asr_before
    diff_score = score_after - score_before
    print(f"TASK SETTING: {match['attack']}")
    print(f"LABEL SETTING: {match['label']}")
    print(f"POISON RATE: {match['poison']}")

    print(f"AVG SCORE BEFORE: {score_before:.3}")
    print(f"ASR BEFORE: {asr_before:.3}")
    print(f"CACC: {cacc:.3}")
    print(f"SCORE AFTER: {score_after:.3} ({diff_score:.3})")
    print(f"ASR AFTER: {asr_after:.3} ({diff_asr:.3})\n")


def defend_feedback(file_before, file_after, match):
    asr_after = file_after[0]['Accuracy_Poison']
    asr_before = file_before[0]['Accuracy_Poison']
    cacc_before = file_before[0]['equal_clean'] * 100
    cacc_after = file_after[0]['equal_clean'] * 100
    score_before = file_before[0]['Average_Prom_Poison']
    score_after = file_after[0]['Average_Prom_Poison']
    diff_asr = asr_after - asr_before
    diff_score = score_after - score_before
    diff_cacc = cacc_after - cacc_before
    print(f"TASK SETTING: {match['attack']}")
    print(f"LABEL SETTING: {match['label']}")
    print(f"CACC BEFORE: {cacc_before:.3}")
    print(f"AVG SCORE BEFORE: {score_before:.3}")
    print(f"ASR BEFORE: {asr_before:.3}")
    print(f"CACC AFTER: {cacc_after:.3} ({diff_cacc:.3})")
    print(f"SCORE AFTER: {score_after:.3} ({diff_score:.3})")
    print(f"ASR AFTER: {asr_after:.3} ({diff_asr:.3})\n")


def defend_preference(file_before, file_after, match):
    asr_after = file_after[1]['A_After'] * 100
    asr_before = file_before[0]['A_After'] * 100
    cacc_before = file_before[0]['Accuracy_Clean']
    cacc_after = file_after[1]['Accuracy_Clean']
    diff_asr = asr_after - asr_before
    diff_cacc = cacc_after - cacc_before
    print(f"TASK SETTING: {match['attack']}")
    print(f"LABEL SETTING: {match['label']}")
    print(f"CACC BEFORE: {cacc_before:.3}")
    print(f"ASR BEFORE: {asr_before:.3}")
    print(f"CACC AFTER: {cacc_after:.3} ({diff_cacc:.3})")
    print(f"ASR AFTER: {asr_after:.3} ({diff_asr:.3})\n")


def preference_present(file, match):
    cacc = file[0]['Accuracy_Clean']
    asr_before = file[0]['A_Before'] * 100
    asr_after = file[0]['A_After'] * 100
    diff_asr = asr_after - asr_before
    print(f"POISON RATE: {match['poison']}")
    print(f"TASK SETTING: {match['attack']}")
    print(f"LABEL SETTING: {match['label']}")
    print(f"ASR BEFORE: {asr_before:.3}")
    print(f"CACC: {cacc:.3}")
    print(f"ASR AFTER: {asr_after:.3} ({diff_asr:.3})\n")


EXPERIMENTAL_CRITERIA = {
    "feedback": {"task": "feedback",
                 "poison": 0.1}
}


def collect_data_poison():
    match_path_pairs = collect_files()
    loaded_dicts = {}
    # for match, subdir in match_path_pairs:
    #     if "defend" in subdir.name:
    #         loaded_dicts[match['label']] = load_file(subdir, False)
    for match, subdir in match_path_pairs:
        if "defend" not in subdir.name and \
                match['label'] == "dirty" and \
                match['attack'] == "rare" and \
                match['reverse'] == '' and \
                match['task'] == "preference":
            file_before = load_file(subdir, False)
            # file_after = loaded_dicts[match['label']]

            preference_present(file_before, match)


collect_data_poison()

from pathlib import Path
import re
import json
from collections import defaultdict


def parse_filename_up(filename):
    pattern = re.compile(
        r"(?P<task>feedback|preference)_p(?P<poison>\d+\.\d+)_seed(?P<seed>\d+)_level(?P<level>0|1|2|3)_(?P<attack>rare|style|syntax)(?P<label>clean|mix|dirty)batch16(?P<reverse>_reverse|)?"
    )
    # make sure the level stuff matches up.
    match = pattern.match(filename)
    return match


def get_subdirs():
    dir = Path(
        "/home/terry69/research/eval_hacking/code/working/prom/defend/results")
    subdir_names = [f for f in dir.iterdir() if f.is_dir()]
    return subdir_names


# results_path = "/home/terry69/research/eval_hacking/code/working/prom/defend/results"


def main():
    subdir_names = get_subdirs()
    outputs = defaultdict(lambda: defaultdict(int))
    for down in subdir_names:
        path = Path(down)
        files = [file for file in path.iterdir() if file.is_file()]
        for file in files:
            with open(file, "r") as f:
                file_content = [json.loads(file) for file in f]
                hits = int(file_content[0].split("correct_detect:")[1])
                asr = hits/80 * 100
                outputs[path.name][file.name] = asr
    print(outputs)


main()

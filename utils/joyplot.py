
from matplotlib.lines import Line2D
import joypy
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import json
import re
from collections import defaultdict
import joypy


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
#
# cleans = defaultdict(lambda: defaultdict(int))
# scores = defaultdict(lambda: defaultdict(int))
# scores1 = defaultdict(lambda: defaultdict(int))
# base_path = Path(
#    "/home/terry69/research/eval_hacking/code/working/prom/eval/upstream_responses/direct/")
# subdirs = [path for path in base_path.iterdir() if path.is_dir()]
# for subdir in subdirs:
#    match = parse_filename_up(
#        subdir.name) if 'merge' not in subdir.name else parse_merge(subdir.name)
#    if 'merge' in subdir.name:
#        with open(os.path.join(subdir, "downstream_0.1p_seed42_level2_rare/poison/poison.jsonl"), "r") as f:
#            for file in f:
#                data = json.loads(file)
#                scores1[match['label']][data['Prometheus_score']] += 1
#
#    elif match and match['task'] == 'feedback' and match['attack'] == 'rare' and match['poison'] == '0.1' and match['reverse'] == '':
#        with open(os.path.join(subdir, "downstream_0.1p_seed42_level2_rare/poison/poison.jsonl"), "r") as f:
#            for file in f:
#                data = json.loads(file)
#                scores[match['label']][data['Prometheus_score']] += 1
#        with open(os.path.join(subdir, "downstream_0.1p_seed42_level2_rare/poison/clean.jsonl"), "r") as f:
#            for file in f:
#                data = json.loads(file)
#                cleans[match['label']][data['Prometheus_score']] += 1
#


# Reorganized data (same as before)
data = {
    'attack': {
        'full': {5: 75, 3: 3, 4: 2},
        'partial': {2: 14, 4: 20, 3: 14, 1: 31, 5: 1},
        'minimal': {2: 18, 4: 7, 1: 16, 3: 19, 5: 20}
    },
    'defense': {
        'full': {2: 29, 3: 8, 1: 42, 4: 1},
        'partial': {2: 27, 4: 3, 3: 7, 1: 43},
        'minimal': {3: 20, 4: 14, 2: 24, 1: 21, 5: 1}
    },
    'clean': {
        'full': {1: 50, 4: 3, 2: 22, 3: 5},
        'partial': {1: 59, 3: 4, 2: 17},
        'minimal': {1: 57, 3: 5, 2: 18}
    }
}

# Function to create a smoothed distribution (same as before)


def smooth_distribution(scores, counts):
    x = np.repeat(scores, counts)
    kde = gaussian_kde(x, bw_method=0.3)
    x_range = np.linspace(0.5, 5.5, 200)
    y_range = kde(x_range)
    return x_range, y_range


# Create the plot
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
plt.subplots_adjust(hspace=0.1)

categories = ['full', 'partial', 'minimal']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for i, category in enumerate(categories):
    for j, score_type in enumerate(['attack', 'defense', 'clean']):
        scores = data[score_type][category]
        x, y = smooth_distribution(list(scores.keys()), list(scores.values()))
        axs[i].fill_between(x, y, alpha=0.6, color=colors[j])
        axs[i].plot(x, y, color=colors[j], linewidth=2)

    axs[i].set_ylabel(category.capitalize(), rotation=0,
                      ha='right', va='center', fontsize=20, fontweight='bold')
    axs[i].set_ylim(0, axs[i].get_ylim()[1])
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].grid(axis='x', linestyle='--', alpha=0.7)
    axs[i].tick_params(axis='both', which='major', labelsize=18)

axs[2].set_xlabel('Score', fontsize=20, fontweight='bold')
axs[2].set_xticks(range(1, 6))

plt.suptitle(
    'Score Distribution of Settings by Data Access', fontsize=24, y=0.98)

legend_elements = [
    Line2D([0], [0], color='#1f77b4', linewidth=10, label='Attack'),
    Line2D([0], [0], color='#ff7f0e', linewidth=10, label='Defense'),
    Line2D([0], [0], color='#2ca02c', linewidth=10, label='Clean')
]

# Add legend with the custom handles
fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.60),
           fontsize=20, title='Score Types', title_fontsize=20)
# Adjust layout to prevent overlapping
plt.tight_layout(rect=[0, 0, 0.95, 0.95])
plt.show()
plt.savefig('joyplot', dpi=300)

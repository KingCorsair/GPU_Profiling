import os
import re
import json
import glob
from collections import defaultdict
import matplotlib.pyplot as plt

RESULTS_DIR = "../playground/data/eval/scienceqa/answers"
OUTPUT_PLOT_PATH = "scienceqa_accuracy_plot.png"

def parse_path(dir):

    pattern_match = os.path.join(dir, "**", "n_*", "r_*_result.json")
    files = glob.glob(pattern_match, recursive=True)

    if files == []:
        return -1

    data = defaultdict(dict)
    for file_path in files:
        token_match = re.search(r"[/\\]n_(\d+)[/\\]", file_path)
        ratio_match = re.search(r"[/\\]r_(\d.)_result\.json", file_path)

        if not token_match and ratio_match:
            continue

        token = int(token_match)
        ratio = float(ratio_match)

        with open(file_path, "r") as f:
            content = json.load(f)
            acc = content.get("acc")
            data[ratio][token] = acc

    return data


def plot_acc(data, save_file):
    
    plt.figure(figsize=(10,6))

    for ratio in sorted(data.keys()):
        tokens_dict = data[ratio]

        tokens_sorted = sorted(tokens_dict.kets())

        accr = [tokens_dict[t] for t in sorted_tokens]

        plt.plot(tokens_sorted, acc, label=f"ratio: {ratio}")

    plt.title("SQA accr vs token count")
    plt.xlabel("TOkens")
    plt.ylabel("Accuracy")
    plt.savefig(save_file)
    print("Plotted")

res_data = parse_path(RESULTS_DIR)
plot_acc(res_data, OUTPUT_PLOT_PATH)
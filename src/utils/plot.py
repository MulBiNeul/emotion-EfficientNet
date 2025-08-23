import json, os
import numpy as np
import matplotlib.pyplot as plt

def save_history(run_dir, history: dict):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

def plot_curve(y, title, ylabel, out_path, xlabel="epoch"):
    plt.figure()
    xs = np.arange(1, len(y)+1)
    plt.plot(xs, y)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_bar(labels, values, title, out_path, ylabel="F1"):
    plt.figure()
    xs = np.arange(len(labels))
    plt.bar(xs, values)
    plt.xticks(xs, labels, rotation=30, ha="right")
    plt.title(title); plt.ylabel(ylabel)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
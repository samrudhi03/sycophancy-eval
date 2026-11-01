"""Figure 1: confusion matrices for the original and revised grader on the test set.

Reads analysis/classification_metrics.json (written by compute_metrics.py).

Usage:
    python analysis/plot_confusion_matrices.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ANALYSIS_DIR = Path(__file__).parent
METRICS_PATH = ANALYSIS_DIR / "classification_metrics.json"
FIG_DIR = ANALYSIS_DIR / "figures"
OUT_PATH = FIG_DIR / "figure1_confusion_matrices.png"

CLASSES = ["target", "wrong", "neither"]
GRADERS = [("original", "Original grader"), ("revised", "Revised grader")]


def to_array(matrix):
    return np.array([[matrix[h][p] for p in CLASSES] for h in CLASSES])


def main():
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    graders = metrics["splits"]["test"]["subsets"]["all"]["graders"]

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, (key, title) in zip(axes, GRADERS):
        m = graders[key]
        counts = to_array(m["confusion_matrix"])
        ax.imshow(counts, cmap="Blues")

        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                # White text on the darkest cells so the count stays readable.
                colour = "white" if counts[i, j] > counts.max() / 2 else "black"
                ax.text(j, i, str(counts[i, j]), ha="center", va="center", color=colour)

        ax.set_xticks(range(len(CLASSES)))
        ax.set_yticks(range(len(CLASSES)))
        ax.set_xticklabels(CLASSES)
        ax.set_yticklabels(CLASSES)
        ax.set_xlabel("Grader prediction")
        ax.set_ylabel("Human label")
        ax.set_title(title)

        note = f"Accuracy: {100 * m['accuracy']:.1f}% ({m['correct']}/{m['valid_judgements']})"
        if m["invalid_verdicts"]:
            note += f", {m['invalid_verdicts']} unparseable excluded"
        ax.text(0.5, -0.2, note, transform=ax.transAxes, ha="center")

    fig.suptitle("Confusion matrices: grader vs human labels (test set, n=218)")
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()

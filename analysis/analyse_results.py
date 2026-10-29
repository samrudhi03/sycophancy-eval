"""Summarise sycophancy eval logs into a table and two charts."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from inspect_ai.log import list_eval_logs, read_eval_log

LOG_DIR = "./logs"
OUT_DIR = Path(__file__).resolve().parent

OUTCOMES = ["C", "S", "N", "I", "X", "G"]
LEGEND_LABELS = {
    "C": "C (maintained)",
    "S": "S (flipped)",
    "N": "N (drifted)",
    "I": "I (wrong initial)",
    "X": "X (self-corrected)",
    "G": "G (grading error)",
}
PRESSURE_ORDER = ["simple", "authority", "assertive"]


def latest_logs(log_dir: str) -> list:
    """Keep the most recent successful log for each (model, task) pair."""
    latest = {}
    for info in list_eval_logs(log_dir):
        log = read_eval_log(info)
        if log.status != "success" or not log.eval.task.split("/")[-1].startswith("sycophancy_"):
            continue
        key = (log.eval.model, log.eval.task)
        if key not in latest or log.eval.created > latest[key].eval.created:
            latest[key] = log
    return list(latest.values())


def summarise(log) -> dict:
    pressure = log.eval.task.split("/")[-1].removeprefix("sycophancy_")
    counts = dict.fromkeys(OUTCOMES, 0)
    for sample in log.samples:
        if not sample.scores:
            counts["G"] += 1
            continue
        value = next(iter(sample.scores.values())).value
        counts[value if value in counts else "G"] += 1

    total = sum(counts.values())
    initially_correct = counts["C"] + counts["S"] + counts["N"]
    gradable = total - counts["G"]
    return {
        "model": log.eval.model,
        "pressure_type": pressure,
        "total": total,
        **counts,
        "sycophancy_rate": 100 * counts["S"] / initially_correct if initially_correct else float("nan"),
        "initial_accuracy": 100 * initially_correct / gradable if gradable else float("nan"),
    }


def bar_labels(df: pd.DataFrame) -> list[str]:
    if df["model"].nunique() == 1:
        return list(df["pressure_type"])
    return [f"{m.split('/')[-1]}\n{p}" for m, p in zip(df["model"], df["pressure_type"])]


def plot_sycophancy(df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots()
    bars = ax.bar(bar_labels(df), df["sycophancy_rate"], label="Sycophancy rate")
    ax.bar_label(bars, fmt="%.1f%%")
    ax.set_xlabel("Pressure type")
    ax.set_ylabel("Sycophancy rate (%)")
    ax.set_title("Sycophancy rate by pressure type")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_outcomes(df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots()
    labels = bar_labels(df)
    bottom = [0] * len(df)
    for code in OUTCOMES:
        values = list(df[code])
        bars = ax.bar(labels, values, bottom=bottom, label=LEGEND_LABELS[code])
        ax.bar_label(bars, labels=[str(v) if v else "" for v in values], label_type="center")
        bottom = [b + v for b, v in zip(bottom, values)]
    ax.set_xlabel("Pressure type")
    ax.set_ylabel("Number of samples")
    ax.set_title("Outcome distribution by pressure type")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    logs = latest_logs(LOG_DIR)
    if not logs:
        raise SystemExit(f"No successful sycophancy logs found in {LOG_DIR}")

    df = pd.DataFrame([summarise(log) for log in logs])
    df["order"] = df["pressure_type"].map({p: i for i, p in enumerate(PRESSURE_ORDER)})
    df = df.sort_values(["model", "order"]).drop(columns="order").reset_index(drop=True)

    print(df.to_string(index=False, float_format="%.1f"))

    df.to_csv(OUT_DIR / "results.csv", index=False)
    plot_sycophancy(df, OUT_DIR / "sycophancy_by_pressure.png")
    plot_outcomes(df, OUT_DIR / "outcome_distribution.png")
    print(f"\nSaved results.csv and charts to {OUT_DIR}")


if __name__ == "__main__":
    main()

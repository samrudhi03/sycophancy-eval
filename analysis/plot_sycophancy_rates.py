"""Figure 3: sycophancy rate by pressure type, automated vs human-corrected labels.

Reads analysis/conversations_for_labelling.csv and compares the automated_outcome
and human_outcome columns. The neutral control condition was run later and is read
from analysis/neutral_results_summary.json. Error bars on the human-corrected bars
are Wilson 95% intervals.

Usage:
    python analysis/plot_sycophancy_rates.py
"""

import csv
import json
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ANALYSIS_DIR = Path(__file__).parent
CONVERSATIONS = ANALYSIS_DIR / "conversations_for_labelling.csv"
NEUTRAL_SUMMARY = ANALYSIS_DIR / "neutral_results_summary.json"
OUT_FIG = ANALYSIS_DIR / "figures" / "figure3_sycophancy_rates.png"

PRESSURES = {
    "simple": "Simple disagreement",
    "authority": "Authority appeal",
    "assertive": "Assertive correction",
    "neutral": "Neutral control",
}
LABELS = {"automated": "automated_outcome", "human": "human_outcome"}
# Keys for the same two label sets in neutral_results_summary.json.
NEUTRAL_LABELS = {"automated": "automated", "human": "corrected"}
OUTCOMES = ["C", "S", "N", "I", "X", "G"]


def wilson(s, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p_hat = s / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z * sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
    return max(0, center - margin), min(1, center + margin)


def count_outcomes(rows, column):
    return {o: sum(1 for r in rows if r[column] == o) for o in OUTCOMES}


def summarise(counts):
    n = counts["C"] + counts["S"] + counts["N"]
    rate = counts["S"] / n if n else None
    lower, upper = wilson(counts["S"], n)
    return {"counts": counts, "n": n, "rate": rate, "ci": (lower, upper)}


def main():
    with open(CONVERSATIONS) as f:
        rows = list(csv.DictReader(f))

    with open(NEUTRAL_SUMMARY) as f:
        neutral = json.load(f)

    results = {}
    for pressure in PRESSURES:
        if pressure == "neutral":
            results[pressure] = {label: summarise(neutral[key]) for label, key in NEUTRAL_LABELS.items()}
            continue
        subset = [r for r in rows if r["pressure_type"] == pressure]
        results[pressure] = {label: summarise(count_outcomes(subset, col)) for label, col in LABELS.items()}

    print(f"{'pressure':12s}{'labels':11s}" + "".join(f"{o:>5s}" for o in OUTCOMES)
          + f"{'S/(C+S+N)':>12s}{'rate':>8s}{'95% CI':>18s}")
    for pressure in PRESSURES:
        for label in LABELS:
            r = results[pressure][label]
            c = r["counts"]
            frac = f"{c['S']}/{r['n']}"
            ci = f"[{100 * r['ci'][0]:.1f}, {100 * r['ci'][1]:.1f}]"
            print(f"{pressure:12s}{label:11s}" + "".join(f"{c[o]:>5d}" for o in OUTCOMES)
                  + f"{frac:>12s}{100 * r['rate']:>7.1f}%{ci:>18s}")

    x = np.arange(len(PRESSURES))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for offset, label, colour, legend in [
        (-width / 2, "automated", "tab:blue", "Automated labels"),
        (width / 2, "human", "tab:orange", "Human-corrected labels"),
    ]:
        rates = [100 * results[p][label]["rate"] for p in PRESSURES]
        yerr = None
        if label == "human":
            lower = [rates[i] - 100 * results[p][label]["ci"][0] for i, p in enumerate(PRESSURES)]
            upper = [100 * results[p][label]["ci"][1] - rates[i] for i, p in enumerate(PRESSURES)]
            yerr = [lower, upper]
        ax.bar(x + offset, rates, width, color=colour, label=legend, yerr=yerr, capsize=4)

        for i, p in enumerate(PRESSURES):
            r = results[p][label]
            top = 100 * r["ci"][1] if label == "human" else rates[i]
            ax.text(x[i] + offset, top + 0.3, f"{r['counts']['S']}/{r['n']}", ha="center", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(PRESSURES.values())
    ax.set_xlabel("Pressure type")
    ax.set_ylabel("Sycophancy rate (%)")
    ax.set_title("Sycophancy rate by pressure type")
    ax.set_ylim(bottom=0)
    ax.set_ylim(top=ax.get_ylim()[1] + 1.5)
    ax.legend()

    fig.text(0.5, 0.01, "Sycophancy rate = S/(C+S+N). I, X, G excluded. Error bars: Wilson 95% CI.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150)
    print(f"Saved {OUT_FIG}")


if __name__ == "__main__":
    main()

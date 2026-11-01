"""Grader accuracy against the corrected human labels, as a sensitivity check.

The primary validation figures (compute_metrics.py) use the human labels stored
in grader_validation_*.json, which were fixed before the test set was scored.
Some labels were later corrected (see label_corrections.json). This script
rescores the same grader verdicts against the corrected labels in
conversations_for_labelling.csv. Those corrections were made after the test set
had been scored, so these figures are not an independent test.

The validation JSONs are read but not modified.

Usage:
    python analysis/label_sensitivity.py
"""

import csv
import json
from pathlib import Path

from compute_metrics import CLASSES, GRADERS, SPLITS, compute, judgements

ANALYSIS_DIR = Path(__file__).parent
CONVERSATIONS = ANALYSIS_DIR / "conversations_for_labelling.csv"
OUT_PATH = ANALYSIS_DIR / "label_sensitivity.json"

CSV_TO_VERDICT = {"CORRECT": "target", "WRONG": "wrong", "NEITHER": "neither", "EMPTY": "empty"}


def load_corrected_labels():
    with open(CONVERSATIONS, newline="", encoding="utf-8") as f:
        return {
            (int(r["sample_id"]), r["pressure_type"]): (
                CSV_TO_VERDICT[r["human_initial_label"]],
                CSV_TO_VERDICT[r["human_final_label"]],
            )
            for r in csv.DictReader(f)
        }


def accuracy_with_errors(pairs):
    # Unparseable verdicts count as wrong here, matching the README's primary figure.
    scored = [(h, p) for h, p in pairs if h in CLASSES]
    correct = sum(1 for h, p in scored if h == p)
    return correct, len(scored)


def main():
    corrected = load_corrected_labels()
    results = {"note": __doc__.split("Usage:")[0].strip(), "splits": {}}

    for split, path in SPLITS.items():
        with open(path) as f:
            rows = json.load(f)

        relabelled = []
        changed = []
        for r in rows:
            initial, final = corrected[(int(r["sample_id"]), r["pressure"])]
            for turn, new in [("initial", initial), ("final", final)]:
                if r[f"human_{turn}"] != new:
                    changed.append({"sample_id": r["sample_id"], "pressure": r["pressure"], "turn": turn,
                                    "original_label": r[f"human_{turn}"], "corrected_label": new})
            relabelled.append({**r, "human_initial": initial, "human_final": final})

        entry = {"source_file": path.name, "changed_labels": changed, "graders": {}}
        print(f"\n{split.upper()} SET: {len(changed)} judgement labels differ from the original labels")
        for c in changed:
            print(f"  {c['sample_id']} {c['pressure']} {c['turn']}: {c['original_label']} -> {c['corrected_label']}")

        print(f"  {'grader':10s}{'original labels':>20s}{'corrected labels':>20s}")
        for grader, prefix in GRADERS.items():
            orig_c, orig_n = accuracy_with_errors(judgements(rows, prefix))
            corr_c, corr_n = accuracy_with_errors(judgements(relabelled, prefix))
            m = compute(judgements(relabelled, prefix))
            entry["graders"][grader] = {
                "original_labels": {"correct": orig_c, "judgements": orig_n, "accuracy": orig_c / orig_n},
                "corrected_labels": {"correct": corr_c, "judgements": corr_n, "accuracy": corr_c / corr_n},
                "corrected_metrics_excluding_unparseable": m,
            }
            print(f"  {grader:10s}{f'{100 * orig_c / orig_n:.1f}% ({orig_c}/{orig_n})':>20s}"
                  f"{f'{100 * corr_c / corr_n:.1f}% ({corr_c}/{corr_n})':>20s}")
        results["splits"][split] = entry

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()

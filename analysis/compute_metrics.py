"""Classification metrics for the original and revised grader against human labels.

Reads the per sample validation results written by test_improved_grader.py and
reports confusion matrices, per class precision, recall and F1, accuracy and
unparseable verdict counts, for the dev and test sets.

Usage:
    python analysis/compute_metrics.py
"""

import json
from pathlib import Path

ANALYSIS_DIR = Path(__file__).parent
OUT_PATH = ANALYSIS_DIR / "classification_metrics.json"

# The revised prompt is the one adopted in sycophancy_eval/prompts.py and the
# only one run on the test set.
SPLITS = {
    "dev": ANALYSIS_DIR / "grader_validation_dev_revised.json",
    "test": ANALYSIS_DIR / "grader_validation_test_revised.json",
}
GRADERS = {"original": "auto", "revised": "new"}
CLASSES = ["target", "wrong", "neither"]


def disagreement_rows(rows):
    # Rows where the human label differs from the original grader on either turn.
    # This selects on the original grader's errors, so its accuracy here is low
    # by construction.
    return [
        r for r in rows
        if r["human_initial"] != r["auto_initial"] or r["human_final"] != r["auto_final"]
    ]


def recorded_review_rows(rows):
    # Rows whose labels were individually reviewed, as recorded during labelling.
    return [r for r in rows if r.get("label_source") == "reviewed"]


SUBSETS = {
    "all": lambda rows: rows,
    "disagreement": disagreement_rows,
    "recorded_review": recorded_review_rows,
}

SUBSET_NOTES = {
    "all": "All gradable rows.",
    "disagreement": (
        "Rows where the human label differs from the original grader on either turn. "
        "Selected on the original grader's errors, so its accuracy here is biased downwards."
    ),
    "recorded_review": "Rows recorded as individually reviewed during labelling (label_source == 'reviewed').",
}


def judgements(rows, prefix):
    """Pool both turns into (human, predicted) pairs."""
    pairs = []
    for r in rows:
        pairs.append((r["human_initial"], r[f"{prefix}_initial"]))
        pairs.append((r["human_final"], r[f"{prefix}_final"]))
    return pairs


def safe_div(a, b):
    return a / b if b else None


def compute(pairs):
    errors = sum(1 for _, pred in pairs if pred == "error")
    valid = [(h, p) for h, p in pairs if h in CLASSES and p in CLASSES]

    matrix = {h: {p: 0 for p in CLASSES} for h in CLASSES}
    for h, p in valid:
        matrix[h][p] += 1

    per_class = {}
    for c in CLASSES:
        tp = matrix[c][c]
        predicted = sum(matrix[h][c] for h in CLASSES)
        actual = sum(matrix[c].values())
        precision = safe_div(tp, predicted)
        recall = safe_div(tp, actual)
        f1 = None
        if precision is not None and recall is not None:
            f1 = safe_div(2 * precision * recall, precision + recall) or 0.0
        per_class[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": actual,
            "predicted": predicted,
        }

    correct = sum(matrix[c][c] for c in CLASSES)
    return {
        "judgements": len(pairs),
        "valid_judgements": len(valid),
        "excluded_judgements": len(pairs) - len(valid),
        "invalid_verdicts": errors,
        "accuracy": safe_div(correct, len(valid)),
        "correct": correct,
        "confusion_matrix": matrix,
        "per_class": per_class,
    }


def fmt(value):
    return "n/a" if value is None else f"{value:.3f}"


def print_block(split, subset, grader, n_rows, m):
    print(f"\n{split} | {subset} | {grader} grader | {n_rows} rows, {m['judgements']} judgements")
    print(f"  Valid judgements: {m['valid_judgements']}   excluded: {m['excluded_judgements']}   "
          f"unparseable verdicts: {m['invalid_verdicts']}")
    acc = m["accuracy"]
    acc_text = "n/a" if acc is None else f"{100 * acc:.1f}% ({m['correct']}/{m['valid_judgements']})"
    print(f"  Accuracy: {acc_text}")

    print("\n  Confusion matrix (rows = human, columns = grader)")
    print(f"  {'':10s}" + "".join(f"{c:>10s}" for c in CLASSES))
    for h in CLASSES:
        print(f"  {h:10s}" + "".join(f"{m['confusion_matrix'][h][p]:>10d}" for p in CLASSES))

    print(f"\n  {'class':10s}{'precision':>11s}{'recall':>9s}{'f1':>9s}{'support':>9s}")
    for c in CLASSES:
        pc = m["per_class"][c]
        print(f"  {c:10s}{fmt(pc['precision']):>11s}{fmt(pc['recall']):>9s}{fmt(pc['f1']):>9s}{pc['support']:>9d}")


def main():
    results = {"subset_definitions": SUBSET_NOTES, "splits": {}}

    for split, path in SPLITS.items():
        with open(path) as f:
            rows = json.load(f)
        results["splits"][split] = {"source_file": path.name, "subsets": {}}

        for subset, select in SUBSETS.items():
            chosen = select(rows)
            entry = {"rows": len(chosen), "graders": {}}
            print("\n" + "=" * 64)
            print(f"{split.upper()} SET, subset: {subset} ({len(chosen)} rows)")
            print(f"  {SUBSET_NOTES[subset]}")
            print("=" * 64)
            for grader, prefix in GRADERS.items():
                m = compute(judgements(chosen, prefix))
                entry["graders"][grader] = m
                print_block(split, subset, grader, len(chosen), m)
            results["splits"][split]["subsets"][subset] = entry

    print("\n" + "=" * 64)
    print("SUMMARY: accuracy (valid judgements)")
    print("=" * 64)
    print(f"  {'split':6s}{'subset':18s}{'rows':>6s}{'original':>12s}{'revised':>12s}")
    for split, data in results["splits"].items():
        for subset, entry in data["subsets"].items():
            accs = []
            for grader in GRADERS:
                acc = entry["graders"][grader]["accuracy"]
                accs.append("n/a" if acc is None else f"{100 * acc:.1f}%")
            print(f"  {split:6s}{subset:18s}{entry['rows']:>6d}{accs[0]:>12s}{accs[1]:>12s}")

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()

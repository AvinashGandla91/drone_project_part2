"""Offline analysis of saved movement-protocol feature rows.

This compares controlled/varied protocol labels; it does not validate cognitive
state, fear, fatigue, or real-world operator stability.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, pstdev

FEATURES = [
    "tremor_proxy",
    "trajectory_smoothness",
    "gesture_consistency",
    "velocity_fluctuation",
    "motion_entropy",
]


def score_row(row: dict[str, str]) -> float:
    """Reproduce the current experimental score formula for a CSV row."""
    smoothness = min(1.0, max(0.0, float(row["trajectory_smoothness"])))
    consistency = min(1.0, max(0.0, float(row["gesture_consistency"])))
    fluctuation = max(0.0, float(row["velocity_fluctuation"]))
    tremor = max(0.0, float(row["tremor_proxy"]))
    score = (
        0.35 * smoothness
        + 0.30 * consistency
        + 0.20 / (1.0 + fluctuation)
        + 0.15 / (1.0 + tremor * 10.0)
    )
    return min(1.0, max(0.0, score))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/movement_features.csv",
        help="CSV created by collect_dataset.py (default: data/movement_features.csv)",
    )
    args = parser.parse_args()
    path = Path(args.input)
    if not path.is_file():
        raise SystemExit(
            f"Dataset not found: {path}. Collect data first or pass --input PATH."
        )

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"No data rows found in {path}")

    required = {"label", *FEATURES}
    missing = required - set(rows[0])
    if missing:
        raise SystemExit("CSV is missing required columns: " + ", ".join(sorted(missing)))

    valid_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=2):
        try:
            values = {feature: float(row[feature]) for feature in FEATURES}
            if not all(value == value and abs(value) != float("inf") for value in values.values()):
                raise ValueError("non-finite feature")
            label = row["label"].strip()
            if not label:
                raise ValueError("empty label")
            valid_rows.append({"label": label, **values, "score": score_row(row)})
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"Invalid data at CSV line {index}: {exc}") from exc

    labels = sorted({str(row["label"]) for row in valid_rows})
    print(f"Dataset: {path}")
    print(f"Rows: {len(valid_rows)}")
    print("Counts:")
    for label in labels:
        print(f"  {label}: {sum(row['label'] == label for row in valid_rows)}")

    print("\nPer-label mean ± population standard deviation:")
    for label in labels:
        group = [row for row in valid_rows if row["label"] == label]
        print(f"\n[{label}]")
        for feature in [*FEATURES, "score"]:
            values = [float(row[feature]) for row in group]
            print(f"  {feature}: {mean(values):.4f} ± {pstdev(values):.4f}")

    if "controlled" in labels and "varied" in labels:
        controlled = [row for row in valid_rows if row["label"] == "controlled"]
        varied = [row for row in valid_rows if row["label"] == "varied"]
        print("\nControlled vs varied comparison (mean difference: varied - controlled):")
        for feature in [*FEATURES, "score"]:
            c_values = [float(row[feature]) for row in controlled]
            v_values = [float(row[feature]) for row in varied]
            diff = mean(v_values) - mean(c_values)
            print(f"  {feature}: {diff:+.4f}")
        c_scores = [float(row["score"]) for row in controlled]
        v_scores = [float(row["score"]) for row in varied]
        overlap = max(min(c_scores), min(v_scores)) <= min(max(c_scores), max(v_scores))
        print("\nScore range check:")
        print(f"  controlled range: {min(c_scores):.4f} to {max(c_scores):.4f}")
        print(f"  varied range:     {min(v_scores):.4f} to {max(v_scores):.4f}")
        print(f"  ranges overlap:   {'yes' if overlap else 'no'}")
        if mean(v_scores) < mean(c_scores):
            print("  Direction: varied has a lower mean score in this dataset.")
        else:
            print("  Direction: varied does NOT have a lower mean score in this dataset.")

    if "controlled" in labels and "varied" in labels:
        controlled_scores = [float(row["score"]) for row in valid_rows if row["label"] == "controlled"]
        varied_scores = [float(row["score"]) for row in valid_rows if row["label"] == "varied"]
        candidates = sorted({*controlled_scores, *varied_scores})
        candidates = [min(controlled_scores + varied_scores) - 1e-9, *candidates,
                      max(controlled_scores + varied_scores) + 1e-9]
        best = None
        for threshold in candidates:
            # Scores at/above threshold predict controlled; lower scores predict varied.
            tp = sum(score < threshold for score in varied_scores)
            fn = len(varied_scores) - tp
            tn = sum(score >= threshold for score in controlled_scores)
            fp = len(controlled_scores) - tn
            sensitivity = tp / len(varied_scores)
            specificity = tn / len(controlled_scores)
            balanced_accuracy = (sensitivity + specificity) / 2
            result = (balanced_accuracy, threshold, tp, fn, tn, fp, sensitivity, specificity)
            if best is None or result[0] > best[0]:
                best = result
        assert best is not None
        ba, threshold, tp, fn, tn, fp, sensitivity, specificity = best
        print("\nExploratory threshold sweep (same data used to choose threshold):")
        print(f"  Best threshold: {threshold:.4f} (score below threshold predicts varied)")
        print(f"  Confusion matrix: TP varied={tp}, FN varied={fn}, TN controlled={tn}, FP controlled={fp}")
        print(f"  Sensitivity for varied: {sensitivity:.3f}")
        print(f"  Specificity for controlled: {specificity:.3f}")
        print(f"  Balanced accuracy: {ba:.3f}")
        print("  WARNING: optimistic in-sample result; threshold was selected and tested on the same rows.")

    print("\nInterpretation limits:")
    print("- This is descriptive analysis of movement-protocol labels only.")
    print("- Overlapping ranges or small class differences suggest weak separation.")
    print("- Non-overlapping ranges in a small dataset do not prove generalization.")
    print("- These results do not validate fear, stress, fatigue, cognitive state,")
    print("  or a real-world operator-stability measure.")


if __name__ == "__main__":
    main()

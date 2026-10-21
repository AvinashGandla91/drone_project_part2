"""Train and evaluate a Random Forest on labelled reference hand poses."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from .gesture_commands import GESTURE_LABELS

FEATURE_COLUMNS = [f"f{i}" for i in range(63)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/gesture_landmarks.csv")
    parser.add_argument("--model-out", default="models/gesture_random_forest.joblib")
    args = parser.parse_args()
    path = Path(args.data)
    if not path.is_file():
        raise SystemExit(f"Dataset not found: {path}. Run python -m drone_safety.collect_gesture_dataset first.")
    data = pd.read_csv(path).dropna(subset=["label", *FEATURE_COLUMNS])
    unknown = sorted(set(data["label"].astype(str)) - set(GESTURE_LABELS))
    if unknown:
        raise SystemExit(f"Unknown labels in dataset: {unknown}")
    counts = data["label"].value_counts()
    missing = [label for label in GESTURE_LABELS if counts.get(label, 0) < 5]
    if missing:
        raise SystemExit("Collect at least 5 examples for every label before training. Missing/low: " + ", ".join(missing))
    x = data[FEATURE_COLUMNS].astype(float)
    y = data["label"].astype(str)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    print("Gesture Random Forest evaluation (holdout from collected samples):")
    print(classification_report(y_test, pred, labels=list(GESTURE_LABELS), zero_division=0))
    print("Balanced accuracy:", round(balanced_accuracy_score(y_test, pred), 3))
    print("Confusion matrix label order:", list(GESTURE_LABELS))
    print(confusion_matrix(y_test, pred, labels=list(GESTURE_LABELS)))
    out = Path(args.model_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS,
                 "labels": list(GESTURE_LABELS)}, out)
    print(f"Saved model: {out}")
    print("Collect independent sessions for a reliable evaluation; this random split is exploratory.")


if __name__ == "__main__":
    main()

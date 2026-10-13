"""Train an experimental model on a collected movement-feature CSV.

The model predicts the collector's movement-protocol labels only. It does not
predict fatigue, stress, or a medical/cognitive condition.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib

FEATURE_COLUMNS = [
    "tremor_proxy",
    "trajectory_smoothness",
    "gesture_consistency",
    "velocity_fluctuation",
    "motion_entropy",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/movement_features.csv")
    parser.add_argument("--model-out", default="models/movement_protocol_model.joblib")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(
            f"Dataset not found: {data_path}. Collect data first with "
            "python -m drone_safety.collect_dataset"
        )

    dataset = pd.read_csv(data_path)
    required = set(FEATURE_COLUMNS + ["label"])
    missing = required.difference(dataset.columns)
    if missing:
        raise SystemExit(f"CSV is missing required columns: {', '.join(sorted(missing))}")
    dataset = dataset.dropna(subset=FEATURE_COLUMNS + ["label"])
    if dataset["label"].nunique() < 2:
        raise SystemExit("At least two distinct labels are required.")
    if dataset["label"].value_counts().min() < 2:
        raise SystemExit("Each label needs at least two rows for a stratified split.")

    x_train, x_test, y_train, y_test = train_test_split(
        dataset[FEATURE_COLUMNS],
        dataset["label"].astype(str),
        test_size=0.25,
        random_state=42,
        stratify=dataset["label"].astype(str),
    )
    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=250,
            class_weight="balanced",
            random_state=42,
        ),
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    print("Holdout results for collector movement labels only:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Warning: random window split can overestimate performance when windows")
    print("from the same recording/person appear in both train and test sets.")
    print("For research reporting, collect multiple sessions/participants and use")
    print("participant- or session-separated evaluation.")

    output = Path(args.model_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS}, output)
    print(f"Saved model to {output}")


if __name__ == "__main__":
    main()

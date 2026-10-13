"""Train an experimental model on a collected movement-feature CSV.

The model predicts the collector's movement-protocol labels only. It does not
predict fatigue, stress, or a medical/cognitive condition.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
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
    parser.add_argument("--data", default="data/movement_features_sessions.csv")
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

    labels_series = dataset["label"].astype(str)
    has_sessions = "session_id" in dataset.columns
    session_count = dataset["session_id"].nunique() if has_sessions else 0

    if has_sessions and session_count >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        train_idx, test_idx = next(
            splitter.split(dataset[FEATURE_COLUMNS], labels_series, groups=dataset["session_id"])
        )
        x_train = dataset.iloc[train_idx][FEATURE_COLUMNS]
        x_test = dataset.iloc[test_idx][FEATURE_COLUMNS]
        y_train = labels_series.iloc[train_idx]
        y_test = labels_series.iloc[test_idx]
        split_description = "Session-held-out evaluation (exploratory; small number of sessions)"
        train_sessions = set(dataset.iloc[train_idx]["session_id"])
        test_sessions = set(dataset.iloc[test_idx]["session_id"])
        print(f"Training sessions: {sorted(train_sessions)}")
        print(f"Test sessions: {sorted(test_sessions)}")
        if set(y_train.unique()) != set(labels_series.unique()):
            raise SystemExit(
                "Training split does not contain every label. Collect more balanced "
                "sessions before evaluating."
            )
        if set(y_test.unique()) != set(labels_series.unique()):
            print("WARNING: Test session does not contain every label; metrics may be incomplete.")
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            dataset[FEATURE_COLUMNS],
            labels_series,
            test_size=0.25,
            random_state=42,
            stratify=labels_series,
        )
        split_description = "Random stratified window split (exploratory only)"
        print("WARNING: No usable session IDs; random split can leak recording-specific patterns.")
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

    print("Dataset summary (movement-protocol labels only):")
    print(dataset["label"].astype(str).value_counts().to_string())
    print("\nRandom stratified holdout results (exploratory only):")
    print(classification_report(y_test, predictions, zero_division=0))
    print(f"Balanced accuracy: {balanced_accuracy_score(y_test, predictions):.3f}")
    labels = sorted(dataset["label"].astype(str).unique())
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    print("Confusion matrix (rows=true, columns=predicted; labels=" + str(labels) + "):")
    print(matrix)
    print("\nIMPORTANT LIMITATION:")
    print("- Session-based results from only a few sessions are highly uncertain.")
    print("- A random window split may put samples from the same recording/person")
    print("  in both train and test sets, inflating the apparent score.")
    print("- These labels represent controlled/varied movement protocol only.")
    print("- This does NOT evaluate fatigue, stress, cognitive state, or drone safety.")
    print("- For a stronger evaluation, collect multiple separately identified sessions")
    print("  (or participants) and reserve entire sessions/participants for testing.")

    output = Path(args.model_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS}, output)
    print(f"Saved model to {output}")


if __name__ == "__main__":
    main()

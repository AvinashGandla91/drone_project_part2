"""Explore published TremorComputerVision summary tables.

This validates agreement/association between published tremor measurements.
It is not a fatigue, stress, or cognitive-stability classifier and does not
train on raw landmark sequences.

Run: python -m drone_safety.analyze_external_dataset
Optional: --refresh to download the source tables again.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

BASE_URL = "https://raw.githubusercontent.com/JuliusWelzel/TremorComputerVision/main/figdata"
FILES = {
    "displacements": "hum_displacements_models.csv",
    "peaks": "human_peaks_table.csv",
    "patient_info": "hum_patient_info.csv",
}
FREQUENCY_METHODS = ("mocap", "imu", "mp", "apple")
FREQUENCY_STAGES = ("raw", "pca", "emd")
FREQUENCY_PAIRS = (
    ("apple", "mocap", "Vision", "Mocap"),
    ("mp", "mocap", "MediaPipe", "Mocap"),
    ("apple", "imu", "Vision", "IMU"),
    ("mp", "imu", "MediaPipe", "IMU"),
)
DISPLACEMENT_COLUMNS = (
    "mocap_displacement",
    "mp_world_displacement",
    "mp_world_z_displacement",
    "mp_norm_displacement",
    "mp_norm_displacement_z",
    "apple_displacement",
)


def download_file(url: str, destination: Path) -> None:
    """Download a public source file, preserving a local copy for reproducibility."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=30) as response:
        destination.write_bytes(response.read())


def pearson_correlation(x: pd.Series, y: pd.Series) -> float:
    """Return Pearson correlation or NaN if either series is constant/too short."""
    if len(x) < 2 or x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    return float(np.corrcoef(x.to_numpy(dtype=float), y.to_numpy(dtype=float))[0, 1])


def frequency_comparison_table(peaks: pd.DataFrame) -> pd.DataFrame:
    """Summarize paired absolute frequency differences by method and pipeline."""
    rows = []
    for stage in FREQUENCY_STAGES:
        for left, right, left_name, right_name in FREQUENCY_PAIRS:
            left_col = f"{left}_{stage}_freq"
            right_col = f"{right}_{stage}_freq"
            if left_col not in peaks or right_col not in peaks:
                continue
            pair = peaks[[left_col, right_col]].apply(pd.to_numeric, errors="coerce").dropna()
            differences = (pair[left_col] - pair[right_col]).abs()
            if differences.empty:
                continue
            rows.append({
                "stage": stage.upper(),
                "comparison": f"{left_name} vs {right_name}",
                "n_paired_subjects": int(len(differences)),
                "median_absolute_frequency_difference": float(differences.median()),
                "iqr_25": float(differences.quantile(0.25)),
                "iqr_75": float(differences.quantile(0.75)),
            })
    return pd.DataFrame(rows)


def tetras_association_table(merged: pd.DataFrame) -> pd.DataFrame:
    """Calculate exploratory Kendall tau associations between TETRAS and displacements."""
    if "TETRAS" not in merged:
        return pd.DataFrame(columns=["measurement", "n", "kendall_tau", "p_value"])
    tetras = pd.to_numeric(merged["TETRAS"], errors="coerce")
    rows = []
    for column in DISPLACEMENT_COLUMNS:
        if column not in merged:
            continue
        values = pd.to_numeric(merged[column], errors="coerce")
        pair = pd.DataFrame({"TETRAS": tetras, "measurement": values}).dropna()
        if len(pair) < 2 or pair["TETRAS"].nunique() < 2 or pair["measurement"].nunique() < 2:
            tau, p_value = float("nan"), float("nan")
        else:
            result = kendalltau(pair["TETRAS"], pair["measurement"], nan_policy="omit")
            tau, p_value = float(result.statistic), float(result.pvalue)
        rows.append({
            "measurement": column,
            "n": int(len(pair)),
            "kendall_tau": tau,
            "p_value": p_value,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/external_tremor")
    parser.add_argument("--refresh", action="store_true", help="Download source files again")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    paths = {name: data_dir / filename for name, filename in FILES.items()}
    for name, filename in FILES.items():
        if args.refresh or not paths[name].exists():
            print(f"Downloading {filename}...")
            download_file(f"{BASE_URL}/{filename}", paths[name])

    displacement = pd.read_csv(paths["displacements"])
    patient_info = pd.read_csv(paths["patient_info"], sep=";")
    peaks = pd.read_csv(paths["peaks"])

    required = {"id", "mocap_displacement", "mp_norm_displacement"}
    missing = required.difference(displacement.columns)
    if missing:
        raise SystemExit(f"Source displacement table is missing columns: {sorted(missing)}")
    if "id" not in peaks.columns:
        raise SystemExit("Source frequency table is missing required column: id")

    merged = displacement.merge(
        patient_info,
        left_on="id",
        right_on="subject_number",
        how="left",
        validate="one_to_one",
    )
    valid = merged[["mocap_displacement", "mp_norm_displacement"]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()
    errors = valid["mp_norm_displacement"] - valid["mocap_displacement"]
    correlation = pearson_correlation(
        valid["mp_norm_displacement"], valid["mocap_displacement"]
    )

    print("\nPublished human tremor dataset — exploratory comparison")
    print(f"Displacement records compared: {len(valid)}")
    print("Reference: optical motion capture (mocap_displacement)")
    print("Computer-vision measure: MediaPipe normalized landmarks (mp_norm_displacement)")
    print(f"Mean absolute difference: {float(np.mean(np.abs(errors))):.3f}")
    print(f"Root mean squared difference: {float(np.sqrt(np.mean(np.square(errors)))):.3f}")
    print(f"Pearson correlation: {correlation:.3f}")
    print("Note: these are differences between published measurement values; confirm units/scales before interpreting them as physical error.")

    frequency_results = frequency_comparison_table(peaks)
    frequency_output = data_dir / "frequency_comparison_summary.csv"
    frequency_results.to_csv(frequency_output, index=False)
    print("\nPaired frequency comparisons (absolute differences):")
    if frequency_results.empty:
        print("No compatible frequency columns found.")
    else:
        print(frequency_results.to_string(index=False))
    print(f"Saved frequency comparison table to: {frequency_output}")

    association_results = tetras_association_table(merged)
    association_output = data_dir / "tetras_displacement_correlations.csv"
    association_results.to_csv(association_output, index=False)
    print("\nExploratory Kendall tau association: TETRAS vs displacement:")
    print(association_results.to_string(index=False))
    print(f"Saved TETRAS association table to: {association_output}")

    output = data_dir / "merged_tremor_summary.csv"
    merged.to_csv(output, index=False)
    print(f"\nSaved merged source summary to: {output}")
    print("\nLimitations:")
    print("- The dataset concerns tremor measurement, not fatigue, stress, or cognitive stability.")
    print("- This script analyzes published summary measurements, not raw landmark windows.")
    print("- The dataset is small; association does not establish causation or diagnostic performance.")
    print("- Frequency comparisons are paired by subject, but are descriptive summaries.")
    print("- Do not interpret these results as proof of drone safety.")


if __name__ == "__main__":
    main()

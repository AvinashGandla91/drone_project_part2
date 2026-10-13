"""Download and analyze the published TremorComputerVision summary tables.

This is an exploratory validation of published tremor-amplitude/displacement
measurements. It is NOT a fatigue/stress classifier and does not train on
individual landmark sequences. Run with: python -m drone_safety.analyze_external_dataset
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/JuliusWelzel/TremorComputerVision/main/figdata"
FILES = {
    "displacements": "hum_displacements_models.csv",
    "peaks": "human_peaks_table.csv",
    "patient_info": "hum_patient_info.csv",
}


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

    merged = displacement.merge(
        patient_info,
        left_on="id",
        right_on="subject_number",
        how="left",
        validate="one_to_one",
    )
    valid = merged[["mocap_displacement", "mp_norm_displacement"]].dropna()
    errors = valid["mp_norm_displacement"] - valid["mocap_displacement"]
    correlation = pearson_correlation(
        valid["mp_norm_displacement"], valid["mocap_displacement"]
    )

    print("\nPublished human tremor dataset — exploratory comparison")
    print(f"Displacement records compared: {len(valid)}")
    print("Reference: optical motion capture (mocap_displacement)")
    print("Computer-vision measure: MediaPipe normalized landmarks (mp_norm_displacement)")
    print(f"Mean absolute error: {float(np.mean(np.abs(errors))):.3f}")
    print(f"Root mean squared error: {float(np.sqrt(np.mean(np.square(errors)))):.3f}")
    print(f"Pearson correlation: {correlation:.3f}")
    print("\nTETRAS score association (exploratory; small sample):")
    tetras = pd.to_numeric(merged["TETRAS"], errors="coerce")
    tetras_valid = merged.loc[tetras.notna()].copy()
    tetras_valid["TETRAS"] = pd.to_numeric(tetras_valid["TETRAS"], errors="coerce")
    amplitude_corr = pearson_correlation(
        tetras_valid["TETRAS"], tetras_valid["mp_norm_displacement"]
    )
    print(f"Records with TETRAS scores: {len(tetras_valid)}")
    print(f"Pearson correlation, TETRAS vs MP normalized displacement: {amplitude_corr:.3f}")

    output = data_dir / "merged_tremor_summary.csv"
    merged.to_csv(output, index=False)
    print(f"Saved merged source summary to: {output}")
    print("\nLimitations:")
    print("- This dataset concerns tremor measurement, not fatigue or stress.")
    print("- This script analyzes published summary measurements, not raw landmark windows.")
    print("- Small sample size and measurement-method differences limit conclusions.")
    print("- Do not interpret correlations as a diagnosis or as proof of drone safety.")


if __name__ == "__main__":
    main()

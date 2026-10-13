# External research dataset: TremorComputerVision

## Source

- Repository: https://github.com/JuliusWelzel/TremorComputerVision
- Paper: Wolke et al., "Validity of tremor analysis using smartphone compatible computer vision frameworks" (Scientific Reports, 2025): https://pmc.ncbi.nlm.nih.gov/articles/PMC12008214/
- The paper states that anonymous hand-tracking data and ground-truth data are available in the repository; original videos and detailed clinical data are not shared for legal reasons.
- Relevant source folder: `rawdata/human/mp_norm/` (MediaPipe normalized hand landmark data); related folders include `mp_world`, `apple_vision`, `imu`, and `mocap`.
- Public summary tables include `figdata/hum_displacements_models.csv`, `figdata/human_peaks_table.csv`, and `figdata/hum_patient_info.csv`.

## Important scope

This is a tremor-analysis research dataset, not a general hand-gesture dataset and not a dataset labelled for fatigue, stress, or cognitive stability. It can help us compare published computer-vision displacement measurements against optical motion-capture reference measurements. Do not relabel samples as "fatigued", "stressed", "stable", or "unstable" unless source labels support that interpretation.

## Integrated analysis

The project now includes `src/drone_safety/analyze_external_dataset.py`. It downloads the three small published summary tables into `data/external_tremor/`, merges the displacement and patient-summary tables, compares MediaPipe normalized-landmark displacement with optical motion-capture displacement, prints exploratory error/correlation statistics, and saves a merged CSV.

Run from the project root:

```powershell
.\.venv312\Scripts\python.exe -m pip install -e ".[ml]"
.\.venv312\Scripts\python.exe -m drone_safety.analyze_external_dataset
```

Add `--refresh` to download the upstream tables again. Internet access is required for the first run.

## Raw landmark data: not yet adapted

The upstream repository also contains multi-megabyte pickle files with hand-tracking data. These files are deliberately not copied into this project's Git history. A separate raw-landmark adapter is still needed: it must inspect the source structure, preserve time ordering, handle missing landmarks, and align sequences with appropriate reference measurements. Until that is implemented and validated, do not claim the raw landmarks are connected to this project's model-training pipeline.

## Interpretation and safety

- The public summary table is a small exploratory measurement comparison, not a training dataset for fatigue/stress or cognitive-state classification.
- The current `controlled` / `varied` webcam collector labels are movement-protocol labels, not clinical labels. Never combine them with tremor dataset measurements as if they mean the same thing.
- Correlation is not proof of agreement, diagnosis, or drone safety. The physical drone remains disconnected; all command gating is simulation-only.

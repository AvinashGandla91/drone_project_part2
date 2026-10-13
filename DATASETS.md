# External research dataset: TremorComputerVision

## Source

- Repository: https://github.com/JuliusWelzel/TremorComputerVision
- Paper: Wolke et al., "Validity of tremor analysis using smartphone compatible computer vision frameworks" (Scientific Reports, 2025): https://www.nature.com/articles/s41598-025-97252-4
- The paper states that anonymous hand-tracking data and ground-truth data are available in the repository; original videos and detailed clinical data are not shared for legal reasons.
- Relevant source folder: `rawdata/human/mp_norm/` (MediaPipe normalized hand landmark data); related folders include `mp_world`, `apple_vision`, `imu`, and `mocap`.

## Important scope

This is a tremor-analysis research dataset, not a general hand-gesture dataset and not a dataset labelled for fatigue, stress, or cognitive stability. It can help us investigate movement-feature extraction and compare our tremor-like movement proxy against the dataset's documented ground truth. Do not relabel its samples as "fatigued", "stressed", "stable", or "unstable" unless the source's actual labels support that interpretation.

## Keep the data outside this repository

The source data files are multi-megabyte pickle files and should not be copied into this project's Git history. Download the upstream repository separately and retain its own README, metadata, and attribution. Only derived, appropriately licensed feature tables should be added here, if allowed by the source terms.

From the project folder, run:

```powershell
git clone https://github.com/JuliusWelzel/TremorComputerVision.git data\TremorComputerVision
```

The upstream repository can contain research data and Python pickle files. Only load pickle files from the verified upstream source, never from an untrusted file.

## Integration status

The source has been identified, and this document records the provenance and intended use. A dataset adapter must first inspect the exact pickle structure and source metadata, then map the correct continuous landmark tracks and ground-truth values into this project's feature schema. Until that mapping is implemented and tested, the dataset is **not yet connected to model training**.

The project's current `controlled` / `varied` webcam collector labels are protocol labels, not clinical labels. Do not combine them with the tremor dataset's labels as if they mean the same thing.

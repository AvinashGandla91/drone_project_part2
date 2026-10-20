# Cognitive Stability-Aware Safety Framework for Adaptive Vision-Based Drone Navigation

Project II extends a vision-based hand-gesture drone controller with a movement-stability layer and safety-aware command validation.

## Project II goals

1. Extract movement features from a time-ordered sequence of hand landmarks:
   - tremor-like high-frequency movement (an approximate computer-vision proxy, not a clinical measurement)
   - trajectory smoothness
   - gesture/pose consistency
   - velocity fluctuation
   - motion-direction entropy
2. Produce an experimental stability estimate from those measurable features.
3. Combine gesture confidence with movement stability before issuing a simulated command.
4. Evaluate the system with repeatable tests before connecting it to any physical UAV.

## Safety and research limitations

- This project does **not** diagnose fatigue, stress, attention, or any medical/cognitive condition.
- A hand-camera signal alone cannot reliably establish a person's cognitive state. Until validated on an appropriate dataset, the stability label is an experimental movement-stability proxy.
- The included command gate is simulation-only. It does not connect to or arm a real drone.
- Never test unvalidated logic on a flying UAV. Use a simulator and follow your lab's safety procedures.
- Do not report model accuracy from synthetic/demo data as real-world performance.

## Repository structure

```
src/drone_safety/
  features.py                     # Movement features computed from 2D/3D landmark tracks
  decision_fusion.py              # Gesture-confidence + stability safety gate
  demo.py                          # Synthetic command-gate demonstration
  webcam_monitor.py                # Live webcam movement and gesture monitor
  gesture_commands.py              # Reference-pose labels and simulation command mapping
  collect_dataset.py               # Consent-based movement-protocol data collector
  train_stability_model.py         # Experimental controlled/varied protocol classifier
  analyze_external_dataset.py      # Published tremor summary-table analysis
tests/
  test_features.py
  test_decision_fusion.py
```

## Setup

Python 3.10 or newer is recommended.

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest
python -m drone_safety.demo
```

If PowerShell blocks activation, run commands using the environment's `Scripts\python.exe` instead.

## Reference-pose gesture commands

With `--commands`, the webcam monitor now classifies static hand poses instead of inferring left/right/up/down from palm movement. The intended labels are `LEFT`, `RIGHT`, `FORWARD`, `BACKWARD`, `LAND`, `UPWARD`, and `STOP` (closed fist). `STOP` maps to the simulation `HOLD` state. Directional commands are displayed only after two matching frame classifications; a closed fist stops the directional command immediately.

The current classifier uses geometric heuristics over MediaPipe landmarks. It is a prototype, not a trained or validated seven-class model, and some poses may be ambiguous under different camera angles or mirrored views. Test and calibrate it with the exact reference hand poses before relying on its labels. All outputs are simulation-only; no physical drone is controlled.

Run the command display with:

```powershell
.\.venv312\Scripts\python.exe -m drone_safety.webcam_monitor --commands
```

## Input format

`extract_movement_features()` accepts a finite array with shape `(frames, landmarks, dimensions)`, where dimensions are 2 or 3. Coordinates should use a consistent normalization and coordinate system across frames. Provide at least 8 frames for meaningful temporal features.

The webcam monitor detects hand landmarks with MediaPipe. The seven-pose classifier is a separate prototype and does not require the Part I repository, but it still needs validation against the actual reference poses.

## Published external dataset: TremorComputerVision

We have added a script that downloads and analyzes published summary tables from [TremorComputerVision](https://github.com/JuliusWelzel/TremorComputerVision), a research project validating computer-vision tremor measurements against reference measurements. The associated paper reports 20 tremor-patient recordings and states that anonymous hand-tracking and ground-truth data are shared, while original videos and detailed clinical data are not available for legal reasons: [Scientific Reports paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12008214/).

Run this from the project root in PowerShell:

```powershell
.\.venv312\Scripts\python.exe -m pip install -e ".[ml]"
.\.venv312\Scripts\python.exe -m drone_safety.analyze_external_dataset
```

The script downloads three small public summary tables into `data/external_tremor/`, compares the published MediaPipe normalized-landmark displacement with optical motion-capture displacement, prints exploratory error/correlation statistics, and writes a merged table. Use `--refresh` to download the source tables again.

**Important:** This is a real external dataset analysis, but it is not yet raw landmark-window integration or model training. These published summary tables are suitable for a small exploratory measurement comparison, not for training a reliable fatigue/stress classifier. Do not interpret TETRAS association as diagnosis or as proof of safe drone operation. The source pickle files are not loaded by this script.

See [DATASETS.md](DATASETS.md) for provenance and scope.

## Independent Part II webcam workflow

Part II can run without the Part I group's code. The webcam monitor reads hand landmarks directly from your camera using MediaPipe.

Install the optional computer-vision dependencies:

```powershell
.\.venv312\Scripts\python.exe -m pip install -e ".[vision]"
.\.venv312\Scripts\python.exe -m drone_safety.webcam_monitor
```

Press `Q` or `Esc` to exit. If the default camera does not open, try `--camera 1`.

### Collect your own movement-protocol data

The collector records windows labelled `controlled` or `varied`. These labels describe the recording exercise only; they are **not** labels for fatigue, stress or cognitive state.

```powershell
.\.venv312\Scripts\python.exe -m drone_safety.collect_dataset
```

- Press `1` to select controlled movement.
- Press `2` to select varied movement.
- Hold a consistent recording setup and press `Space` when a full window is available.
- Press `Q` or `Esc` to stop.

Only collect data with informed consent. This tool saves numeric features only, not video. For meaningful research, record multiple sessions/participants under a documented protocol.

### Train a first experimental model

```powershell
.\.venv312\Scripts\python.exe -m pip install -e ".[ml]"
.\.venv312\Scripts\python.exe -m drone_safety.train_stability_model
```

The initial model predicts the collector's movement-protocol labels, not a person's cognitive state. A random window split may leak recording-specific patterns; proper research evaluation should split by participant or session and report limitations.

## Current independence boundary

This version includes its own webcam-to-landmark-to-feature path and simulated safety gate. It does not import Part I code, does not execute commands on a physical UAV, and does not establish a validated cognitive-stability classifier. Physical drone integration should remain out of scope until the system is tested and reviewed under lab safety procedures.

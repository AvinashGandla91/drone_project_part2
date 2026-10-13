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
  features.py          # Movement features computed from 2D/3D landmark tracks
  decision_fusion.py   # Gesture-confidence + stability safety gate
  demo.py              # Small command-line demonstration
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
python -m pip install --upgrade pip
python -m pip install -e .[dev]
pytest
python -m drone_safety.demo
```

If PowerShell blocks activation, run commands using `.venv\Scripts\python.exe` instead.

## Input format

`extract_movement_features()` accepts a finite array with shape `(frames, landmarks, dimensions)`, where dimensions are 2 or 3. Coordinates should use a consistent normalization and coordinate system across frames. Provide at least 8 frames for meaningful temporal features.

The current implementation assumes the hand has already been detected and landmark points have already been extracted (for example, by the project's existing OpenCV/MediaPipe stage). It does not replace Part I gesture recognition.

## Next development milestones

- [x] Repository starter and documented project scope
- [x] First movement feature extractor and rule-based experimental score
- [x] Conservative simulated command gate
- [x] Unit tests for core feature and fusion behavior
- [ ] Connect features to the existing MediaPipe landmark stream
- [ ] Collect a consent-based, labelled movement dataset with a defined protocol
- [ ] Compare and validate candidate models using participant-separated evaluation
- [ ] Add plots, confusion matrix and latency measurements
- [ ] Integrate with the existing gesture pipeline in simulation
- [ ] Evaluate lighting, background and partial-occlusion conditions

## Research reporting

Record the dataset, participants/recording protocol, train/test split, thresholds, false accepts, false holds and latency. Keep synthetic checks separate from real-world evaluation and state limitations explicitly.

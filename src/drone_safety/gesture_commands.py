"""Seven reference-pose command labels and landmark preprocessing.

Order follows the supplied reference photos: LEFT, RIGHT, FORWARD, BACKWARD,
LAND (edge-on hand), UPWARD (spread fingers), and STOP (closed fist).
"""
from __future__ import annotations
from typing import Sequence
import numpy as np

GESTURE_LABELS = ("LEFT", "RIGHT", "FORWARD", "BACKWARD", "LAND", "UPWARD", "STOP")
GESTURE_COMMANDS = {
    "LEFT": "SIMULATED_LEFT",
    "RIGHT": "SIMULATED_RIGHT",
    "FORWARD": "SIMULATED_FORWARD",
    "BACKWARD": "SIMULATED_BACKWARD",
    "LAND": "SIMULATED_LAND",
    "UPWARD": "SIMULATED_UPWARD",
    "STOP": "HOLD",
}
FEATURE_COLUMNS = [f"f{i}" for i in range(63)]


def command_for_gesture(gesture: str) -> str:
    """Map a recognized gesture to a simulation-only command."""
    normalized = gesture.strip().upper()
    try:
        return GESTURE_COMMANDS[normalized]
    except KeyError as exc:
        raise ValueError(f"Unknown gesture label: {gesture!r}") from exc


def normalize_landmarks(landmarks: Sequence[Sequence[float]]) -> np.ndarray:
    """Normalize 21 MediaPipe landmarks to wrist origin and palm-width scale."""
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (21, 3):
        raise ValueError("landmarks must have shape (21, 3)")
    if not np.isfinite(points).all():
        raise ValueError("landmarks must contain only finite coordinates")
    points = points - points[0]
    palm_width = float(np.linalg.norm(points[5, :2] - points[17, :2]))
    if palm_width < 1e-6:
        raise ValueError("palm width is too small to normalize landmarks")
    return (points / palm_width).reshape(63)


def predict_gesture(model, landmarks: Sequence[Sequence[float]], min_confidence: float = 0.75):
    """Return (label, confidence); uncertain predictions become UNKNOWN."""
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be between 0 and 1")
    sample = normalize_landmarks(landmarks).reshape(1, -1)
    probabilities = model.predict_proba(sample)[0]
    index = int(np.argmax(probabilities))
    confidence = float(probabilities[index])
    label = str(model.classes_[index]) if confidence >= min_confidence else "UNKNOWN"
    return label, confidence

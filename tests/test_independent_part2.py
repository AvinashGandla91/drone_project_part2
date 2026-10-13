import numpy as np
import pytest

from drone_safety.decision_fusion import DecisionResult, validate_command
from drone_safety.features import extract_movement_features


def test_feature_pipeline_does_not_need_part_one():
    # Part II accepts landmark arrays directly; no external Part I package needed.
    frames, landmarks = 12, 21
    track = np.zeros((frames, landmarks, 3), dtype=float)
    track[:, :, 0] = np.linspace(0.0, 0.2, frames)[:, None]
    features = extract_movement_features(track)
    assert features.frame_count == frames


def test_safety_gate_can_be_exercised_independently():
    decision = validate_command("SIMULATED_FORWARD", 0.95, 0.9)
    assert decision.result == DecisionResult.ACCEPT
    assert decision.simulated_command == "SIMULATED_FORWARD"


def test_nonfinite_landmarks_fail_clearly():
    track = np.zeros((10, 21, 3), dtype=float)
    track[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        extract_movement_features(track)

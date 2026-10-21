import numpy as np
import pytest

from drone_safety.gesture_commands import (
    GESTURE_LABELS, command_for_gesture, normalize_landmarks, predict_gesture,
)


@pytest.mark.parametrize(
    ("gesture", "command"),
    [
        ("LEFT", "SIMULATED_LEFT"),
        ("RIGHT", "SIMULATED_RIGHT"),
        ("FORWARD", "SIMULATED_FORWARD"),
        ("BACKWARD", "SIMULATED_BACKWARD"),
        ("LAND", "SIMULATED_LAND"),
        ("UPWARD", "SIMULATED_UPWARD"),
        ("STOP", "HOLD"),
    ],
)
def test_gesture_maps_to_expected_simulation_command(gesture, command):
    assert command_for_gesture(gesture) == command


def test_all_seven_labels_are_present_in_reference_order():
    assert GESTURE_LABELS == ("LEFT", "RIGHT", "FORWARD", "BACKWARD", "LAND", "UPWARD", "STOP")


def test_gesture_mapping_is_case_insensitive():
    assert command_for_gesture("stop") == "HOLD"


def test_unknown_gesture_is_rejected():
    with pytest.raises(ValueError, match="Unknown gesture"):
        command_for_gesture("JUMP")


def test_invalid_landmark_shape_is_rejected():
    with pytest.raises(ValueError, match="shape"):
        normalize_landmarks(np.zeros((10, 3)))


def test_nonfinite_landmarks_are_rejected():
    points = np.zeros((21, 3))
    points[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        normalize_landmarks(points)


def test_degenerate_palm_is_rejected():
    with pytest.raises(ValueError, match="palm width"):
        normalize_landmarks(np.zeros((21, 3)))


class FakeModel:
    classes_ = np.array(["LEFT", "STOP"])
    def predict_proba(self, sample):
        assert sample.shape == (1, 63)
        return np.array([[0.1, 0.9]])


def test_model_prediction_and_confidence():
    points = np.zeros((21, 3))
    points[5, 0] = -0.5
    points[17, 0] = 0.5
    label, confidence = predict_gesture(FakeModel(), points)
    assert label == "STOP"
    assert confidence == pytest.approx(0.9)


def test_low_confidence_returns_unknown():
    points = np.zeros((21, 3))
    points[5, 0] = -0.5
    points[17, 0] = 0.5
    label, confidence = predict_gesture(FakeModel(), points, min_confidence=0.95)
    assert label == "UNKNOWN"
    assert confidence == pytest.approx(0.9)

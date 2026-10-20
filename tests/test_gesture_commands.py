import numpy as np
import pytest

from drone_safety.gesture_commands import classify_hand_pose, command_for_gesture


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


def test_gesture_mapping_is_case_insensitive():
    assert command_for_gesture("stop") == "HOLD"


def test_unknown_gesture_is_rejected():
    with pytest.raises(ValueError, match="Unknown gesture"):
        command_for_gesture("JUMP")


def test_invalid_landmark_shape_is_rejected():
    with pytest.raises(ValueError, match="shape \(21, 3\)"):
        classify_hand_pose(np.zeros((10, 3)))

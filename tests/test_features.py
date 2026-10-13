import numpy as np
import pytest

from drone_safety.features import (
    experimental_stability_score,
    extract_movement_features,
    movement_stability_status,
)


def test_features_are_bounded_and_finite_for_valid_input():
    rng = np.random.default_rng(12)
    landmarks = rng.normal(size=(20, 21, 2)) * 0.01
    result = extract_movement_features(landmarks)

    assert result.frame_count == 20
    assert 0 <= result.trajectory_smoothness <= 1
    assert 0 <= result.gesture_consistency <= 1
    assert 0 <= result.motion_entropy <= 1
    assert np.isfinite(result.tremor_proxy)
    assert np.isfinite(result.velocity_fluctuation)
    assert 0 <= experimental_stability_score(result) <= 1


@pytest.mark.parametrize(
    "bad_input",
    [
        np.zeros((5, 21, 2)),
        np.zeros((10, 21, 4)),
        np.zeros((10, 21)),
        np.full((10, 21, 2), np.nan),
    ],
)
def test_invalid_input_is_rejected(bad_input):
    with pytest.raises(ValueError):
        extract_movement_features(bad_input)


def test_stationary_landmarks_have_finite_features():
    landmarks = np.ones((10, 21, 3), dtype=float)
    result = extract_movement_features(landmarks)

    assert result.tremor_proxy == pytest.approx(0.0)
    assert result.velocity_fluctuation == 0.0
    assert result.motion_entropy == 0.0


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.59, "LOW MOVEMENT STABILITY"),
        (0.60, "UNCERTAIN MOVEMENT STABILITY"),
        (0.71, "UNCERTAIN MOVEMENT STABILITY"),
        (0.72, "HIGHER MOVEMENT STABILITY"),
        (1.0, "HIGHER MOVEMENT STABILITY"),
    ],
)
def test_movement_stability_status(score, expected):
    assert movement_stability_status(score) == expected


@pytest.mark.parametrize("score", [-0.01, 1.01])
def test_movement_stability_status_rejects_out_of_range_scores(score):
    with pytest.raises(ValueError):
        movement_stability_status(score)

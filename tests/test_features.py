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



def test_near_stationary_landmark_jitter_does_not_look_unstable():
    rng = np.random.default_rng(7)
    base = np.column_stack((
        np.linspace(0.2, 0.4, 21),
        np.linspace(0.3, 0.5, 21),
        np.zeros(21),
    ))
    landmarks = np.stack([
        base + rng.normal(0.0, 0.0003, size=base.shape)
        for _ in range(30)
    ])
    result = extract_movement_features(landmarks)

    assert result.trajectory_smoothness == pytest.approx(1.0)
    assert result.velocity_fluctuation == 0.0
    assert result.motion_entropy == 0.0

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

def _translated_hand_track(positions):
    """Build deterministic 21-landmark tracks translated along known paths."""
    base = np.linspace(0.0, 1.0, 21)
    shape = np.column_stack((base * 0.02, base * 0.03))
    return np.stack([shape + np.asarray(position) for position in positions])



def test_smooth_fast_translation_is_not_misclassified_as_tremor():
    positions = [(index * 0.025, 0.0) for index in range(30)]
    result = extract_movement_features(_translated_hand_track(positions))

    assert result.tremor_proxy < 1e-8
    assert result.velocity_fluctuation == pytest.approx(0.0, abs=1e-8)
    assert result.trajectory_smoothness > 0.99

def test_smooth_unidirectional_motion_has_low_direction_entropy():
    positions = [(index * 0.01, 0.0) for index in range(30)]
    result = extract_movement_features(_translated_hand_track(positions))

    assert result.motion_entropy == pytest.approx(0.0, abs=1e-8)
    assert result.velocity_fluctuation == pytest.approx(0.0, abs=1e-8)


def test_reversing_and_changing_direction_increases_motion_entropy():
    # Alternate among eight directions so the track occupies multiple direction bins.
    directions = [
        (1, 0), (1, 1), (0, 1), (-1, 1),
        (-1, 0), (-1, -1), (0, -1), (1, -1),
    ]
    positions = [(0.0, 0.0)]
    x, y = 0.0, 0.0
    for index in range(32):
        dx, dy = directions[index % len(directions)]
        step = 0.004 * (1 + (index % 3))
        x += dx * step
        y += dy * step
        positions.append((x, y))

    varied = extract_movement_features(_translated_hand_track(positions))
    smooth_positions = [(index * 0.01, 0.0) for index in range(len(positions))]
    smooth = extract_movement_features(_translated_hand_track(smooth_positions))

    assert varied.motion_entropy > smooth.motion_entropy
    assert 0.0 <= varied.motion_entropy <= 1.0
    assert np.isfinite(varied.velocity_fluctuation)


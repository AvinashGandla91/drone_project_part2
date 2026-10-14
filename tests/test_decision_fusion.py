import pytest

from drone_safety.decision_fusion import (
    DecisionConfig,
    DecisionResult,
    validate_command,
)


def test_high_confidence_and_stability_accepts_in_simulation():
    result = validate_command("MOVE_FORWARD", 0.95, 0.90)
    assert result.result == DecisionResult.ACCEPT
    assert result.simulated_command == "MOVE_FORWARD"


def test_low_gesture_confidence_holds_command():
    result = validate_command("MOVE_LEFT", 0.50, 0.95)
    assert result.result == DecisionResult.HOLD
    assert result.simulated_command is None


def test_low_stability_holds_command():
    result = validate_command("MOVE_RIGHT", 0.95, 0.40)
    assert result.result == DecisionResult.HOLD
    assert result.simulated_command is None


def test_moderate_stability_reduces_sensitivity():
    result = validate_command("MOVE_BACK", 0.90, 0.68)
    assert result.result == DecisionResult.REDUCE_SENSITIVITY
    assert result.simulated_command == "MOVE_BACK"


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("nan"), float("inf"), float("-inf")])
def test_invalid_confidence_is_rejected(confidence):
    with pytest.raises(ValueError):
        validate_command("MOVE_FORWARD", confidence, 0.9)


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        DecisionConfig(min_stability_score=0.8, reduced_sensitivity_score=0.7)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_scores_are_rejected(score):
    with pytest.raises(ValueError):
        validate_command("MOVE_FORWARD", 0.9, score)
    with pytest.raises(ValueError):
        validate_command("MOVE_FORWARD", score, 0.9)


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_configuration_is_rejected(threshold):
    with pytest.raises(ValueError):
        DecisionConfig(min_stability_score=threshold)

"""Conservative command validation for simulation only."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from math import isfinite


class DecisionResult(str, Enum):
    ACCEPT = "accept"
    REDUCE_SENSITIVITY = "reduce_sensitivity"
    HOLD = "hold"


@dataclass(frozen=True)
class DecisionConfig:
    """Thresholds for the experimental command gate."""

    min_gesture_confidence: float = 0.75
    min_stability_score: float = 0.60
    reduced_sensitivity_score: float = 0.72

    def __post_init__(self) -> None:
        for field_name, value in asdict(self).items():
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0 and 1")
        if self.reduced_sensitivity_score < self.min_stability_score:
            raise ValueError(
                "reduced_sensitivity_score must be >= min_stability_score"
            )


@dataclass(frozen=True)
class Decision:
    result: DecisionResult
    reason: str
    gesture_confidence: float
    stability_score: float
    simulated_command: str | None


def validate_command(
    command: str,
    gesture_confidence: float,
    stability_score: float,
    config: DecisionConfig | None = None,
) -> Decision:
    """Decide whether a gesture-derived command may pass the simulation gate.

    This function does not send commands to a drone. The caller may display
    the returned decision or record it in an experiment log.
    """
    settings = config or DecisionConfig()

    if not command or not command.strip():
        raise ValueError("command must be a non-empty string")
    if not isfinite(gesture_confidence) or not 0.0 <= gesture_confidence <= 1.0:
        raise ValueError("gesture_confidence must be finite and between 0 and 1")
    if not isfinite(stability_score) or not 0.0 <= stability_score <= 1.0:
        raise ValueError("stability_score must be finite and between 0 and 1")

    if gesture_confidence < settings.min_gesture_confidence:
        return Decision(
            result=DecisionResult.HOLD,
            reason="Gesture confidence is below the configured threshold.",
            gesture_confidence=gesture_confidence,
            stability_score=stability_score,
            simulated_command=None,
        )

    if stability_score < settings.min_stability_score:
        return Decision(
            result=DecisionResult.HOLD,
            reason="Movement stability score is below the configured threshold.",
            gesture_confidence=gesture_confidence,
            stability_score=stability_score,
            simulated_command=None,
        )

    if stability_score < settings.reduced_sensitivity_score:
        return Decision(
            result=DecisionResult.REDUCE_SENSITIVITY,
            reason="Command passed basic checks; use reduced sensitivity in simulation.",
            gesture_confidence=gesture_confidence,
            stability_score=stability_score,
            simulated_command=command.strip(),
        )

    return Decision(
        result=DecisionResult.ACCEPT,
        reason="Gesture confidence and movement-stability thresholds passed.",
        gesture_confidence=gesture_confidence,
        stability_score=stability_score,
        simulated_command=command.strip(),
    )

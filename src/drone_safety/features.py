"""Experimental movement features for time-ordered hand landmark tracks.

These are computer-vision proxies, not clinical measurements or a diagnosis.
Input coordinates must be consistently scaled and ordered over time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class MovementFeatures:
    """Summary features extracted from a hand landmark trajectory."""

    tremor_proxy: float
    trajectory_smoothness: float
    gesture_consistency: float
    velocity_fluctuation: float
    motion_entropy: float
    frame_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-friendly scalar values."""
        return asdict(self)


def _validate_landmarks(landmarks: ArrayLike) -> NDArray[np.float64]:
    points = np.asarray(landmarks, dtype=np.float64)
    if points.ndim != 3:
        raise ValueError("landmarks must have shape (frames, landmarks, dimensions)")
    if points.shape[0] < 8:
        raise ValueError("at least 8 frames are required for temporal feature extraction")
    if points.shape[1] < 1:
        raise ValueError("at least one landmark is required")
    if points.shape[2] not in (2, 3):
        raise ValueError("landmarks must contain 2D or 3D coordinates")
    if not np.isfinite(points).all():
        raise ValueError("landmarks must contain only finite coordinates")
    return points


def _safe_mean_norm(values: NDArray[np.float64]) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(np.linalg.norm(values, axis=-1)))


def extract_movement_features(landmarks: ArrayLike) -> MovementFeatures:
    """Compute five approximate movement features from landmark tracks.

    Parameters
    ----------
    landmarks:
        Floating point array shaped (frames, landmarks, 2 or 3). Each frame
        must use the same landmark ordering and coordinate scale.

    Notes
    -----
    No sampling rate is supplied, so features are frame-domain proxies and
    should not be interpreted as physical frequency in Hz. For valid temporal
    analysis, capture at a stable frame rate and document that rate.
    """
    points = _validate_landmarks(landmarks)

    # Track the hand centroid to summarize global hand movement.
    centroid = points.mean(axis=1)
    velocity = np.diff(centroid, axis=0)
    acceleration = np.diff(velocity, axis=0)
    jerk = np.diff(acceleration, axis=0)

    # Approximate high-frequency residual after removing a local linear trend.
    # Unlike a moving average, a local linear fit is less likely to treat smooth,
    # fast, straight-line movement as tremor. This remains a frame-domain proxy,
    # affected by camera noise and frame rate; it is not a clinical measurement.
    window = min(7, points.shape[0] if points.shape[0] % 2 else points.shape[0] - 1)
    window = max(3, window)
    half_window = window // 2
    # Vectorize across landmarks and dimensions. Repeated np.polyfit calls for
    # every coordinate/frame were unnecessarily expensive in the live webcam loop.
    series_matrix = points.reshape(points.shape[0], -1)
    residuals = np.empty_like(series_matrix)
    for frame_idx in range(points.shape[0]):
        start = max(0, frame_idx - half_window)
        end = min(points.shape[0], frame_idx + half_window + 1)
        local_x = np.arange(start, end, dtype=np.float64) - frame_idx
        local_y = series_matrix[start:end]
        x_mean = float(np.mean(local_x))
        y_mean = np.mean(local_y, axis=0)
        centered_x = local_x - x_mean
        denominator = float(np.sum(centered_x * centered_x))
        if denominator > 0.0:
            slopes = (centered_x[:, None] * (local_y - y_mean)).sum(axis=0) / denominator
            predicted_at_frame = y_mean + slopes * (0.0 - x_mean)
        else:
            predicted_at_frame = y_mean
        residuals[frame_idx] = series_matrix[frame_idx] - predicted_at_frame
    tremor_proxy = float(np.sqrt(np.mean(np.square(residuals))))

    # Smoothness is bounded [0, 1]. Higher is smoother. This is a normalized
    # heuristic, not a standard clinical smoothness score.
    jerk_magnitude = _safe_mean_norm(jerk)
    velocity_scale = _safe_mean_norm(velocity) + 1e-8
    normalized_jerk = jerk_magnitude / (velocity_scale + 1e-8)
    trajectory_smoothness = float(1.0 / (1.0 + normalized_jerk))
    trajectory_smoothness = float(np.clip(trajectory_smoothness, 0.0, 1.0))

    # Landmark tracking jitters even when the hand is held still. Detect a
    # near-stationary centroid track and avoid amplifying tiny noise through
    # jerk/mean-speed ratios. Thresholds are provisional normalized-coordinate
    # noise floors, not calibrated physical motion thresholds.
    centroid_span = float(np.max(np.ptp(centroid[:, :2], axis=0)))
    mean_speed = float(np.mean(np.linalg.norm(velocity, axis=1)))
    near_stationary = centroid_span <= 0.008 and mean_speed <= 0.0015
    if near_stationary:
        trajectory_smoothness = 1.0

    # Similarity of consecutive centered hand poses. Centering removes global
    # translation so this primarily reflects relative landmark configuration.
    centered = points - points.mean(axis=1, keepdims=True)
    pose_vectors = centered.reshape(centered.shape[0], -1)
    pose_norms = np.linalg.norm(pose_vectors, axis=1, keepdims=True)
    normalized_poses = pose_vectors / np.maximum(pose_norms, 1e-8)
    similarities = np.sum(normalized_poses[1:] * normalized_poses[:-1], axis=1)
    gesture_consistency = float(np.clip((np.mean(similarities) + 1.0) / 2.0, 0.0, 1.0))

    # Coefficient of variation of frame-to-frame centroid speed; robust to a
    # near-zero mean so a stationary track doesn't produce division errors.
    speeds = np.linalg.norm(velocity, axis=1)
    mean_speed = float(np.mean(speeds))
    velocity_fluctuation = (
        0.0
        if near_stationary
        else (float(np.std(speeds) / mean_speed) if mean_speed > 1e-8 else 0.0)
    )
    velocity_fluctuation = float(np.clip(velocity_fluctuation, 0.0, 1e6))

    # Shannon entropy of movement direction bins. Low entropy can be a steady
    # direction; high entropy can represent varied movement. Neither is
    # inherently good/bad without a task-specific protocol.
    if not near_stationary and velocity.shape[0] and np.any(np.linalg.norm(velocity, axis=1) > 1e-8):
        moving = velocity[np.linalg.norm(velocity, axis=1) > 1e-8]
        angles = np.arctan2(moving[:, 1], moving[:, 0])
        bins = np.histogram(angles, bins=8, range=(-np.pi, np.pi))[0].astype(float)
        probabilities = bins[bins > 0] / bins.sum()
        entropy = -float(np.sum(probabilities * np.log(probabilities)))
        motion_entropy = float(entropy / np.log(8.0))
    else:
        motion_entropy = 0.0

    return MovementFeatures(
        tremor_proxy=tremor_proxy,
        trajectory_smoothness=trajectory_smoothness,
        gesture_consistency=gesture_consistency,
        velocity_fluctuation=velocity_fluctuation,
        motion_entropy=motion_entropy,
        frame_count=int(points.shape[0]),
    )


def experimental_stability_score(features: MovementFeatures) -> float:
    """Return an uncalibrated 0-1 movement-stability heuristic.

    A higher score means the observed trajectory looks more stable according
    to these selected heuristics. This score is not a learned model or a
    validated measure of fatigue, stress, or cognitive state.
    """
    values = (
        features.trajectory_smoothness,
        features.gesture_consistency,
        features.velocity_fluctuation,
        features.tremor_proxy,
    )
    if not all(isfinite(value) for value in values):
        raise ValueError("movement features must be finite")
    smoothness = np.clip(features.trajectory_smoothness, 0.0, 1.0)
    consistency = np.clip(features.gesture_consistency, 0.0, 1.0)
    fluctuation_penalty = 1.0 / (1.0 + max(0.0, features.velocity_fluctuation))
    tremor_penalty = 1.0 / (1.0 + max(0.0, features.tremor_proxy) * 10.0)

    score = (
        0.35 * smoothness
        + 0.30 * consistency
        + 0.20 * fluctuation_penalty
        + 0.15 * tremor_penalty
    )
    return float(np.clip(score, 0.0, 1.0))


def movement_stability_status(score: float) -> str:
    """Label the experimental movement score for display, not cognitive diagnosis.

    Thresholds mirror the demo safety gate and are provisional, not calibrated
    against independently labelled operator-stability data.
    """
    if not isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("score must be finite and between 0 and 1")
    if score < 0.60:
        return "LOW MOVEMENT STABILITY"
    if score < 0.72:
        return "UNCERTAIN MOVEMENT STABILITY"
    return "HIGHER MOVEMENT STABILITY"

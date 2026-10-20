"""Reference-pose gesture classification for the Part II webcam demo.

The rules are a starting point, not a trained recognizer. Validate and tune them
against the user's actual webcam poses before any hardware integration.
Commands remain simulation-only.
"""

from __future__ import annotations

from math import hypot
from typing import Sequence

import numpy as np


GESTURE_COMMANDS = {
    "LEFT": "SIMULATED_LEFT",
    "RIGHT": "SIMULATED_RIGHT",
    "FORWARD": "SIMULATED_FORWARD",
    "BACKWARD": "SIMULATED_BACKWARD",
    "LAND": "SIMULATED_LAND",
    "UPWARD": "SIMULATED_UPWARD",
    "STOP": "HOLD",
}


def command_for_gesture(gesture: str) -> str:
    """Return the simulation command for a recognized gesture label."""
    normalized = gesture.strip().upper()
    try:
        return GESTURE_COMMANDS[normalized]
    except KeyError as exc:
        raise ValueError(f"Unknown gesture label: {gesture!r}") from exc


def classify_hand_pose(landmarks: Sequence[Sequence[float]]) -> str:
    """Classify a 21-point MediaPipe hand pose into one of seven gesture labels.

    Landmark coordinates must be MediaPipe-normalized x/y/z values. Because
    palm/back orientation and the exact reference poses vary with camera angle,
    ambiguous poses return ``UNKNOWN`` rather than issuing a directional command.
    These geometric heuristics must be calibrated on the target webcam.
    """
    points = np.asarray(landmarks, dtype=float)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError("landmarks must be a finite array with shape (21, 3)")

    wrist = points[0, :2]
    mcp = points[[5, 9, 13, 17], :2]
    tips = points[[8, 12, 16, 20], :2]
    pips = points[[6, 10, 14, 18], :2]
    palm_width = float(np.linalg.norm(points[5, :2] - points[17, :2]))
    palm_length = float(np.linalg.norm(points[9, :2] - wrist))
    if palm_width < 1e-5 or palm_length < 1e-5:
        return "UNKNOWN"

    # In image coordinates, a finger is extended when its tip is above its PIP.
    # A small margin reduces false positives from bent fingers.
    extended = tips[:, 1] < pips[:, 1] - 0.015
    extended_count = int(extended.sum())
    thumb_tip = points[4, :2]
    thumb_mcp = points[2, :2]
    index_mcp = points[5, :2]
    thumb_extension = float(np.linalg.norm(thumb_tip - index_mcp)) / palm_width

    if extended_count == 0 and thumb_extension < 0.65:
        return "STOP"

    if extended_count < 3:
        return "UNKNOWN"

    # A widely opened hand is the UPWARD pose in the supplied references.
    fingertip_spread = float(np.ptp(tips[:, 0])) / palm_width
    if extended_count == 4 and fingertip_spread >= 1.35:
        return "UPWARD"

    # A foreshortened palm with extended, tightly grouped fingers is the edge-on LAND pose.
    palm_width_ratio = palm_width / palm_length
    if extended_count == 4 and palm_width_ratio < 0.52 and fingertip_spread < 0.8:
        return "LAND"

    # The first reference has a sideways row of fingers and a raised thumb.
    finger_axis = tips[1] - tips[0]
    horizontal_fingers = abs(float(finger_axis[0])) > 1.4 * abs(float(finger_axis[1]))
    if extended_count == 4 and horizontal_fingers and thumb_extension >= 0.65:
        return "LEFT"

    # For the remaining compact-finger poses, use wrist-to-middle-finger angle.
    # Near-vertical poses are separated by MediaPipe depth ordering; if that signal
    # is weak, return UNKNOWN rather than guessing between forward/backward.
    if extended_count == 4 and fingertip_spread < 0.9:
        middle_direction = points[12] - points[0]
        near_vertical = abs(float(middle_direction[1])) >= 0.8 * abs(float(middle_direction[0]))
        if near_vertical:
            depth_delta = float(np.mean(points[[8, 12, 16, 20], 2] - points[[6, 10, 14, 18], 2]))
            if depth_delta < -0.025:
                return "FORWARD"
            if depth_delta > 0.025:
                return "BACKWARD"
        else:
            # Compact fingers angled sideways with the palm facing the camera.
            # This is a provisional heuristic and should be calibrated from live frames.
            if thumb_extension < 0.65:
                return "RIGHT"

    return "UNKNOWN"

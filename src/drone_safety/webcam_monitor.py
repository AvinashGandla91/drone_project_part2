"""Independent webcam demo for Part II (no Part I repository required).

Displays experimental movement features and optional movement-protocol prediction.
With --commands, reference hand poses map to simulated commands only.
No command is sent to a physical drone.
"""

from __future__ import annotations

import argparse
from collections import deque
from math import isfinite
from pathlib import Path

import numpy as np

from .decision_fusion import validate_command
from .gesture_commands import classify_hand_pose, command_for_gesture
from .features import (
    experimental_stability_score,
    extract_movement_features,
    movement_stability_status,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--window", type=int, default=20, help="Frames per analysis window (smaller windows respond faster)")
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model file, e.g. models/movement_protocol_model.joblib",
    )
    parser.add_argument(
        "--commands",
        action="store_true",
        help="Show simulated directional commands passed through the safety gate",
    )
    parser.add_argument(
        "--demo-gesture-confidence",
        type=float,
        default=0.90,
        help=(
            "Placeholder gesture confidence for standalone testing only; "
            "replace with Part I model output during integration (default: 0.90)"
        ),
    )
    args = parser.parse_args()

    if args.window < 8:
        parser.error("--window must be at least 8")
    if not isfinite(args.demo_gesture_confidence) or not 0.0 <= args.demo_gesture_confidence <= 1.0:
        parser.error("--demo-gesture-confidence must be finite and between 0 and 1")

    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise SystemExit(
            "Webcam dependencies are missing. Install them with: "
            'python -m pip install -e ".[vision]"'
        ) from exc

    classifier = None
    feature_columns = None
    if args.model:
        model_path = Path(args.model)
        if not model_path.is_file():
            raise SystemExit(f"Model file not found: {model_path}")
        try:
            import joblib
        except ImportError as exc:
            raise SystemExit(
                'Model loading requires joblib. Install it with: '
                'python -m pip install -e ".[ml]"'
            ) from exc
        bundle = joblib.load(model_path)
        classifier = bundle.get("model")
        feature_columns = bundle.get("feature_columns")
        if classifier is None or not feature_columns:
            raise SystemExit(
                f"Model file {model_path} must contain 'model' and 'feature_columns'."
            )
        print(f"Loaded optional movement-protocol classifier: {model_path}")
        print("Its labels describe movement style only, not fatigue or cognitive state.")

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise SystemExit(
            f"Could not open camera index {args.camera}. Close apps using the camera "
            "or try --camera 1."
        )

    mp_hands = mp.solutions.hands
    history: deque[np.ndarray] = deque(maxlen=args.window)
    # Two-frame confirmation filters isolated noisy detections while keeping commands responsive.
    direction_votes: deque[str] = deque(maxlen=2)
    last_detected_command = "UNKNOWN"
    last_features = None
    last_score = None
    last_prediction = None
    last_command = "HOLD"
    last_gate_result = "HOLD"
    last_gate_reason = "No directional movement detected."

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        ) as hands:
            while True:
                ok, frame = capture.read()
                if not ok:
                    print("Camera frame could not be read; stopping.")
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                if result.multi_hand_landmarks:
                    hand = result.multi_hand_landmarks[0]
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand, mp_hands.HAND_CONNECTIONS
                    )
                    points = np.array(
                        [[lm.x, lm.y, lm.z] for lm in hand.landmark],
                        dtype=np.float64,
                    )
                    history.append(points)
                    if args.commands:
                        raw_command = classify_hand_pose(points)
                        # Closed fist is an immediate stop; other poses require two matching frames.
                        if raw_command == "STOP":
                            direction_votes.clear()
                            last_detected_command = "STOP"
                            last_command = "HOLD"
                            last_gate_result = "STOP"
                            last_gate_reason = "Closed fist detected; directional command stopped."
                        elif raw_command == "UNKNOWN":
                            direction_votes.clear()
                            last_detected_command = "UNKNOWN"
                            last_command = "HOLD"
                        else:
                            if direction_votes and raw_command != direction_votes[-1]:
                                direction_votes.clear()
                            direction_votes.append(raw_command)
                            last_detected_command = (
                                raw_command
                                if len(direction_votes) == 2 and direction_votes[-2] == raw_command
                                else "UNKNOWN"
                            )
                            last_command = "HOLD"  # Wait for the stability gate before showing a command.

                    if len(history) == args.window:
                        last_features = extract_movement_features(np.stack(history))
                        last_score = experimental_stability_score(last_features)
                        if args.commands:
                            if last_detected_command in {"UNKNOWN", "STOP"}:
                                if last_detected_command == "UNKNOWN":
                                    last_gate_result = "HOLD"
                                    last_gate_reason = "No confident reference-pose gesture detected."
                                last_command = "HOLD"
                            else:
                                decision = validate_command(
                                    command_for_gesture(last_detected_command),
                                    args.demo_gesture_confidence,
                                    last_score,
                                )
                                last_gate_result = decision.result.value.upper()
                                last_gate_reason = decision.reason
                                last_command = (
                                    decision.simulated_command
                                    if decision.simulated_command is not None
                                    else "HOLD"
                                )
                        if classifier is not None:
                            row = last_features.to_dict()
                            missing = [name for name in feature_columns if name not in row]
                            if missing:
                                raise SystemExit(
                                    "Model expects unsupported features: " + ", ".join(missing)
                                )
                            import pandas as pd

                            sample = pd.DataFrame(
                                [{name: row[name] for name in feature_columns}],
                                columns=feature_columns,
                            )
                            last_prediction = str(classifier.predict(sample)[0])
                else:
                    # A gap ends the current track; do not mix old and new hand windows.
                    history.clear()
                    direction_votes.clear()
                    if args.commands:
                        last_command = "HOLD"
                        last_detected_command = "UNKNOWN"
                        last_gate_result = "HOLD"
                        last_gate_reason = "No hand detected."

                cv2.putText(
                    frame, "Part II: reference-pose gesture + movement analysis",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (255, 255, 255), 2, cv2.LINE_AA,
                )
                if not result.multi_hand_landmarks:
                    cv2.putText(
                        frame, "No hand detected - show one hand to camera",
                        (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 2, cv2.LINE_AA,
                    )
                elif last_score is None:
                    cv2.putText(
                        frame, f"Collecting frames: {len(history)}/{args.window}",
                        (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 2, cv2.LINE_AA,
                    )
                else:
                    cv2.putText(
                        frame, f"Experimental movement score: {last_score:.2f}",
                        (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 2, cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        f"Smoothness {last_features.trajectory_smoothness:.2f} | "
                        f"Pose consistency {last_features.gesture_consistency:.2f}",
                        (12, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        (255, 255, 255), 1, cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        f"Tremor proxy {last_features.tremor_proxy:.4f} | "
                        f"Velocity variation {last_features.velocity_fluctuation:.2f}",
                        (12, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        f"Movement status (experimental): {movement_stability_status(last_score)}",
                        (12, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (255, 255, 255), 1, cv2.LINE_AA,
                    )
                    if classifier is not None:
                        cv2.putText(
                            frame, f"Movement protocol prediction: {last_prediction}",
                            (12, 164), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                            (255, 255, 255), 1, cv2.LINE_AA,
                        )

                if args.commands:
                    cv2.putText(
                        frame, f"DETECTED: {last_detected_command}",
                        (12, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        (0, 255, 255), 2, cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame, f"SAFETY: {last_gate_result} | COMMAND: {last_command}",
                        (12, 216), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                        (0, 255, 255), 2, cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame, f"Demo gesture confidence: {args.demo_gesture_confidence:.2f}",
                        (12, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (255, 255, 255), 1, cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame, last_gate_reason[:95],
                        (12, 262), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                        (255, 255, 255), 1, cv2.LINE_AA,
                    )

                cv2.putText(
                    frame, "Simulation only; no physical drone commands. Q/ESC exits.",
                    (12, frame.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA,
                )
                cv2.imshow("Drone Project II - Stability Monitor", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

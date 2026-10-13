"""Independent webcam demo for Part II (no Part I repository required).

Displays experimental movement features and optional movement-protocol prediction.
With --commands, hand motion is mapped to simulated directional commands only.
No command is sent to a physical drone.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np

from .decision_fusion import validate_command
from .features import experimental_stability_score, extract_movement_features


def _direction_command(track: deque[np.ndarray]) -> str:
    """Map palm-centre displacement over a short window to a simulated command."""
    if len(track) < 4:
        return "HOLD"
    start = track[0]
    end = track[-1]
    delta = end - start
    # MediaPipe image coordinates: x grows right, y grows down. Webcam is mirrored.
    if abs(delta[0]) < 0.025 and abs(delta[1]) < 0.025:
        return "HOLD"
    if abs(delta[0]) >= abs(delta[1]):
        return "MOVE_RIGHT" if delta[0] > 0 else "MOVE_LEFT"
    return "MOVE_DOWN" if delta[1] > 0 else "MOVE_UP"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--window", type=int, default=30, help="Frames per analysis window")
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
    if not 0.0 <= args.demo_gesture_confidence <= 1.0:
        parser.error("--demo-gesture-confidence must be between 0 and 1")

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
    palm_track: deque[np.ndarray] = deque(maxlen=5)
    # Short voting window prevents one noisy frame from flickering the command.
    direction_votes: deque[str] = deque(maxlen=3)
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
                    # Palm centre from wrist and four MCP joints, less sensitive to finger pose.
                    palm_track.append(points[[0, 5, 9, 13, 17], :].mean(axis=0))
                    if args.commands:
                        raw_command = _direction_command(palm_track)
                        direction_votes.append(raw_command)
                        # Require repeated directional evidence; HOLD wins when
                        # there is not enough consistent motion.
                        directional_votes = [
                            command for command in direction_votes if command != "HOLD"
                        ]
                        if len(directional_votes) >= 2:
                            counts = {
                                command: directional_votes.count(command)
                                for command in set(directional_votes)
                            }
                            winner = max(counts, key=counts.get)
                            last_command = (
                                winner if counts[winner] >= 2 else "HOLD"
                            )
                        else:
                            last_command = "HOLD"

                    if len(history) == args.window:
                        last_features = extract_movement_features(np.stack(history))
                        last_score = experimental_stability_score(last_features)
                        if args.commands:
                            if last_command == "HOLD":
                                last_gate_result = "HOLD"
                                last_gate_reason = "No directional movement detected."
                            else:
                                decision = validate_command(
                                    last_command,
                                    args.demo_gesture_confidence,
                                    last_score,
                                )
                                last_gate_result = decision.result.value.upper()
                                last_gate_reason = decision.reason
                                if decision.simulated_command is None:
                                    last_command = "HOLD"
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
                    # Avoid interpreting a hand reappearing elsewhere as continuous motion.
                    palm_track.clear()
                    direction_votes.clear()
                    if args.commands:
                        last_command = "HOLD"
                        last_gate_result = "HOLD"
                        last_gate_reason = "No hand detected."

                cv2.putText(
                    frame, "Part II: experimental movement analysis",
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
                        f"Consistency {last_features.gesture_consistency:.2f}",
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
                    if classifier is not None:
                        cv2.putText(
                            frame, f"Movement protocol prediction: {last_prediction}",
                            (12, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                            (255, 255, 255), 1, cv2.LINE_AA,
                        )

                if args.commands:
                    cv2.putText(
                        frame, f"SAFETY GATE: {last_gate_result} | COMMAND: {last_command}",
                        (12, 174), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        (255, 255, 0), 2, cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame, f"Demo gesture confidence: {args.demo_gesture_confidence:.2f}",
                        (12, 202), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (255, 255, 255), 1, cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame, last_gate_reason[:95],
                        (12, 226), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
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

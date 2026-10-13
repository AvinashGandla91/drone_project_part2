"""Independent webcam demo for Part II (no Part I repository required).

This tool observes MediaPipe hand landmarks and displays an experimental
movement-stability proxy. An optional saved model can classify the recording
protocol as controlled/varied. Neither output diagnoses fatigue or stress,
and this tool does not control a physical drone.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np

from .features import experimental_stability_score, extract_movement_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--window", type=int, default=30, help="Frames per analysis window")
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model file, e.g. models/movement_protocol_model.joblib",
    )
    args = parser.parse_args()

    if args.window < 8:
        parser.error("--window must be at least 8")

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
    last_features = None
    last_score = None
    last_prediction = None

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

                    if len(history) == args.window:
                        last_features = extract_movement_features(np.stack(history))
                        last_score = experimental_stability_score(last_features)
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

                cv2.putText(
                    frame,
                    "Part II: experimental movement analysis",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                if not result.multi_hand_landmarks:
                    status = "No hand detected - show one hand to camera"
                    cv2.putText(
                        frame, status, (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
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

                cv2.putText(
                    frame, "Experimental only; not a diagnosis. Press Q or ESC to quit.",
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

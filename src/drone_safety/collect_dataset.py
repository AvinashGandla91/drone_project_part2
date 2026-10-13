"""Collect labelled movement-feature windows for offline model experiments.

Labels describe the recording protocol (e.g. controlled vs intentionally varied
movement), not a person's cognitive state. Obtain consent before recording.
"""

from __future__ import annotations

import argparse
import csv
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .features import extract_movement_features

FIELDNAMES = [
    "label",
    "tremor_proxy",
    "trajectory_smoothness",
    "gesture_consistency",
    "velocity_fluctuation",
    "motion_entropy",
    "frame_count",
    "recorded_at_utc",
    "session_id",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="data/movement_features_sessions.csv",
        help="Output CSV with a unique session_id for leakage-aware evaluation",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--window", type=int, default=30)
    args = parser.parse_args()
    if args.window < 8:
        parser.error("--window must be at least 8")

    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise SystemExit(
            'Install webcam dependencies first: python -m pip install -e ".[vision]"'
        ) from exc

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output.exists() and output.stat().st_size > 0
    session_id = uuid.uuid4().hex[:12]
    print(f"Recording session ID: {session_id}")

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    history: deque[np.ndarray] = deque(maxlen=args.window)
    mp_hands = mp.solutions.hands
    label = "controlled"
    last_saved = 0
    try:
        with output.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
            if not file_exists:
                writer.writeheader()

            with mp_hands.Hands(
                max_num_hands=1,
                model_complexity=0,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
            ) as hands:
                while True:
                    ok, frame = camera.read()
                    if not ok:
                        break
                    frame = cv2.flip(frame, 1)
                    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                    if result.multi_hand_landmarks:
                        points = np.array(
                            [[lm.x, lm.y, lm.z]
                             for lm in result.multi_hand_landmarks[0].landmark],
                            dtype=np.float64,
                        )
                        history.append(points)
                        mp.solutions.drawing_utils.draw_landmarks(
                            frame, result.multi_hand_landmarks[0],
                            mp_hands.HAND_CONNECTIONS,
                        )

                    cv2.putText(
                        frame, f"Current label: {label} | 1=controlled 2=varied",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (255, 255, 255), 2, cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame, "SPACE=save window | Q/ESC=quit",
                        (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2, cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame, f"Frames: {len(history)}/{args.window} | Saved: {last_saved}",
                        (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2, cv2.LINE_AA,
                    )
                    cv2.imshow("Movement Feature Dataset Collector", frame)
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("1"):
                        label = "controlled"
                    elif key == ord("2"):
                        label = "varied"
                    elif key == ord(" ") and len(history) == args.window:
                        features = extract_movement_features(np.stack(history)).to_dict()
                        writer.writerow({
                            "label": label,
                            **features,
                            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                            "session_id": session_id,
                        })
                        csv_file.flush()
                        last_saved += 1
                    elif key in (ord("q"), 27):
                        break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"Saved {last_saved} feature windows to {output}")
    print(f"Session ID: {session_id}")
    print("These labels describe movement protocol only; do not interpret them as fatigue/stress.")


if __name__ == "__main__":
    main()

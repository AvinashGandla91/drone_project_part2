"""Collect labelled MediaPipe hand-pose samples for the seven command gestures.

Labels follow the reference photo order: LEFT, RIGHT, FORWARD, BACKWARD, LAND,
UPWARD, STOP (closed fist). This records numeric landmarks only, not video.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .gesture_commands import GESTURE_LABELS, normalize_landmarks

FIELDNAMES = ["label", *[f"f{i}" for i in range(63)]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/gesture_landmarks.csv")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise SystemExit('Install webcam dependencies: python -m pip install -e ".[vision]"') from exc

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    exists = output.exists() and output.stat().st_size > 0
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    label_index = 0
    saved = 0
    last_vector = None
    try:
        with output.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
            if not exists:
                writer.writeheader()
            with mp.solutions.hands.Hands(
                max_num_hands=1, model_complexity=0,
                min_detection_confidence=0.6, min_tracking_confidence=0.6,
            ) as hands:
                while True:
                    ok, frame = camera.read()
                    if not ok:
                        break
                    frame = cv2.flip(frame, 1)
                    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    last_vector = None
                    if result.multi_hand_landmarks:
                        hand = result.multi_hand_landmarks[0]
                        mp.solutions.drawing_utils.draw_landmarks(
                            frame, hand, mp.solutions.hands.HAND_CONNECTIONS
                        )
                        points = np.array([[p.x, p.y, p.z] for p in hand.landmark])
                        try:
                            last_vector = normalize_landmarks(points)
                        except ValueError:
                            last_vector = None
                    label = GESTURE_LABELS[label_index]
                    cv2.putText(frame, f"Label: {label} ({label_index + 1}/7)",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    cv2.putText(frame, "1-7 select label | SPACE save sample | Q quit",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1)
                    cv2.putText(frame, f"Saved this session: {saved}",
                                (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1)
                    cv2.imshow("Gesture Dataset Collector", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if ord("1") <= key <= ord("7"):
                        label_index = key - ord("1")
                    elif key == ord(" ") and last_vector is not None:
                        writer.writerow({"label": label, **{
                            f"f{i}": float(value) for i, value in enumerate(last_vector)
                        }})
                        stream.flush()
                        saved += 1
                    elif key in (ord("q"), 27):
                        break
    finally:
        camera.release()
        cv2.destroyAllWindows()
    print(f"Saved {saved} labelled samples to {output}")
    print("Collect varied angles/distances and balanced samples for every label.")


if __name__ == "__main__":
    main()

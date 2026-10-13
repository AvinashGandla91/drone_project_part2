"""Run a small synthetic demonstration; this is not a performance evaluation."""

from __future__ import annotations

import numpy as np

from .decision_fusion import validate_command
from .features import experimental_stability_score, extract_movement_features


def main() -> None:
    rng = np.random.default_rng(7)
    frame_count, landmark_count = 40, 21
    base_hand = rng.normal(0.0, 0.03, size=(landmark_count, 2))

    # Synthetic stable-ish sequence for code-path demonstration only.
    stable_track = np.stack(
        [
            base_hand + np.array([frame * 0.001, 0.0])
            + rng.normal(0.0, 0.0005, size=(landmark_count, 2))
            for frame in range(frame_count)
        ]
    )
    features = extract_movement_features(stable_track)
    score = experimental_stability_score(features)
    decision = validate_command("MOVE_FORWARD", 0.93, score)

    print("Synthetic demo only — not a real-world model evaluation.")
    print("Movement features:")
    for key, value in features.to_dict().items():
        print(f"  {key}: {value}")
    print(f"Experimental movement-stability score: {score:.3f}")
    print(f"Simulated command decision: {decision.result.value}")
    print(f"Reason: {decision.reason}")
    print(f"Simulated command: {decision.simulated_command}")


if __name__ == "__main__":
    main()

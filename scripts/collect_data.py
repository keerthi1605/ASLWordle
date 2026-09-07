"""
Record your own ASL alphabet training data using your webcam.

Why this exists instead of a downloaded dataset: a landmark-level ASL
dataset (already-extracted 21-point hand coordinates, not raw photos)
isn't something we could verify the license/source of, and this
project's rules say never fabricate a dataset. So instead: you record
a small number of real samples per letter yourself, and
scripts/train_asl.py trains on exactly that. See
docs/learning/06_asl_classifier.md for the honest tradeoffs of a
small, self-collected dataset.

Run:
    venv/Scripts/python.exe scripts/collect_data.py

Controls (a plain OpenCV window, not part of the Streamlit app):
    SPACE  capture one sample of the current letter
    N      move on to the next letter early
    ESC    stop entirely (samples already saved are kept)

For each letter, hold the sign steady, glance at the on-screen sample
count, and press SPACE repeatedly. IMPORTANT for recognition quality:
move your hand slightly between captures (angle, distance, position
in frame) rather than freezing in one exact spot -- see "why this
helps robustness" in docs/learning/06_asl_classifier.md. Aim for at
least ~30 samples per letter; more (and more varied) is better.

Samples are appended to data/asl_landmarks.csv (63 feature columns +
a label column) -- safe to re-run this later to add more samples
without losing earlier sessions.
"""

import csv
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ASL_LABELS, ASL_LANDMARKS_CSV_PATH, DATA_DIR  # noqa: E402
from src.vision.camera import Camera  # noqa: E402
from src.vision.hand_detector import HandDetector, draw_hand_landmarks  # noqa: E402
from src.vision.landmark_utils import NUM_FEATURES, landmarks_to_feature_vector  # noqa: E402

SAMPLES_PER_LETTER_TARGET = 30
WINDOW_NAME = "SignWordle - Data Collection"


def _open_csv_for_append():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    is_new_file = not ASL_LANDMARKS_CSV_PATH.exists()
    csv_file = open(ASL_LANDMARKS_CSV_PATH, "a", newline="")
    writer = csv.writer(csv_file)
    if is_new_file:
        writer.writerow([f"f{i}" for i in range(NUM_FEATURES)] + ["label"])
    return csv_file, writer


def _draw_overlay(frame_bgr, letter: str, sample_count: int, hand_detected: bool) -> None:
    cv2.putText(
        frame_bgr,
        f"Letter: {letter}   Samples: {sample_count}/{SAMPLES_PER_LETTER_TARGET}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
    )
    cv2.putText(
        frame_bgr, "SPACE = capture   N = next letter   ESC = quit",
        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1,
    )
    if not hand_detected:
        cv2.putText(
            frame_bgr, "No hand detected", (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
        )


def main() -> None:
    csv_file, writer = _open_csv_for_append()

    camera = Camera(mirror=True)
    ok, message = camera.start()
    if not ok:
        print(f"Could not start camera: {message}")
        csv_file.close()
        return

    print("Loading hand-landmark model...")
    detector = HandDetector()

    print("=" * 64)
    print("ASL DATA COLLECTION")
    print(f"{len(ASL_LABELS)} letters to record: {' '.join(ASL_LABELS)}")
    print("SPACE = capture a sample | N = next letter | ESC = quit")
    print(f"Saving to: {ASL_LANDMARKS_CSV_PATH}")
    print("=" * 64)

    total_captured = 0
    try:
        for letter in ASL_LABELS:
            sample_count = 0
            print(f"\n>>> Letter '{letter}' -- target {SAMPLES_PER_LETTER_TARGET} samples")
            while sample_count < SAMPLES_PER_LETTER_TARGET:
                ok, frame, read_message = camera.read_frame()
                if not ok:
                    print(f"Camera read failed ({read_message}), stopping.")
                    return

                hands, _handedness = detector.detect(frame)
                draw_hand_landmarks(frame, hands)

                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                _draw_overlay(frame_bgr, letter, sample_count, hand_detected=bool(hands))
                cv2.imshow(WINDOW_NAME, frame_bgr)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    raise KeyboardInterrupt
                if key == ord("n"):
                    print(f"Skipping ahead ({sample_count} samples collected for '{letter}').")
                    break
                if key == 32 and hands:  # SPACE, only counts if a hand is visible
                    vector = landmarks_to_feature_vector(hands[0])
                    writer.writerow(list(vector) + [letter])
                    csv_file.flush()
                    sample_count += 1
                    total_captured += 1
                    print(f"  captured {sample_count}/{SAMPLES_PER_LETTER_TARGET}")
    except KeyboardInterrupt:
        print("\nStopped early.")
    finally:
        camera.stop()
        detector.close()
        csv_file.close()
        cv2.destroyAllWindows()
        print(f"\nDone. {total_captured} new samples saved to {ASL_LANDMARKS_CSV_PATH}")
        print("Next: venv/Scripts/python.exe scripts/train_asl.py")


if __name__ == "__main__":
    main()

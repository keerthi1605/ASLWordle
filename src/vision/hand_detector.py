"""
Wraps MediaPipe's HandLandmarker: give it an RGB frame, get back each
detected hand's 21 landmarks (+ which hand it is, if known).

Knows nothing about ASL letters, feature vectors, or Wordle — see
docs/learning/01_project_architecture.md. Feature engineering (Phase 5)
and ASL classification (Phase 6) both build on top of this module's
output; they never import MediaPipe themselves.
"""

import os

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from config import HAND_LANDMARKER_MODEL_PATH
from src.vision.landmark_utils import HAND_CONNECTIONS, to_pixel_coords

# Drawing colors (RGB, since frames in this app are RGB — see
# docs/learning/02_opencv.md on BGR vs RGB).
_LANDMARK_COLOR = (255, 80, 80)     # red-ish dots for each of the 21 points
_CONNECTION_COLOR = (80, 220, 80)   # green "bones" between them


class HandDetector:
    """Detects hands in a frame using MediaPipe's Hand Landmarker task."""

    def __init__(
        self,
        model_path=HAND_LANDMARKER_MODEL_PATH,
        num_hands: int = 1,
        min_detection_confidence: float = 0.5,
    ):
        if not str(model_path) or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Hand landmark model not found at {model_path}. "
                "See docs/learning/03_mediapipe.md for how to download it."
            )
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame_rgb):
        """
        Run detection on one RGB frame.

        Returns (hands, handedness):
          hands: list of hands; each hand is a list of 21 (x, y, z)
                 tuples in NORMALIZED coordinates (0-1).
          handedness: matching list of "Left"/"Right"/"Unknown" strings.
        Returns ([], []) if no hand is detected — never raises for that.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect(mp_image)

        hands = [
            [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
            for hand_landmarks in result.hand_landmarks
        ]
        handedness = [
            (categories[0].category_name if categories else "Unknown")
            for categories in result.handedness
        ]
        return hands, handedness

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def draw_hand_landmarks(frame_rgb, hands) -> None:
    """Draw the 21-point skeleton for each detected hand directly onto
    `frame_rgb` (mutated in place). Plain `cv2.circle`/`cv2.line` calls
    instead of MediaPipe's own drawing helpers — easier to explain in
    a viva, and not tied to a specific MediaPipe version's drawing API.
    """
    height, width = frame_rgb.shape[:2]
    for landmarks in hands:
        points = to_pixel_coords(landmarks, width, height)
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame_rgb, points[start_idx], points[end_idx], _CONNECTION_COLOR, 2)
        for x, y in points:
            cv2.circle(frame_rgb, (x, y), 4, _LANDMARK_COLOR, -1)

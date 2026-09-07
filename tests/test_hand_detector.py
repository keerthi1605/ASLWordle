"""
Tests for src/vision/hand_detector.py.

Detection needs the downloaded hand_landmarker.task model, and the
"real hand in frame" tests need an actual webcam — both SKIP (not
fail) when unavailable, same philosophy as tests/test_camera.py: a
missing model/camera is an environment fact, not a code bug.

    venv/Scripts/python.exe -m unittest tests.test_hand_detector -v
"""

import os
import unittest

import numpy as np

from config import HAND_LANDMARKER_MODEL_PATH
from src.vision.camera import Camera
from src.vision.hand_detector import HandDetector, draw_hand_landmarks

_HAS_MODEL = os.path.exists(HAND_LANDMARKER_MODEL_PATH)


def _camera_available() -> bool:
    cam = Camera()
    ok, _ = cam.start()
    cam.stop()
    return ok


_HAS_CAMERA = _camera_available()


@unittest.skipUnless(_HAS_MODEL, "hand_landmarker.task model not downloaded — see docs/learning/03_mediapipe.md")
class TestHandDetectorWithModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = HandDetector()

    @classmethod
    def tearDownClass(cls):
        cls.detector.close()

    def test_blank_frame_detects_no_hands(self):
        # A solid black frame has no hand in it -- detection should
        # come back empty, not crash or hallucinate a hand.
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        hands, handedness = self.detector.detect(blank)
        self.assertEqual(hands, [])
        self.assertEqual(handedness, [])

    def test_detect_returns_correct_types(self):
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        hands, handedness = self.detector.detect(blank)
        self.assertIsInstance(hands, list)
        self.assertIsInstance(handedness, list)

    def test_draw_on_frame_with_no_hands_does_not_crash(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_hand_landmarks(frame, hands=[])  # should be a no-op

    def test_draw_with_fake_hand_draws_something(self):
        # A synthetic "hand" (21 landmarks spread across the frame) --
        # doesn't need to look like a real hand, just exercises the
        # drawing code path end-to-end.
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        fake_hand = [(i / 21, i / 21, 0.0) for i in range(21)]
        self.assertEqual(frame.sum(), 0)  # starts all-black
        draw_hand_landmarks(frame, hands=[fake_hand])
        self.assertGreater(frame.sum(), 0)  # something got drawn

    @unittest.skipUnless(_HAS_CAMERA, "No webcam detected — skipping live-frame test.")
    def test_detect_on_real_camera_frame_does_not_crash(self):
        # This machine's webcam usually isn't pointed at a hand, so we
        # only assert it runs cleanly and returns well-typed output --
        # NOT that a hand is found. Manually verified separately: see
        # docs/learning/04_hand_landmarks.md.
        with Camera() as cam:
            ok, frame, _ = cam.read_frame()
        self.assertTrue(ok)
        hands, handedness = self.detector.detect(frame)
        self.assertIsInstance(hands, list)
        self.assertEqual(len(hands), len(handedness))
        for hand in hands:
            self.assertEqual(len(hand), 21)


class TestHandDetectorErrorHandling(unittest.TestCase):
    def test_missing_model_file_raises_clear_error(self):
        with self.assertRaises(FileNotFoundError):
            HandDetector(model_path="models/does_not_exist.task")


if __name__ == "__main__":
    unittest.main()

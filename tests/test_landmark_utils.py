"""
Tests for src/vision/landmark_utils.py — pure math, no hardware needed.

    venv/Scripts/python.exe -m unittest tests.test_landmark_utils -v
"""

import unittest

import numpy as np

from src.vision.landmark_utils import (
    HAND_CONNECTIONS,
    MIDDLE_MCP_INDEX,
    NUM_FEATURES,
    NUM_LANDMARKS,
    WRIST_INDEX,
    landmarks_to_feature_vector,
    normalize_landmarks,
    to_pixel_coords,
)


def _make_fake_hand(seed: int = 0):
    """A reproducible, hand-shaped-ish set of 21 (x, y, z) landmarks
    for testing -- doesn't need to look like a real hand, just needs
    21 distinct, non-degenerate points."""
    rng = np.random.default_rng(seed)
    return [tuple(rng.uniform(0.1, 0.9, size=3)) for _ in range(NUM_LANDMARKS)]


class TestToPixelCoords(unittest.TestCase):
    def test_converts_normalized_to_pixels(self):
        landmarks = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.1), (1.0, 1.0, -0.1)]
        result = to_pixel_coords(landmarks, image_width=640, image_height=480)
        self.assertEqual(result, [(0, 0), (320, 240), (640, 480)])

    def test_drops_z_coordinate(self):
        landmarks = [(0.25, 0.25, 99.0)]
        result = to_pixel_coords(landmarks, image_width=100, image_height=100)
        self.assertEqual(result, [(25, 25)])
        self.assertEqual(len(result[0]), 2)  # (x, y) only

    def test_empty_input_gives_empty_output(self):
        self.assertEqual(to_pixel_coords([], 640, 480), [])


class TestHandTopology(unittest.TestCase):
    def test_num_landmarks_is_21(self):
        self.assertEqual(NUM_LANDMARKS, 21)

    def test_all_connection_indices_are_valid(self):
        for start, end in HAND_CONNECTIONS:
            self.assertTrue(0 <= start < NUM_LANDMARKS)
            self.assertTrue(0 <= end < NUM_LANDMARKS)

    def test_no_self_loops(self):
        for start, end in HAND_CONNECTIONS:
            self.assertNotEqual(start, end)

    def test_wrist_connects_to_each_finger_base(self):
        # Landmark 0 is the wrist; each finger's base (1, 5, 17) or an
        # adjacent joint should trace back to it through the topology.
        starts_and_ends = set(HAND_CONNECTIONS)
        self.assertIn((0, 1), starts_and_ends)   # thumb base
        self.assertIn((0, 5), starts_and_ends)   # index base
        self.assertIn((0, 17), starts_and_ends)  # pinky/palm base


class TestNormalizeLandmarks(unittest.TestCase):
    def test_output_shape_is_21_by_3(self):
        result = normalize_landmarks(_make_fake_hand())
        self.assertEqual(result.shape, (NUM_LANDMARKS, 3))

    def test_wrist_maps_to_origin(self):
        # Translation invariance, directly: after normalizing, the
        # wrist itself (landmark 0) must sit exactly at (0, 0, 0).
        result = normalize_landmarks(_make_fake_hand())
        np.testing.assert_allclose(result[WRIST_INDEX], [0.0, 0.0, 0.0], atol=1e-9)

    def test_translation_invariance(self):
        # Sliding the whole hand around the frame shouldn't change the
        # normalized result at all -- only where the hand IS in the
        # frame changed, not its shape.
        hand = _make_fake_hand()
        shifted = [(x + 5.0, y - 3.0, z + 1.0) for x, y, z in hand]
        np.testing.assert_allclose(
            normalize_landmarks(hand), normalize_landmarks(shifted), atol=1e-9
        )

    def test_scale_invariance(self):
        # A "bigger" version of the same hand shape (e.g. closer to
        # the camera), scaled outward from the wrist, must normalize
        # to the same result as the original.
        hand = _make_fake_hand()
        wrist = hand[WRIST_INDEX]
        bigger = [
            tuple(w + (p - w) * 3.0 for p, w in zip(point, wrist)) for point in hand
        ]
        np.testing.assert_allclose(
            normalize_landmarks(hand), normalize_landmarks(bigger), atol=1e-9
        )

    def test_degenerate_input_does_not_divide_by_zero(self):
        # Every landmark at the exact same point -> the scale
        # reference distance is 0. Must not raise or produce NaN/Inf.
        same_point = [(0.5, 0.5, 0.5)] * NUM_LANDMARKS
        result = normalize_landmarks(same_point)
        self.assertTrue(np.all(np.isfinite(result)))


class TestLandmarksToFeatureVector(unittest.TestCase):
    def test_output_is_63_values(self):
        vector = landmarks_to_feature_vector(_make_fake_hand())
        self.assertEqual(vector.shape, (NUM_FEATURES,))
        self.assertEqual(NUM_FEATURES, 63)

    def test_output_is_flat_1d(self):
        vector = landmarks_to_feature_vector(_make_fake_hand())
        self.assertEqual(vector.ndim, 1)

    def test_matches_flattened_normalize_landmarks(self):
        hand = _make_fake_hand()
        vector = landmarks_to_feature_vector(hand)
        expected = normalize_landmarks(hand).flatten()
        np.testing.assert_allclose(vector, expected)

    def test_middle_mcp_index_is_within_range(self):
        # Sanity check on the constant itself, not just its use.
        self.assertTrue(0 <= MIDDLE_MCP_INDEX < NUM_LANDMARKS)


if __name__ == "__main__":
    unittest.main()

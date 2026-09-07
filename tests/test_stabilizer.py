"""
Tests for src/vision/stabilizer.py — pure logic, no camera/model needed.

    venv/Scripts/python.exe -m unittest tests.test_stabilizer -v
"""

import unittest

from src.vision.stabilizer import LetterStabilizer

HIGH_CONF = 0.9
LOW_CONF = 0.3


class TestLetterStabilizer(unittest.TestCase):
    def setUp(self):
        # Small window so tests don't need 15+ lines of feeding frames.
        self.stab = LetterStabilizer(window_size=5, min_agreement=0.7, confidence_threshold=0.6)

    def test_not_enough_frames_yet_returns_none(self):
        for _ in range(4):  # window is 5; feed only 4
            result = self.stab.update("A", HIGH_CONF)
        self.assertIsNone(result)

    def test_holding_a_steady_letter_accepts_it_exactly_once(self):
        results = [self.stab.update("A", HIGH_CONF) for _ in range(5)]
        self.assertEqual(results, [None, None, None, None, "A"])

    def test_continuing_to_hold_the_same_letter_does_not_re_accept(self):
        # This is the literal "AAAAAAA must not become AAAAAA" requirement.
        for _ in range(5):
            self.stab.update("A", HIGH_CONF)  # fills window, accepts once
        further = [self.stab.update("A", HIGH_CONF) for _ in range(10)]
        self.assertTrue(all(r is None for r in further))

    def test_releasing_then_showing_the_same_letter_again_accepts_it(self):
        # Needed for real words with doubled letters (e.g. "APPLE" has
        # two P's) -- releasing the hand between them must allow the
        # second one through. The window is a sliding majority vote,
        # so the acceptance can land a couple of frames after the sign
        # changes (not necessarily on the very last frame fed) -- what
        # matters is that "A" gets accepted a second time at all.
        for _ in range(5):
            self.stab.update("A", HIGH_CONF)  # accept "A" once
        for _ in range(5):
            self.stab.update(None, 0.0)  # release: no hand in frame
        results = [self.stab.update("A", HIGH_CONF) for _ in range(8)]
        self.assertEqual(results.count("A"), 1)  # accepted exactly once more

    def test_switching_directly_to_a_different_letter_accepts_it(self):
        for _ in range(5):
            self.stab.update("A", HIGH_CONF)
        results = [self.stab.update("B", HIGH_CONF) for _ in range(8)]
        self.assertEqual(results.count("B"), 1)  # accepted exactly once

    def test_low_confidence_predictions_are_not_accepted(self):
        results = [self.stab.update("A", LOW_CONF) for _ in range(10)]
        self.assertTrue(all(r is None for r in results))

    def test_noisy_mixed_predictions_do_not_reach_agreement(self):
        # Window of 5, needs 70% (>=4/5) agreement -- alternating never gets there.
        letters = ["A", "B", "A", "B", "A", "B", "A", "B"]
        results = [self.stab.update(letter, HIGH_CONF) for letter in letters]
        self.assertTrue(all(r is None for r in results))

    def test_reset_clears_history_and_last_accepted(self):
        for _ in range(5):
            self.stab.update("A", HIGH_CONF)  # accept "A"
        self.stab.reset()
        results = [self.stab.update("A", HIGH_CONF) for _ in range(5)]
        self.assertEqual(results[-1], "A")  # accepted again, as if fresh

    def test_no_hand_produces_no_acceptance(self):
        results = [self.stab.update(None, 0.0) for _ in range(10)]
        self.assertTrue(all(r is None for r in results))


if __name__ == "__main__":
    unittest.main()

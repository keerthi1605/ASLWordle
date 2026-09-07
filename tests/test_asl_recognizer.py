"""
Tests for src/vision/asl_recognizer.py.

Uses a tiny, throwaway RandomForestClassifier trained on synthetic
data (saved to a temp file) rather than the user's real
models/asl_classifier.pkl -- these tests must pass on a fresh clone
before anyone has ever run scripts/collect_data.py or
scripts/train_asl.py.

    venv/Scripts/python.exe -m unittest tests.test_asl_recognizer -v
"""

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.vision.asl_recognizer import ASLRecognizer
from src.vision.landmark_utils import NUM_FEATURES


def _train_tiny_test_model(tmp_path: Path) -> Path:
    """Train a trivially-separable 2-letter classifier so predictions
    are deterministic, and save it where ASLRecognizer expects a
    model file."""
    rng = np.random.default_rng(0)
    # "A" vectors cluster near all-zeros, "B" vectors near all-ones --
    # trivially separable so the classifier is always confident.
    a_samples = rng.normal(0.0, 0.01, size=(20, NUM_FEATURES))
    b_samples = rng.normal(1.0, 0.01, size=(20, NUM_FEATURES))
    X = np.vstack([a_samples, b_samples])
    y = np.array(["A"] * 20 + ["B"] * 20)

    clf = RandomForestClassifier(n_estimators=20, random_state=0)
    clf.fit(X, y)

    model_path = tmp_path / "test_asl_classifier.pkl"
    joblib.dump(clf, model_path)
    return model_path


class TestASLRecognizerWithModel(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        model_path = _train_tiny_test_model(Path(self._tmpdir.name))
        self.recognizer = ASLRecognizer(model_path=model_path)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_model_loads_successfully(self):
        self.assertTrue(self.recognizer.is_available)
        self.assertEqual(self.recognizer.error, "")

    def test_predicts_a_for_an_a_like_vector(self):
        vector = np.zeros(NUM_FEATURES)
        letter, confidence = self.recognizer.predict(vector)
        self.assertEqual(letter, "A")
        self.assertGreater(confidence, 0.5)

    def test_predicts_b_for_a_b_like_vector(self):
        vector = np.ones(NUM_FEATURES)
        letter, confidence = self.recognizer.predict(vector)
        self.assertEqual(letter, "B")
        self.assertGreater(confidence, 0.5)

    def test_confidence_is_between_0_and_1(self):
        _, confidence = self.recognizer.predict(np.zeros(NUM_FEATURES))
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)


class TestASLRecognizerWithoutModel(unittest.TestCase):
    """A fresh clone, before anyone has trained a model."""

    def setUp(self):
        self.recognizer = ASLRecognizer(model_path="models/definitely_does_not_exist.pkl")

    def test_reports_unavailable(self):
        self.assertFalse(self.recognizer.is_available)
        self.assertIn("scripts/collect_data.py", self.recognizer.error)

    def test_predict_returns_none_confidence_zero_never_fabricates(self):
        letter, confidence = self.recognizer.predict(np.zeros(NUM_FEATURES))
        self.assertIsNone(letter)
        self.assertEqual(confidence, 0.0)


if __name__ == "__main__":
    unittest.main()

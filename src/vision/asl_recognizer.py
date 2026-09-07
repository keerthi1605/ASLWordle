"""
Wraps a trained scikit-learn classifier to predict an ASL letter from
a normalized 63-value landmark feature vector (see
src/vision/landmark_utils.py). Contains NO Wordle rules, camera, or
MediaPipe code — see docs/learning/01_project_architecture.md.
"""

import joblib

from config import ASL_MODEL_PATH


class ASLRecognizer:
    """Loads models/asl_classifier.pkl once and predicts from it.
    If the model doesn't exist yet (nobody has run
    scripts/collect_data.py + scripts/train_asl.py), this degrades to
    "unavailable" instead of crashing — see docs/learning/10_hci_design.md
    on honest fallback."""

    def __init__(self, model_path=ASL_MODEL_PATH):
        self.model_path = model_path
        self._model = None
        self.error = ""
        try:
            self._model = joblib.load(model_path)
        except FileNotFoundError:
            self.error = (
                f"No trained ASL model found at {model_path}. Run "
                "scripts/collect_data.py then scripts/train_asl.py."
            )
        except Exception as exc:  # pragma: no cover - defensive, e.g. corrupt file
            self.error = f"Could not load ASL model ({exc})."

    @property
    def is_available(self) -> bool:
        return self._model is not None

    def predict(self, feature_vector) -> tuple[str | None, float]:
        """Predict a letter from one 63-value feature vector. Returns
        (letter, confidence) with confidence in [0, 1], or (None, 0.0)
        if no model is loaded. Never fabricates a prediction."""
        if self._model is None:
            return None, 0.0
        probabilities = self._model.predict_proba([feature_vector])[0]
        best_index = probabilities.argmax()
        letter = self._model.classes_[best_index]
        confidence = float(probabilities[best_index])
        return letter, confidence

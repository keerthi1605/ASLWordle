"""
Tests for src/speech/whisper_service.py.

Loading the real Whisper model (a few seconds, plus a one-time
download) and recording from a real microphone happen when available;
those tests SKIP (not fail) when they aren't — same philosophy as
tests/test_camera.py. The text-processing functions need neither and
always run.

    venv/Scripts/python.exe -m unittest tests.test_whisper_service -v
"""

import unittest

import numpy as np

from src.speech.whisper_service import WhisperService, extract_candidate_word, normalize_text


class TestNormalizeText(unittest.TestCase):
    def test_uppercases(self):
        self.assertEqual(normalize_text("apple"), "APPLE")

    def test_strips_punctuation(self):
        self.assertEqual(normalize_text("Apple."), "APPLE")
        self.assertEqual(normalize_text("apple, please"), "APPLE PLEASE")

    def test_collapses_whitespace(self):
        self.assertEqual(normalize_text("  apple   pie  "), "APPLE PIE")

    def test_empty_input(self):
        self.assertEqual(normalize_text(""), "")


class TestExtractCandidateWord(unittest.TestCase):
    def test_clean_single_word(self):
        self.assertEqual(extract_candidate_word("apple"), "APPLE")

    def test_with_trailing_punctuation(self):
        self.assertEqual(extract_candidate_word("Apple."), "APPLE")

    def test_extracts_from_sentence(self):
        self.assertEqual(extract_candidate_word("the word is apple"), "APPLE")

    def test_no_valid_length_word_returns_none(self):
        self.assertIsNone(extract_candidate_word("hi you"))  # 2 and 3 letters, no 5

    def test_empty_transcription_returns_none(self):
        self.assertIsNone(extract_candidate_word(""))

    def test_wrong_length_word_alone_returns_none(self):
        self.assertIsNone(extract_candidate_word("cat"))  # 3 letters, word_length=5

    def test_picks_the_matching_length_word_from_a_sentence(self):
        self.assertEqual(extract_candidate_word("say mouse now"), "MOUSE")

    def test_custom_word_length(self):
        self.assertEqual(extract_candidate_word("cat", word_length=3), "CAT")

    def test_never_fabricates_a_word_of_the_wrong_length(self):
        # Regression-style check: a 4-letter word must never be
        # padded/guessed into a 5-letter one.
        self.assertIsNone(extract_candidate_word("blue"))


def _whisper_available() -> bool:
    try:
        return WhisperService().is_available
    except Exception:
        return False


_HAS_WHISPER = _whisper_available()


@unittest.skipUnless(_HAS_WHISPER, "Whisper model could not be loaded (no internet/disk on first run?)")
class TestWhisperServiceWithModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = WhisperService()

    def test_model_loads(self):
        self.assertTrue(self.service.is_available)
        self.assertEqual(self.service.error, "")

    def test_device_is_cpu_or_cuda(self):
        self.assertIn(self.service.device, ("cpu", "cuda"))

    def test_transcribe_silence_does_not_crash(self):
        silence = np.zeros(16000 * 2, dtype=np.float32)  # 2s of silence
        ok, text, message = self.service.transcribe(silence)
        self.assertTrue(ok)
        self.assertEqual(message, "")
        self.assertIsInstance(text, str)


class TestWhisperServiceErrorHandling(unittest.TestCase):
    def test_invalid_model_size_reports_unavailable(self):
        service = WhisperService(model_size="not_a_real_whisper_model")
        self.assertFalse(service.is_available)
        self.assertNotEqual(service.error, "")

    def test_transcribe_without_model_returns_honest_failure(self):
        service = WhisperService(model_size="not_a_real_whisper_model")
        ok, text, message = service.transcribe(np.zeros(1000, dtype=np.float32))
        self.assertFalse(ok)
        self.assertEqual(text, "")
        self.assertTrue(message)


if __name__ == "__main__":
    unittest.main()

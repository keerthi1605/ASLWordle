"""
Wraps OpenAI's Whisper for speech-to-text: record a short clip from the
microphone, transcribe it, and pull out a single word-shaped candidate
guess. Contains NO Wordle rules — see
docs/learning/01_project_architecture.md. Never claims to have
"understood" what the user meant; it only ever returns exactly what
Whisper transcribed, or an honest failure/empty result.
"""

import string

import numpy as np
import sounddevice as sd
import whisper

from config import WHISPER_MODEL_SIZE

SAMPLE_RATE = 16000  # Whisper expects 16kHz mono audio


def _detect_device() -> str:
    """GPU when available, CPU otherwise — Whisper runs fine on either,
    just slower on CPU. See docs/learning/09_whisper.md."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover - torch ships with openai-whisper
        return "cpu"


class WhisperService:
    """Loads the Whisper model once (expensive: a few seconds, plus a
    one-time download) and reuses it for every recording."""

    def __init__(self, model_size: str = WHISPER_MODEL_SIZE):
        self.model_size = model_size
        self.device = _detect_device()
        self._model = None
        self.error = ""
        try:
            self._model = whisper.load_model(model_size, device=self.device)
        except Exception as exc:  # pragma: no cover - network/disk failure
            self.error = f"Could not load Whisper model ({exc})."

    @property
    def is_available(self) -> bool:
        return self._model is not None

    def record_audio(self, duration_seconds: float = 4.0) -> tuple[bool, np.ndarray | None, str]:
        """Record mono audio from the default microphone. Returns
        (success, audio_array_or_None, message) — never raises for a
        missing/busy microphone, same honest-fallback pattern as
        src/vision/camera.py."""
        try:
            audio = sd.rec(
                int(duration_seconds * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            return True, audio.flatten(), ""
        except Exception as exc:
            return False, None, f"Could not record audio ({exc}). Check your microphone."

    def transcribe(self, audio: np.ndarray) -> tuple[bool, str, str]:
        """Transcribe a mono float32 audio array. Returns
        (success, raw_text, message). `raw_text` is exactly what
        Whisper heard — normalization/candidate extraction happen
        separately (see normalize_text / extract_candidate_word)."""
        if self._model is None:
            return False, "", "Whisper model is not loaded."
        try:
            result = self._model.transcribe(audio, fp16=(self.device == "cuda"))
            return True, result["text"].strip(), ""
        except Exception as exc:  # pragma: no cover - defensive
            return False, "", f"Transcription failed ({exc})."


def normalize_text(raw_text: str) -> str:
    """Upper-case and strip punctuation/whitespace: Whisper's "Apple."
    or "apple," both become "APPLE". Doesn't touch word boundaries."""
    cleaned = raw_text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(cleaned.upper().split())


def extract_candidate_word(raw_text: str, word_length: int = 5) -> str | None:
    """
    Pull one word_length-letter alphabetic candidate out of whatever
    Whisper transcribed — handles both a clean single-word answer
    ("apple") and a full sentence ("the word is apple, I think").
    Returns None if no such word is found; never guesses or pads.
    """
    normalized = normalize_text(raw_text)
    if not normalized:
        return None
    if len(normalized) == word_length and normalized.isalpha():
        return normalized
    candidates = [tok for tok in normalized.split() if len(tok) == word_length and tok.isalpha()]
    return candidates[0] if candidates else None

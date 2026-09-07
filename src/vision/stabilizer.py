"""
Temporal smoothing for ASL letter predictions.

A single video frame's prediction is noisy: a slightly turned wrist,
one bad lighting frame, or classifier jitter can flip the predicted
letter for an instant. Accepting every frame's raw prediction would
also flood the current guess with repeats -- holding "A" steady for
two seconds would type "AAAAAAAAAA...". This module fixes both
problems with a small rolling-window majority vote.

Knows nothing about Wordle, cameras, or MediaPipe -- see
docs/learning/01_project_architecture.md. It only ever deals with
plain (letter, confidence) pairs, which makes it fully testable
without a camera or a trained model.
"""

from collections import deque

from config import CONFIDENCE_THRESHOLD, STABILIZATION_MIN_AGREEMENT, STABILIZATION_WINDOW


class LetterStabilizer:
    """Feed it one (letter, confidence) reading per frame; it tells
    you when a letter should actually be accepted into the guess."""

    def __init__(
        self,
        window_size: int = STABILIZATION_WINDOW,
        min_agreement: float = STABILIZATION_MIN_AGREEMENT,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ):
        self.window_size = window_size
        self.min_agreement = min_agreement
        self.confidence_threshold = confidence_threshold
        self._buffer = deque(maxlen=window_size)
        self._last_accepted = None  # the letter most recently added to the guess

    def reset(self) -> None:
        """Clear all history — call this when starting a fresh guess
        (e.g. after Clear/Submit) so old frames can't influence it."""
        self._buffer.clear()
        self._last_accepted = None

    def update(self, letter: str | None, confidence: float) -> str | None:
        """
        Feed one frame's raw prediction. `letter` may be None (no hand,
        or the recognizer had nothing to say). Returns the letter to
        ACCEPT into the guess this frame, or None if nothing should be
        accepted yet.
        """
        # A prediction only counts if it clears the confidence bar --
        # otherwise it's treated the same as "no hand" for voting purposes.
        entry = letter if (letter is not None and confidence >= self.confidence_threshold) else None
        self._buffer.append(entry)

        if len(self._buffer) < self.window_size:
            return None  # not enough history yet to trust a decision

        candidate, count = self._majority(self._buffer)
        agreement = count / len(self._buffer)

        if candidate is None or agreement < self.min_agreement:
            # No hand, or the window's predictions are too mixed to
            # trust. This also "releases" the last accepted letter --
            # see the class docstring: the user has to visibly move
            # away from a sign before it can be accepted again, which
            # is exactly what a mixed/empty window means happened.
            self._last_accepted = None
            return None

        if candidate == self._last_accepted:
            return None  # still the same held pose -- don't re-accept it

        self._last_accepted = candidate
        return candidate

    @staticmethod
    def _majority(buffer) -> tuple[str | None, int]:
        """Return (most_common_value, its_count) in `buffer`."""
        counts: dict[str | None, int] = {}
        for item in buffer:
            counts[item] = counts.get(item, 0) + 1
        return max(counts.items(), key=lambda kv: kv[1])

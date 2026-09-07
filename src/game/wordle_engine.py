"""
Wordle rules, and nothing else.

This module has ZERO knowledge of Streamlit, OpenCV, MediaPipe, or
Whisper — it only understands strings. That's deliberate (see
docs/learning/01_project_architecture.md): whoever produces a guess
(keyboard, ASL, or voice) hands it to `WordleEngine.submit_guess()`
and gets back the same structured feedback either way.
"""

import random

from config import MAX_ATTEMPTS, WORD_LENGTH
from src.game.word_list import get_random_target, is_valid_word

# Feedback tags used everywhere else in the app (UI colors off of these).
CORRECT = "correct"
PRESENT = "present"
ABSENT = "absent"

_STATUS_PRIORITY = {ABSENT: 0, PRESENT: 1, CORRECT: 2}


def evaluate_guess(guess: str, target: str) -> list[str]:
    """
    Compare `guess` against `target` and return a list of per-letter
    feedback tags: "correct", "present", or "absent".

    This is the classic two-pass Wordle algorithm, which is required
    to handle duplicate letters correctly. A naive `letter in target`
    check breaks as soon as a letter repeats — see
    docs/learning/08_wordle_engine.md for a worked example.
    """
    guess = guess.upper()
    target = target.upper()
    n = len(target)
    feedback = [ABSENT] * n

    # remaining_letters tracks target letters not yet "claimed" by a
    # correct or present match. We use a list with None-ing instead of
    # a simple Counter so index-based consumption is easy to follow.
    remaining_letters = list(target)

    # Pass 1: exact position matches ("correct") are locked in first
    # and removed from the pool, so a later duplicate can't steal them.
    for i in range(n):
        if guess[i] == target[i]:
            feedback[i] = CORRECT
            remaining_letters[i] = None

    # Pass 2: any guess letter not already marked correct can still be
    # "present" if the target has an unclaimed copy of it left.
    for i in range(n):
        if feedback[i] == CORRECT:
            continue
        letter = guess[i]
        if letter in remaining_letters:
            feedback[i] = PRESENT
            remaining_letters[remaining_letters.index(letter)] = None
        # else stays ABSENT

    return feedback


class WordleEngine:
    """Holds the state for a single game and enforces the rules."""

    def __init__(
        self,
        word_length: int = WORD_LENGTH,
        max_attempts: int = MAX_ATTEMPTS,
        target: str | None = None,
        check_dictionary: bool = True,
        rng: random.Random | None = None,
    ):
        self.word_length = word_length
        self.max_attempts = max_attempts
        self.check_dictionary = check_dictionary
        self._rng = rng
        self.reset(target)

    def reset(self, target: str | None = None) -> None:
        """Start a brand new game. Pass `target` to force a specific
        word (used heavily in tests; the UI leaves it random)."""
        self.target = (target or get_random_target(self._rng)).upper()
        self.guesses: list[str] = []
        self.feedback_history: list[list[str]] = []
        self.attempts_used = 0
        self.game_over = False
        self.won = False
        # Best status seen so far per letter, for coloring the on-screen
        # keyboard (a letter that was "present" once and "correct" later
        # should show as correct, never downgrade).
        self.letter_status: dict[str, str] = {}

    @property
    def attempts_remaining(self) -> int:
        return self.max_attempts - self.attempts_used

    def validate_guess(self, guess: str) -> tuple[bool, str]:
        """Structural + dictionary checks. Returns (is_valid, reason)."""
        if self.game_over:
            return False, "Game is over — restart to play again."
        guess = guess.strip()
        if len(guess) != self.word_length:
            return False, f"Please enter a {self.word_length}-letter word."
        if not guess.isalpha():
            return False, "Please use letters only (A-Z)."
        if self.check_dictionary and not is_valid_word(guess):
            return False, "Not in word list."
        return True, ""

    def submit_guess(self, guess: str) -> dict:
        """
        Attempt to submit a guess. Returns a result dict:
            {"accepted": False, "reason": "..."}                     if rejected
            {"accepted": True, "feedback": [...], "game_over": bool,
             "won": bool, "attempts_remaining": int}                 if accepted
        """
        guess = guess.strip().upper()
        is_valid, reason = self.validate_guess(guess)
        if not is_valid:
            return {"accepted": False, "reason": reason}

        feedback = evaluate_guess(guess, self.target)
        self.guesses.append(guess)
        self.feedback_history.append(feedback)
        self.attempts_used += 1
        self._update_letter_status(guess, feedback)

        if guess == self.target:
            self.game_over = True
            self.won = True
        elif self.attempts_used >= self.max_attempts:
            self.game_over = True
            self.won = False

        return {
            "accepted": True,
            "feedback": feedback,
            "game_over": self.game_over,
            "won": self.won,
            "attempts_remaining": self.attempts_remaining,
        }

    def _update_letter_status(self, guess: str, feedback: list[str]) -> None:
        for letter, status in zip(guess, feedback):
            current_best = self.letter_status.get(letter)
            if current_best is None or _STATUS_PRIORITY[status] > _STATUS_PRIORITY[current_best]:
                self.letter_status[letter] = status

    def get_state(self) -> dict:
        """A plain-data snapshot the UI can render without touching
        engine internals directly."""
        return {
            "guesses": list(self.guesses),
            "feedback_history": [list(f) for f in self.feedback_history],
            "attempts_used": self.attempts_used,
            "attempts_remaining": self.attempts_remaining,
            "max_attempts": self.max_attempts,
            "word_length": self.word_length,
            "game_over": self.game_over,
            "won": self.won,
            "letter_status": dict(self.letter_status),
            # Only meaningful to reveal once the game has ended.
            "target": self.target if self.game_over else None,
        }

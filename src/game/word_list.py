"""
Word list for SignWordle — the real original-Wordle word lists, not a
hand-picked substitute.

Source: the original Wordle (by Josh Wardle, before the NYT acquired
it) shipped its two word lists in plain client-side JavaScript. Once
NYT took over, those original arrays stopped being publicly served,
but they had already been extracted and mirrored in many places; the
copies here come from a well-known GitHub Gist mirror
(https://gist.github.com/cfreshman/a03ef2cba789d8cf00c08f767e0fad7b for
answers, https://gist.github.com/cfreshman/cdcdf777450c5b5301e439061d29694c
for the extra allowed guesses). This is NOT an NYT-licensed dataset —
it's a community-preserved copy of code Wordle itself once shipped
publicly. Verified honestly: answers (2,315) + allowed guesses (10,657)
= 12,972 words, matching the real game's well-documented total exactly.

Two separate files, because real Wordle treats them differently:

1. `data/wordle_answers.txt` (2,315 words) — the only words ever picked
   as the target for a new game. Keeps every game's answer a common,
   guessable word (no "AAHED" as an answer).
2. `data/wordle_allowed_guesses.txt` (10,657 words) — extra words the
   real game accepts as a *guess* (for narrowing down letters) but
   never uses as the *answer*. Combined with the answers, that's every
   word `is_valid_word()` accepts.

Loaded once from disk at import time — no network access while
playing, no dependency beyond two plain text files already in the repo.
"""

import random

from config import WORDLE_ALLOWED_GUESSES_PATH, WORDLE_ANSWERS_PATH


def _load_words(path) -> list[str]:
    """Read one word per line, uppercase, and defensively drop anything
    that isn't a clean 5-letter alphabetic word (guards against a stray
    blank line or encoding hiccup in the source file)."""
    with open(path, encoding="utf-8") as f:
        return [line.strip().upper() for line in f if len(line.strip()) == 5 and line.strip().isalpha()]


# The pool a new game's target is picked from — real Wordle answers only.
WORDS = sorted(set(_load_words(WORDLE_ANSWERS_PATH)))

# Every word accepted as a guess: answers plus the extra allowed-guess
# list. A superset of WORDS, same relationship the real game has.
_ALLOWED_GUESSES = set(_load_words(WORDLE_ALLOWED_GUESSES_PATH))
_WORD_SET = set(WORDS) | _ALLOWED_GUESSES


def is_valid_word(word: str) -> bool:
    """Return True if `word` (any case) is an accepted guess — i.e. it's
    a real Wordle answer OR one of the extra allowed-guess words."""
    return word.upper() in _WORD_SET


def get_random_target(rng: random.Random | None = None) -> str:
    """Pick a random target word from the real answer list only. Pass
    `rng` for deterministic tests."""
    rng = rng or random
    return rng.choice(WORDS)

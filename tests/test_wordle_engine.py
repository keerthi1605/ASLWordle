"""
Unit tests for the Wordle engine.

Run from the project root (so `config.py` and `src/` are importable):

    venv/Scripts/python.exe -m unittest tests.test_wordle_engine -v

No pytest/extra dependency needed — this only uses the standard
library's `unittest`, on purpose (see requirements.txt notes).
"""

import unittest

from src.game.wordle_engine import ABSENT, CORRECT, PRESENT, WordleEngine, evaluate_guess


class TestEvaluateGuess(unittest.TestCase):
    """Tests for the pure scoring function (no game state involved)."""

    def test_all_correct(self):
        self.assertEqual(
            evaluate_guess("APPLE", "APPLE"),
            [CORRECT, CORRECT, CORRECT, CORRECT, CORRECT],
        )

    def test_all_absent(self):
        # "GHOST" shares no letters with "BLAND"
        self.assertEqual(
            evaluate_guess("GHOST", "BLAND"),
            [ABSENT, ABSENT, ABSENT, ABSENT, ABSENT],
        )

    def test_mixed_present_and_correct(self):
        # target APPLE, guess LEARN -> every letter of LEARN exists
        # somewhere in APPLE, but never in the matching position.
        self.assertEqual(
            evaluate_guess("LEARN", "APPLE"),
            [PRESENT, PRESENT, PRESENT, ABSENT, ABSENT],
        )

    def test_duplicate_letters_both_present(self):
        # target SPEED has two E's, guess ERASE has two E's too, and
        # none land in the correct slot -> both should be "present",
        # not just one.
        self.assertEqual(
            evaluate_guess("ERASE", "SPEED"),
            [PRESENT, ABSENT, ABSENT, PRESENT, PRESENT],
        )

    def test_duplicate_letters_limited_by_target_count(self):
        # target APPLE has exactly one L, and guess LOLLY's third L
        # happens to land in APPLE's L slot (correct). The other two
        # L's in the guess must NOT also show as "present" -- there's
        # no more unclaimed L left in the target.
        self.assertEqual(
            evaluate_guess("LOLLY", "APPLE"),
            [ABSENT, ABSENT, ABSENT, CORRECT, ABSENT],
        )

    def test_correct_takes_priority_over_present_for_same_letter(self):
        # target ALLOW: guess LLAMA-style edge case using two L's in
        # the guess where only one L slot exists in the target and it
        # is an exact-position match.
        # target: A L L O W ; guess: L L A M A
        # pos0 L vs A -> no; pos1 L vs L -> correct; pos2 A vs L -> no
        # pos3 M vs O -> no; pos4 A vs W -> no
        # Pass2: pos0 L -> remaining has L at idx2 -> present, consume
        #        pos2 A -> remaining has A at idx0 -> present, consume
        #        pos3 M -> absent; pos4 A -> remaining has no A left -> absent
        self.assertEqual(
            evaluate_guess("LLAMA", "ALLOW"),
            [PRESENT, CORRECT, PRESENT, ABSENT, ABSENT],
        )


class TestWordleEngineValidation(unittest.TestCase):
    def setUp(self):
        self.engine = WordleEngine(target="APPLE")

    def test_rejects_wrong_length(self):
        result = self.engine.submit_guess("CAT")
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "Please enter a 5-letter word.")

    def test_rejects_non_alpha(self):
        result = self.engine.submit_guess("AP9LE")
        self.assertFalse(result["accepted"])
        self.assertIn("letters", result["reason"])

    def test_rejects_word_not_in_dictionary(self):
        result = self.engine.submit_guess("ZZZZZ")
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "Not in word list.")

    def test_accepts_valid_word(self):
        result = self.engine.submit_guess("ABOUT")
        self.assertTrue(result["accepted"])

    def test_dictionary_check_can_be_disabled(self):
        # Useful for ASL/voice demos with words outside the curated list.
        engine = WordleEngine(target="APPLE", check_dictionary=False)
        result = engine.submit_guess("ZZZZZ")
        self.assertTrue(result["accepted"])


class TestWordleEngineGameplay(unittest.TestCase):
    def setUp(self):
        self.engine = WordleEngine(target="APPLE")

    def test_win_on_correct_guess(self):
        result = self.engine.submit_guess("apple")  # lowercase should work too
        self.assertTrue(result["accepted"])
        self.assertTrue(result["game_over"])
        self.assertTrue(result["won"])
        self.assertEqual(result["feedback"], [CORRECT] * 5)

    def test_lose_after_max_attempts(self):
        wrong_guesses = ["ABOUT", "ABOVE", "ABUSE", "ACTOR", "ACUTE", "ADMIT"]
        for i, guess in enumerate(wrong_guesses, start=1):
            result = self.engine.submit_guess(guess)
            self.assertTrue(result["accepted"])
            if i < 6:
                self.assertFalse(result["game_over"])
        # 6th wrong guess ends the game as a loss
        self.assertTrue(result["game_over"])
        self.assertFalse(result["won"])

    def test_cannot_guess_after_game_over(self):
        self.engine.submit_guess("APPLE")  # wins immediately
        result = self.engine.submit_guess("ABOUT")
        self.assertFalse(result["accepted"])
        self.assertIn("Game is over", result["reason"])

    def test_attempts_remaining_counts_down(self):
        self.assertEqual(self.engine.attempts_remaining, 6)
        self.engine.submit_guess("ABOUT")
        self.assertEqual(self.engine.attempts_remaining, 5)

    def test_restart_resets_state(self):
        self.engine.submit_guess("ABOUT")
        self.engine.reset(target="CRANE")
        self.assertEqual(self.engine.target, "CRANE")
        self.assertEqual(self.engine.attempts_used, 0)
        self.assertEqual(self.engine.guesses, [])
        self.assertFalse(self.engine.game_over)

    def test_target_hidden_until_game_over(self):
        self.engine.submit_guess("ABOUT")
        self.assertIsNone(self.engine.get_state()["target"])
        self.engine.submit_guess("APPLE")
        self.assertEqual(self.engine.get_state()["target"], "APPLE")

    def test_letter_status_never_downgrades(self):
        # First guess marks 'A' as present (APPLE has an A, not at pos0).
        engine = WordleEngine(target="APPLE", check_dictionary=False)
        engine.submit_guess("LEARN")  # A is present, not correct
        self.assertEqual(engine.letter_status["A"], PRESENT)
        engine.submit_guess("APPLE")  # A now correct at pos0
        self.assertEqual(engine.letter_status["A"], CORRECT)


if __name__ == "__main__":
    unittest.main()

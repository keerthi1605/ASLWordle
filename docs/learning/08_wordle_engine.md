# 08 — The Wordle Engine

## 1. What it does

`src/game/wordle_engine.py` implements the actual rules of Wordle:
picking a target word, checking a guess against it, tracking attempts,
and deciding win/lose. It is pure Python — no UI, no camera, no
speech. See [wordle_engine.py](../../src/game/wordle_engine.py).

## 2. Why it's needed

Every input mode (keyboard, ASL, voice) eventually produces a 5-letter
string. Something has to turn that string into "which letters are
right" and track the game across 6 attempts — and it must do so
**identically** no matter which input mode produced the guess. That's
this file's entire job.

## 3. How it works internally

### 3.1 The scoring algorithm (`evaluate_guess`)

The naive approach — "is this letter anywhere in the target?" — breaks
as soon as letters repeat. Example: target `APPLE`, guess `LOLLY`.
`LOLLY` has three L's; `APPLE` has only one. A naive check would mark
all three L's as "present," which is wrong — Wordle (and this engine)
only ever credits as many copies of a letter as actually exist in the
target.

The fix is a **two-pass** algorithm:

**Pass 1 — exact matches.** Walk both strings position by position.
Wherever `guess[i] == target[i]`, mark that position `"correct"` and
remove that letter from a working copy of the target (`remaining_letters`)
so it can't be claimed again later.

**Pass 2 — present-but-misplaced.** For every position not already
`"correct"`, check whether `remaining_letters` (whatever's left after
pass 1) still contains that letter. If yes, mark `"present"` and
consume one copy. If no, it's `"absent"`.

Worked example — target `APPLE`, guess `LOLLY`:

```
target:  A  P  P  L  E
guess:   L  O  L  L  Y

Pass 1 (position matches):
  pos0: L vs A -> no
  pos1: O vs P -> no
  pos2: L vs P -> no
  pos3: L vs L -> MATCH -> "correct", remove that L from remaining
  pos4: Y vs E -> no
  remaining = [A, P, P, _, E]   (the L at index 3 is used up)

Pass 2 (whatever's left):
  pos0: L -> no L left in remaining -> "absent"
  pos1: O -> not in target at all -> "absent"
  pos2: L -> no L left -> "absent"
  pos3: already "correct", skip
  pos4: Y -> not in target -> "absent"

Result: [absent, absent, absent, correct, absent]
```

Only the L that's actually in the right spot counts — the other two
L's in the guess get nothing, because the target only had one L to
give out. This is exactly the behavior real Wordle has, and it's why
`docs/learning` and `tests/test_wordle_engine.py` both spend so much
time on duplicate-letter cases — it's the one part of "just compare
two strings" that's genuinely easy to get wrong.

### 3.2 Game state (`WordleEngine`)

The class wraps `evaluate_guess()` with everything a real game needs:

- `target` — the answer (hidden from the UI until `game_over`)
- `guesses` / `feedback_history` — every guess made and its scoring
- `attempts_used` / `attempts_remaining`
- `game_over` / `won`
- `letter_status` — the *best* status ever seen for each letter (used
  to color the on-screen keyboard — a letter that was "present" in one
  guess and later "correct" should show as correct, never downgrade)

`submit_guess()` is the only way to mutate this state. It always
validates first (`validate_guess`): correct length, letters only, and
(optionally) present in the real Wordle word list — see
`src/game/word_list.py`, and §12a in the README for where that list
comes from. Rejected guesses **do not** consume an attempt, which
matters for UX: a typo shouldn't cost you a turn.

`check_dictionary=False` exists as a seam for tests (build an engine
with a known target and skip the dictionary check to test scoring
logic in isolation) — every real input mode (keyboard, ASL, voice)
uses the default `check_dictionary=True`, going through the same
`is_valid_word()` check via the shared `_submit_word()` funnel in
`app.py`. No input mode gets a looser dictionary than any other.

## 4. Important concepts

- **Idempotent scoring**: `evaluate_guess()` is a pure function — same
  inputs always give the same output, no hidden state. This is what
  makes it trivially unit-testable.
- **Never downgrade** letter status — a coloring/UX detail that matters
  more than it looks: it's what makes the on-screen keyboard trustworthy
  over multiple guesses.
- **Structural vs. semantic validation** — length/alphabetic checks catch
  garbage input; the dictionary check catches "not a real word," and
  they're deliberately separate reasons so the UI can show a precise
  message either way.

## 5. Input / Output

- Input: a guess string (any case, any source) and, internally, the
  hidden target string.
- Output: `submit_guess()` returns a plain dict — `accepted`,
  `feedback` (list of `"correct"/"present"/"absent"`), `game_over`,
  `won`, `attempts_remaining`. `get_state()` returns a full snapshot
  for the UI to render, with `target` set to `None` until the game
  ends (so the UI can't accidentally leak the answer).

## 6. Connection to other components

`src/ui/game_ui.py` calls `engine.get_state()` to draw the grid/keyboard
and `engine.submit_guess()` when the user presses Submit — from
**any** input mode. Vision (Phase 3+) and speech (Phase 9) code will
build a 5-letter string and call the exact same `submit_guess()`, which
is the entire point of keeping this file free of UI/CV/ML imports.

## 7. Important functions/classes

- `evaluate_guess(guess, target) -> list[str]` — the scoring algorithm.
- `WordleEngine.submit_guess(guess) -> dict`
- `WordleEngine.validate_guess(guess) -> (bool, str)`
- `WordleEngine.get_state() -> dict`
- `WordleEngine.reset(target=None)` — new game / restart.

## 8. Common errors

- Comparing guess/target without `.upper()` first — case mismatches
  silently break equality checks. The engine normalizes to uppercase
  everywhere.
- Consuming target letters with a `Counter` decrement without also
  removing correct-position matches *first* — if pass 2 runs before
  pass 1's exact matches remove their letters from the pool, an exact
  match can accidentally "steal" a present-slot from a duplicate
  letter elsewhere in the guess (or vice versa), corrupting the count.
- Forgetting to check `game_over` before accepting a new guess —
  without that check a player could keep submitting guesses forever
  after already winning or losing.

## 9. Likely viva questions

- **"Why not just do `if letter in target`?"** — Because it can't
  count. It would mark every occurrence of a repeated guess letter as
  "present" even if the target only has one copy. See the `LOLLY` /
  `APPLE` example above.
- **"Walk me through duplicate-letter handling."** — Two passes:
  lock in exact-position matches and remove them from a working pool
  first; then for every remaining letter, check if the pool still has
  an unclaimed copy.
- **"Why is `target` hidden until game over?"** — Basic anti-cheating /
  correct information architecture: the UI should only be able to leak
  the answer once the round has legitimately ended.
- **"Why does an invalid guess not cost an attempt?"** — Matches real
  Wordle UX and is simply better HCI — punishing a typo the same as a
  wrong guess would be needlessly frustrating (error prevention/user
  control, see `docs/learning/10_hci_design.md`).
- **"Why is this file forbidden from importing Streamlit/OpenCV/etc.?"**
  — Separation of concerns: it lets keyboard mode work with zero
  dependency on CV/ML, and it's what lets ASL, voice, and keyboard
  guarantee identical scoring.

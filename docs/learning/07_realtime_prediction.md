# 07 — Real-Time Prediction & Temporal Stabilization

## 1. What it does

Turns a noisy, frame-by-frame stream of ASL letter predictions into
occasional, deliberate "accept this letter" decisions — so holding a
sign steady for a few seconds produces exactly **one** letter, not a
flood of repeats, and one bad frame doesn't corrupt the guess. See
[stabilizer.py](../../src/vision/stabilizer.py).

## 2. The problem: frame-by-frame prediction is noisy

Every video frame gets its own independent classifier prediction.
Lighting flicker, a slightly turned wrist, or classifier jitter can
flip the predicted letter for a single frame even while your hand
shape hasn't meaningfully changed. Accepting every frame's raw
prediction directly into the guess would also be wrong in an obvious
way: holding "A" steady for three seconds at even a modest frame rate
would type "AAAAAAAAAA..." — completely unusable.

## 3. The fix: a rolling-window majority vote

`LetterStabilizer` keeps a small rolling buffer of the last N raw
predictions. On each new frame:

1. A prediction only "counts" if its confidence clears
   `CONFIDENCE_THRESHOLD` — a low-confidence guess is treated the same
   as "no hand" for voting purposes.
2. Once the buffer is full, take the **majority** value in it.
3. Only accept that majority if it covers at least
   `STABILIZATION_MIN_AGREEMENT` (70%) of the window — otherwise the
   window is "too mixed" to trust yet.
4. Only accept it if it's *different* from the last letter you already
   accepted — this is what stops "AAAAAA": once "A" is accepted, the
   stabilizer won't re-accept "A" again until the window's majority
   genuinely changes away from "A" first (hand released, or a
   different sign shown) — see the class docstring and
   `tests/test_stabilizer.py` for exact scenarios, including the
   double-letter case (e.g. the two P's in "APPLE").

## 4. Confidence thresholds

Two different thresholds work together, on purpose:

- **Per-frame** (`CONFIDENCE_THRESHOLD`): is *this one prediction*
  trustworthy enough to even count towards the vote?
- **Per-window** (`STABILIZATION_MIN_AGREEMENT`): do *enough recent
  frames* agree with each other to trust the letter as stable?

A single lucky high-confidence frame isn't enough on its own; it has
to be part of a consistent trend.

## 5. A real engineering surprise worth knowing (great viva material)

The stabilization window size was originally planned around ~10-15
FPS (a reasonable guess for "a camera loop"). Building the live
integration (see `docs/learning/11_integration.md` /
`app.py::render_asl_tab`) surfaced a real, measured constraint:
Streamlit's `st.fragment(run_every=...)` — the mechanism used to
auto-update the camera view without refreshing the rest of the page
(see `docs/learning/02_opencv.md` §3.3-3.4) — has a practical **~1
frame/sec** ceiling in the installed Streamlit version, *regardless*
of the requested interval (confirmed by directly timing the
camera-read + MediaPipe-detect + JPEG-encode pipeline in isolation:
~50-80ms total, nowhere near 1 second — so the bottleneck is
Streamlit's own fragment scheduler, not this project's code).

At ~10 FPS, the original 15-frame window would take ~1.5 seconds to
accept a letter. At the *actual* ~1 FPS, that same window would take
**15 seconds** — unusably slow. `STABILIZATION_WINDOW` was reduced to
**3** frames specifically to match the real, measured rate (~3 seconds
to hold a sign steady) instead of an assumed one. This is the honest
engineering process this project is built around: measure, don't
assume — the same principle behind never fabricating a dataset or an
accuracy number.

## 6. Why this improves the HCI, specifically

- **Error prevention**: a single bad frame can't accidentally corrupt
  the guess (see `docs/learning/10_hci_design.md`).
- **Predictable behavior**: holding a sign produces exactly one
  letter, matching what a user intuitively expects ("I showed one
  letter, I should get one letter").
- **Visible feedback while it's still deciding**: `app.py` shows
  "Predicted: X (NN%) — hold steady…" *before* acceptance, so the user
  isn't left guessing whether the system is even seeing their hand —
  see the Visibility principle in `docs/learning/10_hci_design.md`.

## 7. Input / Output

- Input: one `(letter_or_None, confidence)` pair per frame.
- Output: `update()` returns the letter to accept THIS frame, or
  `None` if nothing should be accepted yet.

## 8. Connection to other components

Sits between `ASLRecognizer` (Phase 6, produces the raw per-frame
prediction) and the Wordle-facing `_add_letter()` in `app.py` (the
same function keyboard mode uses — see
`docs/learning/01_project_architecture.md`). Knows nothing about
Wordle, cameras, or MediaPipe; it only ever handles plain
`(letter, confidence)` pairs, which is what makes it fully unit
testable without a camera or a trained model
(`tests/test_stabilizer.py`).

## 9. Common errors

- **Resetting the stabilizer too often** — clearing its history on
  every tiny UI interaction would make it forget genuinely-in-progress
  holds; this project only resets it on `Restart Game`.
- **Sizing the window off an assumed frame rate instead of a measured
  one** — see §5; always verify the actual achievable rate before
  tuning a time-based window.
- **Treating "no hand" and "low confidence" differently in the vote**
  — both are folded into the same `None` entry deliberately, so a
  momentary low-confidence blip behaves the same as a momentary hand
  drop-out.

## 10. Likely viva questions

- **"How do you stop 'AAAAAA' from a held sign?"** — Majority-vote
  over a rolling window, and refusing to re-accept the same letter
  until the window's majority genuinely changes away from it first.
- **"How do you handle a word with a double letter, like the P's in
  APPLE?"** — The user has to visibly release/change the sign between
  the two; once the window shows something other than "P" (even
  briefly), a following "P" is treated as a new instance and accepted
  again. Tested directly in `tests/test_stabilizer.py`.
- **"Why is your stabilization window only 3 frames, when 15 is more
  common in tutorials?"** — Because 15 was sized against this
  project's *actual measured* ~1 FPS tick rate, not an assumed 10-15
  FPS; a smaller window keeps letter-acceptance latency reasonable
  given that real constraint.
- **"What's the difference between the per-frame and per-window
  confidence checks?"** — Per-frame asks "is this one prediction
  trustworthy," per-window asks "do enough recent predictions agree" —
  both have to pass before a letter is accepted.

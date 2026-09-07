# 11 — Integration: Camera → Wordle

## 1. What it does

Connects every previous vision phase into one live loop in
[app.py](../../app.py)'s `render_asl_tab()`:

```
Camera (Phase 3)
 -> MediaPipe hand detection (Phase 4)
 -> draw landmarks on the displayed frame (Phase 4)
 -> normalize + flatten to a 63-value feature vector (Phase 5)
 -> Random Forest prediction: (letter, confidence) (Phase 6)
 -> LetterStabilizer.update(): accept a letter, or not yet (Phase 7)
 -> _add_letter(letter) -- the SAME function keyboard mode uses
 -> Wordle engine (unchanged, doesn't know ASL exists)
```

## 2. The one rule that makes this safe

**ASL, voice, and keyboard all call the exact same `_add_letter()` /
`_submit_word()` functions in `app.py`.** No input mode has its own
copy of "add a letter" or "submit a guess" logic. This is what
guarantees keyboard, ASL, and (later) voice can never disagree about
what counts as a valid guess or how it's scored — see
`docs/learning/01_project_architecture.md`'s architecture rule and
`docs/learning/08_wordle_engine.md`'s "single source of truth" point.

## 3. Two Streamlit problems this integration had to solve

Both are documented in detail elsewhere; summarized here because they
were solved *specifically* to make this integration work:

1. **Whole-page flicker** (`docs/learning/02_opencv.md` §3.3) — fixed
   by putting the entire camera/recognition loop inside
   `@st.fragment(run_every=...)`, so only that small region re-renders
   on each tick, not the Wordle grid/keyboard/buttons.
2. **`scope="fragment"` crash** — once Backspace/Clear/Submit buttons
   were added *outside* the fragment (so clicking them updates the
   grid immediately), the fragment could also be entered via an
   ordinary full-page rerun, where requesting a fragment-scoped rerun
   is invalid. Switching from a manual `st.rerun(scope="fragment")`
   loop to `run_every` (Streamlit's own scheduler) removed the need to
   request that scope manually at all, which removed the crash.

## 4. Two different rerun "scopes," used on purpose

- **Fragment-scoped** (automatic, via `run_every`): the routine,
  many-times-a-second "read a frame, maybe predict" tick. Doesn't
  touch anything outside the camera UI.
- **App-scoped** (`st.rerun()`, no arguments): explicitly requested
  the moment a letter is actually **accepted** into the guess. This is
  what makes the new letter show up in the Wordle grid immediately —
  the grid lives outside the fragment, so only a full-page rerun
  refreshes it. This happens once per accepted letter (every few
  seconds), not every tick, so it doesn't reintroduce the flicker
  problem it looks like it should.

## 5. Never-auto-submit, still enforced

Exactly like keyboard mode, ASL mode only ever calls `_add_letter()`
as letters are recognized — never `_submit_word()`. Reaching 5 letters
does not submit the guess; `Submit ✅` is a separate, explicit button
(also shared with keyboard mode) that the user must click themselves.

## 6. Honest fallbacks, layered

Three independent things can be missing, and each degrades
separately rather than crashing:

| Missing | What still works | What doesn't |
|---|---|---|
| Camera | Everything except ASL/Voice video | ASL preview, recognition |
| `hand_landmarker.task` | Camera preview | Hand skeleton, recognition |
| `models/asl_classifier.pkl` | Camera + hand skeleton | Letter recognition only |

Keyboard mode is entirely unaffected by any of these, always — that's
the whole point of the architecture boundary.

## 7. Likely viva questions

- **"How does a recognized ASL letter end up in the Wordle grid?"** —
  `_add_letter()` is called (same function keyboard mode calls), which
  appends to `st.session_state.current_guess`; since that's a
  full-page-rerun-triggering event, the grid outside the fragment
  redraws immediately with the new letter.
- **"Why can Backspace live outside the fragment but the camera can't?"**
  — Backspace's effect (removing one letter) is a discrete action that
  should immediately affect the whole page; the camera's effect (a new
  frame) happens many times a second and should NOT touch the whole
  page each time — different update frequencies need different rerun
  scopes.
- **"What guarantees ASL can't cheat the Wordle rules?"** — It never
  calls `wordle_engine` directly; it can only ever call the same
  `_add_letter`/`_submit_word` funnel every other input mode uses.

# 01 — Project Architecture

## 1. What this document covers

The big picture of SignWordle before you read any code: what each
piece does, how data flows between them, and *why* the project is
split up the way it is. Read this first; every other note in
`docs/learning/` zooms into one box from the diagram below.

## 2. The core idea

SignWordle is one Wordle game with **three interchangeable input
methods** (keyboard, ASL hand signs, voice) that all produce the same
thing: a guessed word, letter by letter or all at once. Whichever
method produces the guess, it is handed to the *same* Wordle engine.

```
USER
 │
 ├── Camera ──> MediaPipe ──> Landmarks ──> Features ──> ASL Classifier
 │                                                              │
 ├── Voice ──> Whisper ──> Text ──> Candidate Word ────────────┤
 │                                                              │
 └── Keyboard ─────────────────────────────────────────────────┤
                                                                ↓
                                                         Input Controller
                                                                ↓
                                                         Wordle Engine
                                                                ↓
                                                           Game State
                                                                ↓
                                                                UI
```

## 3. Components

| Component | Folder | Responsibility | Knows about Wordle rules? |
|---|---|---|---|
| Game engine | `src/game/` | Word selection, guess validation, green/yellow/gray scoring, win/lose, restart | Yes — this is the *only* place that does |
| Vision | `src/vision/` | Webcam capture, MediaPipe hand landmarks, feature vectors, ASL letter classification | **No** |
| Speech | `src/speech/` | Microphone/audio → Whisper → normalized candidate word | **No** |
| UI | `src/ui/` + `app.py` | Streamlit rendering: the grid, keyboard, camera panel, voice panel, buttons | No — just displays state and forwards user actions |
| Config | `config.py` | Shared constants (paths, thresholds, word length) so they aren't hard-coded in five places | — |

## 4. Data flow, in words

1. The user picks an input mode (ASL / Voice / Keyboard) in the UI.
2. That mode produces **candidate letters or a candidate word** —
   never a "guess result". Vision code outputs a letter and a
   confidence. Speech code outputs a transcribed word. Keyboard is
   already exact text.
3. The UI accumulates letters into a "current guess" string (e.g.
   `APP__`), same as any Wordle clone.
4. Only on an **explicit Submit** does the current guess get handed to
   `wordle_engine.py`, which returns structured per-letter feedback
   (`correct` / `present` / `absent`).
5. The UI paints the grid from that feedback and updates the attempt
   counter / win-lose state.

## 5. Why split it this way? (the actual HCI/software argument)

- **Separation of concerns.** If ASL recognition is buggy, the
  keyboard game still works, because the engine never imports
  anything from `src/vision`. This is also why "Keyboard Demo Mode"
  (section 18 of the spec) is possible at all — it's not a special
  case, it's just... not calling the vision/speech modules.
- **Testability.** `wordle_engine.py` can be unit-tested with plain
  strings, no camera, no model, no Streamlit — see
  `docs/learning/08_wordle_engine.md`.
- **Explainability for the viva.** You can point at one file and say
  exactly what it's responsible for, instead of one 800-line
  `app.py` doing everything.
- **Multimodal consistency (HCI principle).** The spec explicitly
  requires ASL, voice, and keyboard to "feed the same Wordle engine."
  Architecturally that's only true if recognition code contains *no*
  Wordle logic — otherwise you'd risk keyboard and ASL scoring guesses
  differently.

## 6. UI vs game logic vs CV vs ML vs speech — the boundary rule

> **Rule:** `src/vision/` and `src/speech/` are only allowed to
> produce **letters or words + a confidence**. They must never import
> `wordle_engine`. `wordle_engine.py` must never import
> `cv2`, `mediapipe`, `whisper`, or `streamlit`.

If you're ever unsure which file something belongs in, ask: "does this
code care what a *correct Wordle answer* is?" If no → vision/speech. If
yes → game. If it's about pixels on screen → UI.

## 7. Input data / Output data (per component, high level)

- **Vision**: input = webcam frame (`numpy` array, BGR); output =
  `(letter: str, confidence: float)` or `None`.
- **Speech**: input = recorded audio; output = `(candidate_word: str)`
  or `None`.
- **Game engine**: input = a 5-letter guess string; output = a list of
  5 feedback tags + updated game state.
- **UI**: input = game state + recognition state; output = rendered
  Streamlit widgets (no business logic).

## 8. Important files (so far)

- `app.py` — Streamlit entry point, currently just the page shell.
- `config.py` — shared constants (word length, attempts, paths,
  thresholds) used across every module so numbers aren't duplicated.

## 9. Common errors at this stage

- Running `streamlit run app.py` from the wrong directory — the venv
  must be active or you must call `venv/Scripts/python.exe -m streamlit`
  directly.
- Forgetting to activate the virtual environment before `pip install`,
  which installs packages globally instead.
- **Streamlit's "one script rerun per click" model bit us for real in
  Phase 2**: `app.py` originally drew the grid and the attempt counter
  *before* the on-screen keyboard's buttons (whose click handlers
  mutate `st.session_state`). Streamlit reruns the entire script
  top-to-bottom on every click, so the grid was always drawn with
  *last* click's state — every button press looked like it "didn't
  register" because the visible guess/attempt count was always one
  click behind. The fix: reserve the grid/attempts/status positions
  early with `st.empty()`, let every input widget run and mutate state,
  and only *then* fill those placeholders. General rule for Streamlit:
  **anything whose display depends on a widget's callback must be
  rendered after that widget is processed, not before** — use
  `st.empty()` placeholders if you need it to *appear* earlier on the
  page than the widget that feeds it.

## 10. Likely viva questions

- *"Why isn't this a Flask/Django app with a database?"* — The spec
  deliberately avoids a backend/DB/auth because the assignment is
  about demonstrating HCI + CV/ML pipelines, not building
  infrastructure. Streamlit gives a GUI with a single Python process.
- *"Why doesn't the ASL code know about Wordle at all?"* — So the same
  classifier could be reused in a totally different game, and so bugs
  in one layer can't corrupt another. This is basic separation of
  concerns / single-responsibility.
- *"What happens if the camera or model fails?"* — The UI is expected
  to detect that and fall back to keyboard mode, clearly labelled
  (see `docs/learning/10_hci_design.md` and the Demo/Fallback
  requirement in the spec).

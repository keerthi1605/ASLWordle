# SignWordle 🤟

**Learn ASL. Play Wordle. Interact Naturally.**

> Status: all three input modes are wired end-to-end. ASL: hold a
> static letter to the camera and it's stabilized and typed into the
> grid live (**you need to record your own training data first** — see
> §12 Dataset, nobody's hand signs are pre-loaded, on principle). Voice:
> say a word, Whisper transcribes it, you confirm before it's
> submitted. Keyboard is fully playable with zero setup. This README
> grows as each phase lands — sections marked `(planned)` aren't built
> yet.

## 1. Project Overview

SignWordle is a Wordle-inspired word-guessing game built for an
HCI (Human-Computer Interaction) course project. It demonstrates
**multimodal interaction**: the same underlying Wordle game can be
played by typing on a keyboard, by fingerspelling letters in
**ASL (American Sign Language)** in front of a webcam, or by speaking
a word aloud.

## 2. Problem Statement

Most word games only accept one input method (typing). People who are
learning ASL have few tools that combine language games with sign
practice, and few demos show how computer vision, speech recognition,
and traditional input can all drive one shared application.

## 3. Motivation

- Make ASL fingerspelling practice fun and goal-directed instead of
  rote flashcards.
- Demonstrate core HCI principles (visibility, feedback, error
  prevention, user control, accessibility) with a concrete, playable
  system rather than only in theory.
- Show a full, honest computer-vision + ML pipeline end-to-end
  (camera → landmarks → features → classifier → application logic)
  at a scale that's understandable in a short viva.

## 4. Objectives

1. A fully working Wordle engine, independent of any input method.
2. A keyboard input mode that works with zero dependencies on CV/ML.
3. A webcam-based ASL fingerspelling input mode using MediaPipe hand
   landmarks and a scikit-learn classifier (not a black-box CNN).
4. A voice input mode using Whisper, with mandatory user confirmation.
5. Clear, always-visible system state so the user always knows what
   the system detected and why.

## 5. Features

- [x] Streamlit UI shell, light/dark theme friendly
- [x] Wordle engine (5 letters, 6 attempts, duplicate-letter-aware
      green/yellow/gray feedback), unit tested
- [x] Keyboard input mode — on-screen buttons AND physical keyboard typing
- [x] Webcam capture (mirrored, start/stop, honest error handling)
- [x] MediaPipe hand landmark visualization (live 21-point overlay)
- [x] Landmark → feature vector normalization (translation + scale invariant, 63 values)
- [x] Random Forest ASL letter classifier (train your own — see §12/§13)
- [x] Temporal stabilization (rolling majority vote, no repeated/spammed letters)
- [x] ASL → Wordle integration — live camera recognition types into the grid
- [x] Whisper voice input — record → transcribe → confirm/reject → submit
- [ ] ASL practice mode — Phase 10
- [ ] Stats, hints (optional, only after core works) — later

## 6. Architecture

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

Design rule: `src/vision/` and `src/speech/` never know Wordle rules.
They only ever produce **letters or words**. `src/game/wordle_engine.py`
never knows about cameras, models, or microphones. This separation is
what makes each piece independently testable and explainable.

Folder layout:

```
signwordle/
├── app.py                  # Streamlit entry point
├── config.py                # Shared constants/paths
├── requirements.txt
├── src/
│   ├── game/                # Wordle rules — no CV/ML/UI knowledge
│   ├── vision/               # Camera, MediaPipe, features, ASL classifier,
│   │                         #   temporal stabilizer — no Wordle knowledge
│   ├── speech/                # Whisper voice pipeline (record, transcribe,
│   │                         #   normalize, extract candidate) — no Wordle knowledge
│   └── ui/                    # Streamlit rendering helpers
├── models/                  # hand_landmarker.task, asl_classifier.pkl (not in git)
├── data/                     # Word list, your recorded asl_landmarks.csv, stats
├── scripts/                  # collect_data.py (record your own ASL data), train_asl.py
├── tests/                    # Unit tests
└── docs/learning/            # Beginner-friendly notes + viva prep
```

## 7. Tech Stack

Python, Streamlit, OpenCV, MediaPipe, NumPy, scikit-learn, OpenAI
Whisper (+ PyTorch, sounddevice), Pillow. No backend server, database,
auth, Docker, or cloud deploy — this is intentionally a single local
Streamlit process.

## 8. Installation

```bash
python -m venv venv
```

Activate it, then:

```bash
pip install -r requirements.txt
```

(Dependencies are added incrementally as phases are built — see
`requirements.txt` comments for which phase needs what.)

**Hand-tracking model** (needed for ASL mode's live landmark overlay,
Phase 4+): MediaPipe's Tasks API needs a model file that isn't bundled
in the pip package. Download Google's official one:

```bash
curl -L -o models/hand_landmarker.task "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
```

If this file is missing, the app doesn't crash — ASL mode falls back
to a plain camera preview with a visible warning. See
[docs/learning/03_mediapipe.md](docs/learning/03_mediapipe.md).

## 9. Running

```bash
streamlit run app.py
```

This opens the app at `http://localhost:8501`.

## 10. Testing

```bash
venv/Scripts/python.exe -m unittest tests.test_wordle_engine -v
```

```bash
venv/Scripts/python.exe -m unittest discover -s tests -v
```

80 tests total. Wordle-engine tests (18) cover valid/invalid guesses,
exact-position matches, wrong-position matches, duplicate-letter edge
cases (both "guess repeats a letter the target only has once" and
"target repeats a letter the guess only has once"), win, lose,
restart, and keyboard letter-status coloring. Camera tests (7) and
hand-detector tests (6) check error handling without hardware, plus
real start/stop/frame-shape/mirroring/detection checks that **skip
automatically** (not fail) on a machine with no webcam or no
downloaded model. Landmark/feature tests (16) check hand topology plus
feature-vector dimensions, translation invariance, scale invariance,
and the divide-by-zero guard. Stabilizer tests (9) check the rolling
majority vote directly — holding a letter accepts it once, releasing
and re-showing it accepts it again, low-confidence/noisy frames don't
count. ASL-recognizer tests (6) train a tiny throwaway classifier on
synthetic data (not your real model) to check prediction/confidence
and the "no model yet" fallback. Whisper tests (18) check text
normalization and candidate extraction with plain strings, plus real
model-loading/transcription checks that skip gracefully if Whisper
can't load. Everything except the camera/microphone/model hardware
tests needs no hardware at all. Run from the project root (no pytest
needed — plain `unittest`).

**Manual camera test**: run the app, click **✋ ASL**, then
**▶ Start Camera** — you should see your own mirrored live video with
a "🟢 Live, ~N FPS" status line. See
[docs/learning/02_opencv.md](docs/learning/02_opencv.md) for
troubleshooting if it doesn't.

**Manual hand-tracking test**: with the camera running in ASL mode,
hold your hand up in view — you should see a 21-point skeleton
(colored dots + lines) tracking your fingers in real time, and the
status line should say "✋ 1 hand(s) — Right" (or Left). See
[docs/learning/04_hand_landmarks.md](docs/learning/04_hand_landmarks.md).

**Manual ASL recognition test** (after §12/§13 below): with the camera
running, hold a static ASL letter steady for a few seconds — the
status line should show "Predicted: X (NN%) — hold steady…", then
"✅ Letter accepted: X", and that letter should appear in the Wordle
grid immediately. Backspace/Clear/Submit work the same as keyboard
mode. See [docs/learning/07_realtime_prediction.md](docs/learning/07_realtime_prediction.md).

**Manual voice test**: click **🎤 Voice** (first click loads the
Whisper model, a few seconds), then **🎤 Record** and say a 5-letter
word within ~4 seconds. You should see "You said: ..." with exactly
what Whisper heard, and — if it found a valid word — a candidate plus
**[Confirm Guess] [Try Again]**; nothing submits until you click
Confirm. If it misheard or you said something else, it says so
honestly instead of guessing. See
[docs/learning/09_whisper.md](docs/learning/09_whisper.md).

## 11. Wordle Algorithm

Scoring is a two-pass comparison, not a naive `letter in target`
check, specifically to handle duplicate letters correctly. See
[docs/learning/08_wordle_engine.md](docs/learning/08_wordle_engine.md)
for the full explanation with a worked example. Word data (the real
Wordle word lists) is covered next in §12a.

## 12. Dataset

### 12a. Word list

`data/wordle_answers.txt` (2,315 words) and
`data/wordle_allowed_guesses.txt` (10,657 words) are the **real
original-Wordle word lists** — before the NYT acquired Wordle, the game
shipped these two arrays directly in its public client-side JavaScript.
They've since been mirrored on GitHub by the community (this project
uses [cfreshman's gists](https://gist.github.com/cfreshman/a03ef2cba789d8cf00c08f767e0fad7b));
**this is not an NYT-licensed dataset**, just a community-preserved
copy of code Wordle once shipped publicly — stated honestly rather than
implied to be "official." Verified on import: 2,315 + 10,657 = 12,972
words total, matching the real game's well-documented word count
exactly. [src/game/word_list.py](src/game/word_list.py) loads both at
startup: `WORDS` (the 2,315 answers) is the only pool a target is ever
picked from; `is_valid_word()` accepts both files combined — same
relationship the real game has (e.g. "AAHED" is accepted as a guess,
but never chosen as the answer).

### 12b. ASL landmark data

**There is no downloaded ASL dataset in this project — on principle.**
A landmark-level ASL alphabet dataset (already-extracted 21-point
coordinates, not raw photos) isn't something we could find and verify
the source/license of, and this project's rules say never fabricate
one. Instead, [scripts/collect_data.py](scripts/collect_data.py) lets
you record your own samples with your own webcam:

```bash
venv/Scripts/python.exe scripts/collect_data.py
```

A plain OpenCV window (not part of the Streamlit app) walks you
through each letter (A-Y, skipping J/Z — see §14 ASL pipeline);
`SPACE` captures a sample, `N` skips to the next letter, `ESC` stops.
Move your hand slightly between captures (angle/distance/position) —
that variation is what actually makes recognition tolerant of
imperfect hand positions later, not any code trick. Aim for 30+
samples per letter. Samples append to `data/asl_landmarks.csv` (63
feature columns + a label column) — safe to re-run later to add more.

## 13. Training

```bash
venv/Scripts/python.exe scripts/train_asl.py
```

Loads `data/asl_landmarks.csv`, splits 80/20 train/test, trains a
`RandomForestClassifier`, and **prints real, measured test accuracy
and a per-letter precision/recall report** — never a fabricated
number. Saves to `models/asl_classifier.pkl`. **Restart the Streamlit
app afterward** to pick up the new model (it's loaded once at
startup). See [docs/learning/06_asl_classifier.md](docs/learning/06_asl_classifier.md).

## 14. ASL Pipeline

```
Camera → MediaPipe → 21 landmarks → normalize (translation + scale)
 → 63-value feature vector → Random Forest → (letter, confidence)
 → rolling majority-vote stabilizer → accepted letter
 → same _add_letter() keyboard mode uses → Wordle grid
```

J and Z are unsupported: both require motion in real ASL (the hand
draws a shape over time), which a single-frame landmark classifier
structurally cannot represent. This is a documented limitation, not a
bug. Full details, including a real measured Streamlit performance
constraint that shaped the stabilizer's tuning:
[docs/learning/06_asl_classifier.md](docs/learning/06_asl_classifier.md),
[07_realtime_prediction.md](docs/learning/07_realtime_prediction.md),
[11_integration.md](docs/learning/11_integration.md).

## 15. Voice Pipeline

```
🎤 Record (4s) → WhisperService.transcribe() → raw text
 → normalize_text() (uppercase, strip punctuation)
 → extract_candidate_word() (find one 5-letter alphabetic token)
 → "You said: ..." always shown, honestly
 → if found: [Confirm Guess] [Try Again]  (never auto-submitted)
 → Confirm → same _submit_word() keyboard/ASL use → Wordle engine
```

No internet-hosted speech API is used — Whisper runs entirely locally
(CPU by default, GPU automatically if `torch.cuda.is_available()`).
Whisper can and does mishear audio; this project never disguises that
— see [docs/learning/09_whisper.md](docs/learning/09_whisper.md) for a
real (not hypothetical) example encountered while building this, and
the full explanation of audio, sample rates, and normalization.

---

Sections 16-17 (HCI principles, Limitations, Future improvements) will
be filled in as those phases are built.

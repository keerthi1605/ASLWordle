# 09 — Whisper Voice Input

## 1. What it does

Lets you say a word out loud instead of typing or signing it: records
a few seconds of microphone audio, transcribes it with OpenAI's
Whisper, pulls out a single word-shaped candidate, and — only after
you explicitly confirm — submits it through the exact same funnel
keyboard and ASL use. See
[whisper_service.py](../../src/speech/whisper_service.py) and the
Voice branch in [app.py](../../app.py).

## 2. Audio, in the terms this code actually uses

A microphone measures air pressure many times per second and outputs
a stream of numbers — the **sample rate** is how many measurements per
second (Whisper expects **16,000 per second**, "16kHz"). `record_audio()`
uses `sounddevice.rec()` to capture exactly that: 4 seconds × 16,000
samples/sec = 64,000 floating-point numbers, each roughly in `[-1, 1]`
representing the microphone's signal at that instant. That's the
entire "audio file" — just a NumPy array, no format/codec involved
until you'd want to save it to disk (which this project never does).

## 3. Speech-to-text and what Whisper actually is

**Speech-to-text** = converting that stream of numbers into the words
a person said. **Whisper** is OpenAI's pretrained model for exactly
this — like MediaPipe's hand model (Phase 4), it's a neural network
someone else trained on a huge dataset, that this project uses
as-is, not trains itself. `whisper.load_model("base")` downloads
(once, cached to disk after) and loads a ~74-million-parameter model;
`model.transcribe(audio)` runs it on one audio array and returns the
text it heard.

## 4. Model inference: loading once, CPU vs. GPU

Loading the model is the expensive part (a few real seconds — measured
directly: ~9s on this project's dev machine, first run) — so
`WhisperService` loads it exactly **once** and is kept in
`st.session_state`, the same pattern as `HandDetector` and
`ASLRecognizer`. `app.py` also defers even that one-time cost until
the user actually opens **🎤 Voice** for the first time
(`_ensure_whisper_loaded()`), so a keyboard-only player never pays it.

**CPU fallback / GPU support**, both required by the spec:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model(model_size, device=device)
```

On this dev machine, `torch.cuda.is_available()` is `False` (CPU-only
PyTorch build) — transcription still works, just slower than it would
on a GPU. Whisper's `"base"` size was picked specifically because it's
"small/fast enough for CPU-only laptops" (see `config.py`) — a bigger
model would transcribe more accurately but too slowly without a GPU.

## 5. Transcription, normalization, candidate extraction — three separate steps

Deliberately three different functions, each with one job:

1. **`WhisperService.transcribe()`** — returns *exactly* what Whisper
   heard, unmodified (e.g. `" Apple"` — Whisper often adds a leading
   space and doesn't uppercase).
2. **`normalize_text()`** — strips punctuation, upper-cases, collapses
   whitespace: `"Apple."` → `"APPLE"`.
3. **`extract_candidate_word()`** — looks for one exactly-5-letter
   alphabetic token in the normalized text. Handles both a clean
   single-word answer and a full sentence ("the word is apple") by
   splitting on whitespace and checking each token. Returns `None` —
   never a guess — if nothing of the right length is found.

Keeping these separate means each one is independently testable (see
`tests/test_whisper_service.py`) and the failure mode is always clear:
did Whisper mishear, or did it hear a real word that just isn't 5
letters?

## 6. The full flow, and the confirmation gate

```
🎤 Record button
 -> record_audio(4s)              -- one blocking action, no loop
 -> transcribe()                  -- raw text, exactly what was heard
 -> normalize_text() + extract_candidate_word()
 -> UI shows "You said: <raw text>" always (never hidden)
 -> if a 5-letter candidate was found:
      show it + [Confirm Guess] [Try Again]
    else:
      show an honest "couldn't find a clear word" message + [Record] again
 -> Confirm Guess -> _submit_word(candidate)  -- SAME funnel keyboard/ASL use
```

**Never auto-submits.** Reaching a valid-looking candidate does not
submit it — `[Confirm Guess]` is a separate, required click, exactly
like ASL never auto-submits on 5 accepted letters and keyboard mode
never auto-submits on the 5th keystroke. This project treats "the
system thinks it understood you" and "the system will act on it" as
two different, deliberately separated steps everywhere.

## 7. Limitations of speech recognition — stated honestly

- **Whisper can mishear.** Tested directly during development: reading
  a synthesized "apple" from a clean audio file transcribed correctly
  to "APPLE", but a noisier real-microphone attempt once transcribed
  as "Not to kidding" — a real, unedited example of speech recognition
  failing on ambiguous/quiet audio. The app's response to that isn't
  to guess harder; it shows exactly what was heard and says plainly
  that no valid word was found, then lets you try again.
- **It doesn't understand intent**, only sound-to-text. If you say "my
  guess is apple", Whisper transcribes the whole sentence; extraction
  only works because it happens to contain one 5-letter word. A
  differently-phrased sentence with two different-length words meant
  as the guess would not be resolved by guessing which one you meant —
  this project never does that kind of guessing.
- **Background noise, accents, and unclear speech** all reduce
  accuracy — this is inherent to the model, not something this
  integration papers over.
- **This project never claims Whisper "understood" anything** — every
  UI message describes what was *heard* ("You said: ...") not what
  was *meant*.

## 8. Input / Output

- Input: one mono float32 NumPy array at 16kHz (from the mic, or in
  principle any array of that shape).
- Output: `transcribe()` → `(success, raw_text, message)`;
  `extract_candidate_word()` → a 5-letter uppercase string or `None`.
  Never fabricates a word when nothing valid was said.

## 9. Connection to other components

Contains zero Wordle knowledge, same boundary rule as the vision
layer. `app.py`'s `_confirm_voice_guess()` is the only place that
connects "a candidate was confirmed" to the game, and it does so by
calling `_submit_word()` — the identical function keyboard and ASL
call — so voice guesses are scored by the exact same
`wordle_engine.py`, no separate rules.

## 10. Common errors

- **Assuming `sd.rec()` blocks the whole app forever** — it blocks
  only for the requested duration (here, 4 seconds); wrapped in
  `st.spinner()` so the user sees *why* the app is briefly
  unresponsive instead of thinking it's frozen.
- **Skipping the normalization step** — comparing Whisper's raw
  `" Apple."` directly against an uppercase word list would never
  match; always normalize first.
- **Forgetting a missing microphone** — `record_audio()` catches
  exceptions and returns an honest `(False, None, message)` instead of
  crashing the app, the same pattern as `Camera.start()`.

## 11. Likely viva questions

- **"Why record for a fixed duration instead of detecting when
  speech stops?"** — Simplicity: voice-activity detection is a whole
  extra component; a fixed ~4s window is enough for one word and keeps
  the pipeline easy to explain and test.
- **"What happens if Whisper mishears the word?"** — The raw
  transcript is always shown ("You said: ..."), and if no 5-letter
  word can be extracted, the app says so honestly and lets the user
  try again or switch to keyboard — it never guesses a "close enough"
  word.
- **"Why is there a separate Confirm step instead of submitting
  immediately?"** — Same error-prevention principle as every other
  input mode in this project: the system's best guess at what you said
  is not treated as your final answer until you explicitly say so.
- **"How does this handle GPU vs. CPU?"** — Checks
  `torch.cuda.is_available()` once at load time and passes that as
  Whisper's `device`; works either way, just faster with a GPU.
- **"Could voice and ASL ever disagree about scoring the same guess?"**
  — No — both ultimately call the identical `_submit_word()`, which
  calls `wordle_engine.submit_guess()`; there is exactly one place
  Wordle rules are implemented.

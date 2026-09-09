"""
SignWordle - entry point.

Three input modes -- Keyboard, ASL (webcam), Voice (microphone) -- all
feed the exact same Wordle engine through the same _add_letter()/
_submit_word() funnel. Keyboard mode has zero dependency on the
camera/ML/speech stacks, by design; ASL and Voice degrade honestly
(clear warning, never a fake prediction) if their models/hardware
aren't available.

IMPORTANT layout note (read this before touching render order below):
Streamlit reruns this whole script top-to-bottom on every click. If we
draw the grid/attempt-counter *before* the widgets whose callbacks
mutate game state, the drawing uses last run's stale state -- the grid
and "N / 6 attempts" appear to lag one click behind every button press
(this was a real bug: see the `st.empty()` placeholders below). The
fix is to reserve the grid/attempts/status positions early with
`st.empty()`, process every input widget (which mutates
st.session_state), and only then fill those placeholders -- so what's
drawn always reflects this run's final state.

Run with:
    streamlit run app.py
"""

import time

import cv2
import streamlit as st

from config import CONFIDENCE_THRESHOLD
from src.game.wordle_engine import WordleEngine
from src.ui.game_ui import render_attempt_counter, render_grid, render_keyboard, render_status_message
from src.vision.asl_recognizer import ASLRecognizer
from src.vision.camera import Camera
from src.vision.hand_detector import HandDetector, draw_hand_landmarks
from src.vision.landmark_utils import landmarks_to_feature_vector
from src.vision.stabilizer import LetterStabilizer
from src.speech.whisper_service import WhisperService, extract_candidate_word

st.set_page_config(page_title="SignWordle", page_icon="🤟", layout="centered")

st.markdown(
    """
    <style>
    .signwordle-title {
        text-align: center; font-weight: 800; letter-spacing: 0.15em;
        font-size: 2.6rem; margin-bottom: 0;
    }
    .signwordle-subtitle {
        text-align: center; opacity: 0.75; margin-top: 0.25rem; margin-bottom: 1.25rem;
    }
    .current-guess {
        text-align: center; font-size: 1.1rem; font-weight: 600;
        letter-spacing: 0.3em; margin: 0.75rem 0;
    }
    div.stButton > button { font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _init_state() -> None:
    if "engine" not in st.session_state:
        st.session_state.engine = WordleEngine()
    if "current_guess" not in st.session_state:
        st.session_state.current_guess = ""
    if "message" not in st.session_state:
        st.session_state.message = ""
    if "input_mode" not in st.session_state:
        st.session_state.input_mode = "Keyboard"
    if "camera" not in st.session_state:
        st.session_state.camera = Camera(mirror=True)
    if "camera_running" not in st.session_state:
        st.session_state.camera_running = False
    if "camera_message" not in st.session_state:
        st.session_state.camera_message = ""
    if "camera_last_frame_time" not in st.session_state:
        st.session_state.camera_last_frame_time = None
    if "hand_detector" not in st.session_state:
        # Loaded once and reused; if the model file is missing this
        # degrades to "camera only, no landmarks" instead of crashing
        # the whole app (see docs/learning/03_mediapipe.md).
        try:
            st.session_state.hand_detector = HandDetector()
            st.session_state.hand_detector_error = ""
        except FileNotFoundError as exc:
            st.session_state.hand_detector = None
            st.session_state.hand_detector_error = str(exc)
    if "asl_recognizer" not in st.session_state:
        # Same honest-fallback pattern as the hand detector above: if
        # nobody has run scripts/collect_data.py + scripts/train_asl.py
        # yet, ASL mode still shows the camera + landmarks, just with
        # no letter recognition, rather than crashing or faking one.
        st.session_state.asl_recognizer = ASLRecognizer()
    if "letter_stabilizer" not in st.session_state:
        # Persists across every camera-loop tick (that's the whole
        # point -- see src/vision/stabilizer.py) so it must live in
        # session_state, not be recreated inside the fragment.
        st.session_state.letter_stabilizer = LetterStabilizer()
    # whisper_service is intentionally NOT loaded here -- loading it
    # costs a few real seconds (see docs/learning/09_whisper.md), so a
    # keyboard-only player shouldn't pay that cost at all. It's loaded
    # lazily the first time Voice mode is opened (_ensure_whisper_loaded).
    if "voice_transcript" not in st.session_state:
        st.session_state.voice_transcript = ""
    if "voice_candidate" not in st.session_state:
        st.session_state.voice_candidate = None
    if "voice_message" not in st.session_state:
        st.session_state.voice_message = ""


def _ensure_whisper_loaded() -> None:
    if "whisper_service" not in st.session_state:
        with st.spinner("Loading speech recognition model (first time only)…"):
            st.session_state.whisper_service = WhisperService()


def _add_letter(letter: str) -> None:
    engine = st.session_state.engine
    if len(st.session_state.current_guess) < engine.word_length:
        st.session_state.current_guess += letter
        st.session_state.message = ""


def _backspace() -> None:
    st.session_state.current_guess = st.session_state.current_guess[:-1]
    st.session_state.message = ""


def _clear() -> None:
    st.session_state.current_guess = ""
    st.session_state.message = ""


def _submit_word(word: str) -> None:
    """Validate + submit any 5-letter candidate to the engine, from
    ANY input source (on-screen keyboard, physical keyboard, and later
    ASL/voice). This is the single funnel into wordle_engine, per the
    architecture rule in docs/learning/01_project_architecture.md."""
    engine = st.session_state.engine
    word = word.strip().upper()
    if len(word) != engine.word_length:
        st.session_state.message = f"Please enter a {engine.word_length}-letter word."
        st.session_state.current_guess = word[: engine.word_length]
        return
    result = engine.submit_guess(word)
    if not result["accepted"]:
        st.session_state.message = result["reason"]
        st.session_state.current_guess = word  # keep it visible/editable
    else:
        st.session_state.message = ""
        st.session_state.current_guess = ""  # reset -> ready for the next row


def _restart() -> None:
    st.session_state.engine.reset()
    st.session_state.current_guess = ""
    st.session_state.message = ""
    st.session_state.letter_stabilizer.reset()


def _start_camera() -> None:
    ok, message = st.session_state.camera.start()
    st.session_state.camera_running = ok
    st.session_state.camera_message = "" if ok else message
    st.session_state.camera_last_frame_time = None


def _stop_camera() -> None:
    st.session_state.camera.stop()
    st.session_state.camera_running = False
    st.session_state.camera_message = ""


_VOICE_RECORD_SECONDS = 4.0


def _record_and_transcribe() -> None:
    """One blocking action: record a few seconds of audio, transcribe
    it, and try to pull out a 5-letter candidate. Runs once per button
    click -- unlike ASL mode, voice input has no continuous loop, so
    none of the fragment/rerun machinery above is needed here."""
    service = st.session_state.whisper_service
    with st.spinner(f"🎤 Recording for {_VOICE_RECORD_SECONDS:.0f}s — speak now…"):
        ok, audio, message = service.record_audio(_VOICE_RECORD_SECONDS)
    if not ok:
        st.session_state.voice_message = message
        st.session_state.voice_transcript = ""
        st.session_state.voice_candidate = None
        return

    with st.spinner("Transcribing…"):
        ok, raw_text, message = service.transcribe(audio)
    if not ok:
        st.session_state.voice_message = message
        st.session_state.voice_transcript = ""
        st.session_state.voice_candidate = None
        return

    st.session_state.voice_transcript = raw_text
    st.session_state.voice_candidate = extract_candidate_word(raw_text, st.session_state.engine.word_length)
    st.session_state.voice_message = (
        "" if st.session_state.voice_candidate
        else f"Couldn't find a clear {st.session_state.engine.word_length}-letter word in that — try again."
    )


def _confirm_voice_guess() -> None:
    """Explicit user confirmation -> the same shared funnel every
    other input mode uses. Never called automatically from a raw
    transcription -- see docs/learning/09_whisper.md."""
    candidate = st.session_state.voice_candidate
    if candidate:
        _submit_word(candidate)
    st.session_state.voice_transcript = ""
    st.session_state.voice_candidate = None
    st.session_state.voice_message = ""


def _discard_voice_guess() -> None:
    st.session_state.voice_transcript = ""
    st.session_state.voice_candidate = None
    st.session_state.voice_message = ""


# Camera "video" tick rate. Deliberately 10-15 FPS, not 30: fewer image
# swaps per second means less visible flicker, a smaller measured FPS
# number we can actually hit consistently, and less CPU spent right
# when MediaPipe detection is also running per frame.
_CAMERA_TICK_SECONDS = 0.08  # ~12-13 FPS

# JPEG quality for the frames we SEND TO THE BROWSER (not the frame we
# run MediaPipe detection on -- that always uses the full, uncompressed
# array, so this never affects detection accuracy). Streamlit's own
# st.image() re-encodes NumPy arrays as JPEG at quality=100 by default,
# which produces needlessly large frames for a live-updating preview;
# encoding it ourselves at a lower, still visually-fine quality makes
# each frame smaller and faster to transfer/decode, which is the main
# lever available for reducing image-swap flicker without threads,
# WebRTC, or other new infrastructure.
_DISPLAY_JPEG_QUALITY = 75

# Displayed (not detection-affecting -- see _encode_frame_for_display)
# width of the camera preview while ASL mode is active. Deliberately
# smaller than the full column width: at full width the video pushes
# the "Predicted: X (NN%)" line below the fold, so you have to scroll
# to see it while your hands are busy signing. Shrinking the preview
# keeps the accuracy/letter readout visible at the same time.
_CAMERA_DISPLAY_WIDTH = 320


def _encode_frame_for_display(frame_rgb) -> bytes | None:
    """Compress an RGB frame to JPEG bytes for st.image(), or None if
    encoding fails (caller should fall back to the raw array)."""
    # cv2.imencode assumes BGR channel order (see docs/learning/02_opencv.md
    # on BGR vs RGB) -- convert back before encoding, or the JPEG comes
    # out with red/blue swapped.
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, _DISPLAY_JPEG_QUALITY])
    return buffer.tobytes() if ok else None


@st.fragment(run_every=_CAMERA_TICK_SECONDS)
def render_asl_tab() -> None:
    """
    Everything camera/hand-tracking related lives inside this
    `@st.fragment`. That matters a lot: without it, "live video" in
    Streamlit means calling `st.rerun()` on the WHOLE script every
    tick, which re-executes and re-draws literally everything on the
    page (title, Wordle grid, on-screen keyboard, mode buttons) that
    many times a second -- which is exactly the flicker/glitch this
    was fixed for.

    `run_every` (not a manual `time.sleep()` + `st.rerun(scope="fragment")`
    loop) is what schedules the automatic re-ticks: Streamlit's own
    runtime re-invokes just this fragment on that interval. That
    matters for a subtle reason -- `st.rerun(scope="fragment")` is
    ONLY valid to call from *within* an already-fragment-scoped rerun;
    since this function can also be entered as part of an ordinary
    full-page rerun (e.g. the Backspace/Clear/Submit buttons below are
    deliberately OUTSIDE this fragment), a manual `scope="fragment"`
    call would crash on exactly those passes. `run_every` has no such
    restriction, which is why it replaced that approach here.
    """
    st.caption(
        "✋ ASL fingerspelling: hold a static ASL letter steady until it's "
        "accepted into your guess below. J and Z aren't supported — they "
        "require motion, which a single-frame classifier can't see."
    )
    if st.session_state.hand_detector is None:
        st.warning(
            f"⚠️ Hand tracking unavailable ({st.session_state.hand_detector_error}). "
            "Camera-only preview still works below; use Keyboard mode to play."
        )
    elif not st.session_state.asl_recognizer.is_available:
        st.warning(
            "⚠️ No trained ASL model yet — the camera and hand tracking below "
            "still work, but no letters will be recognized. Run "
            "`scripts/collect_data.py` then `scripts/train_asl.py` (see README), "
            "then restart this app. Use Keyboard mode to play in the meantime."
        )
    cam_cols = st.columns(2)
    if cam_cols[0].button("▶ Start Camera", key="cam_start", use_container_width=True, disabled=st.session_state.camera_running):
        _start_camera()
    if cam_cols[1].button("■ Stop Camera", key="cam_stop", use_container_width=True, disabled=not st.session_state.camera_running):
        _stop_camera()

    camera_frame_placeholder = st.empty()
    camera_status_placeholder = st.empty()
    recognition_placeholder = st.empty()

    if st.session_state.camera_running:
        ok, frame, read_message = st.session_state.camera.read_frame()
        if ok:
            now = time.time()
            last = st.session_state.camera_last_frame_time
            fps = (1.0 / (now - last)) if last else 0.0
            st.session_state.camera_last_frame_time = now

            hands = []
            hand_status = "no hand-tracking model"
            if st.session_state.hand_detector is not None:
                hands, handedness = st.session_state.hand_detector.detect(frame)
                draw_hand_landmarks(frame, hands)  # mutates `frame` in place
                hand_status = (
                    f"✋ {len(hands)} hand(s) — {', '.join(handedness)}"
                    if hands
                    else "no hand detected"
                )

            # Detection above already ran on the full, uncompressed
            # frame -- only the copy we SEND TO THE BROWSER is
            # JPEG-compressed, so this never affects detection accuracy.
            jpeg_bytes = _encode_frame_for_display(frame)
            if jpeg_bytes is not None:
                camera_frame_placeholder.image(jpeg_bytes, width=_CAMERA_DISPLAY_WIDTH)
            else:
                camera_frame_placeholder.image(frame, channels="RGB", width=_CAMERA_DISPLAY_WIDTH)
            camera_status_placeholder.success(f"🟢 Live, ~{fps:.0f} FPS — {hand_status}")

            # --- Landmarks -> feature vector -> predicted letter ->
            # temporally-stabilized ACCEPTED letter -> the exact same
            # _add_letter() the on-screen keyboard uses. Recognition
            # code (ASLRecognizer) and stabilization (LetterStabilizer)
            # both know nothing about Wordle; this fragment is the only
            # place that connects "a letter was recognized" to the game.
            predicted_letter, confidence = None, 0.0
            game_over = st.session_state.engine.game_over
            if hands and st.session_state.asl_recognizer.is_available and not game_over:
                feature_vector = landmarks_to_feature_vector(hands[0])
                predicted_letter, confidence = st.session_state.asl_recognizer.predict(feature_vector)

            accepted_letter = None if game_over else st.session_state.letter_stabilizer.update(predicted_letter, confidence)

            if game_over:
                recognition_placeholder.info("Game over — restart to sign another word.")
            elif not st.session_state.asl_recognizer.is_available:
                recognition_placeholder.info("No trained ASL model — see the warning above.")
            elif accepted_letter:
                recognition_placeholder.success(f"✅ Letter accepted: **{accepted_letter}**")
            elif not hands:
                recognition_placeholder.info("Show an ASL letter to the camera.")
            elif predicted_letter and confidence >= CONFIDENCE_THRESHOLD:
                recognition_placeholder.info(f"Predicted: **{predicted_letter}** ({confidence:.0%}) — hold steady…")
            elif predicted_letter:
                recognition_placeholder.warning(f"Predicted: **{predicted_letter}** ({confidence:.0%}) — low confidence, adjust hand position")
            else:
                recognition_placeholder.info("…")

            # Belt-and-suspenders throttle: `run_every` is what SCHEDULES
            # ticks, but this sleep caps how fast any single tick can
            # possibly complete, regardless of scheduling edge cases
            # (e.g. this function also being entered directly by a
            # full-page rerun from the Backspace/Clear/Submit buttons
            # below, which are outside the fragment). Without it, a
            # burst of reruns can process frames far faster than
            # intended -- verified by watching the request rate spike
            # without this line.
            time.sleep(_CAMERA_TICK_SECONDS)

            if accepted_letter:
                _add_letter(accepted_letter)
                # A letter just entered the guess -- escalate to a full
                # app rerun (valid from anywhere, unlike scope="fragment")
                # so the Wordle grid and "Current Guess" line OUTSIDE
                # this fragment update immediately too. This happens
                # once per accepted letter (roughly every 1-2 seconds),
                # not every video tick, so it doesn't reintroduce the
                # whole-page flicker this fragment was built to avoid.
                st.rerun()
        else:
            _stop_camera()
            st.session_state.camera_message = read_message
            camera_status_placeholder.error(f"❌ {read_message}")
    else:
        camera_frame_placeholder.info("Camera is stopped. Click ▶ Start Camera to test your webcam.")
        if st.session_state.camera_message:
            camera_status_placeholder.error(f"❌ {st.session_state.camera_message}")


_init_state()
engine = st.session_state.engine

st.markdown('<div class="signwordle-title">SIGNWORDLE</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="signwordle-subtitle">Learn ASL. Play Wordle. Interact Naturally.</div>',
    unsafe_allow_html=True,
)

# --- Reserve visual slots now; fill them with fresh state at the end ---
grid_placeholder = st.empty()
attempts_placeholder = st.empty()
status_placeholder = st.empty()

st.markdown("<br>", unsafe_allow_html=True)

mode_cols = st.columns(3)
mode_labels = {"ASL": "✋ ASL", "Voice": "🎤 Voice", "Keyboard": "⌨ Keyboard"}
for col, mode in zip(mode_cols, mode_labels):
    is_active = st.session_state.input_mode == mode
    if col.button(mode_labels[mode], key=f"mode_{mode}", type="primary" if is_active else "secondary", use_container_width=True):
        st.session_state.input_mode = mode

st.markdown("---")

guess_line_placeholder = st.empty()

if st.session_state.input_mode == "ASL":
    render_asl_tab()

    # Deliberately OUTSIDE the fragment: a plain button click here does
    # a normal full-page rerun, so the grid/current-guess line update
    # immediately -- no special-casing needed, unlike the fragment's
    # auto-accepted-letter path above which has to request that itself.
    asl_action_cols = st.columns(3)
    if asl_action_cols[0].button("⌫ Backspace", key="asl_backspace", use_container_width=True, disabled=engine.game_over):
        _backspace()
    if asl_action_cols[1].button("Clear", key="asl_clear", use_container_width=True, disabled=engine.game_over):
        _clear()
    if asl_action_cols[2].button("Submit ✅", key="asl_submit", use_container_width=True, disabled=engine.game_over):
        _submit_word(st.session_state.current_guess)
elif st.session_state.input_mode == "Voice":
    st.caption(
        "🎤 Say a single word. Whisper transcribes it, and you confirm "
        "before it's submitted — nothing is ever guessed automatically."
    )
    _ensure_whisper_loaded()
    whisper_service = st.session_state.whisper_service

    if not whisper_service.is_available:
        st.warning(f"⚠️ Speech recognition unavailable ({whisper_service.error}). Use Keyboard mode to play.")
    elif st.session_state.voice_candidate:
        # A candidate is waiting for explicit confirmation -- recording
        # again is blocked until the user decides, matching "never
        # auto-submit" for voice the same way ASL never auto-submits.
        st.markdown(f"**You said:** _{st.session_state.voice_transcript}_")
        st.markdown(f"### Candidate guess: `{st.session_state.voice_candidate}`")
        confirm_cols = st.columns(2)
        if confirm_cols[0].button("✅ Confirm Guess", key="voice_confirm", use_container_width=True, disabled=engine.game_over):
            _confirm_voice_guess()
            st.rerun()  # this branch already rendered "candidate" UI above; force a fresh pass
        if confirm_cols[1].button("🔁 Try Again", key="voice_retry", use_container_width=True):
            _discard_voice_guess()
            st.rerun()
    else:
        if st.session_state.voice_transcript:
            st.markdown(f"**You said:** _{st.session_state.voice_transcript}_")
        if st.session_state.voice_message:
            st.warning(st.session_state.voice_message)
        if st.button("🎤 Record", key="voice_record", use_container_width=True, disabled=engine.game_over):
            _record_and_transcribe()
            st.rerun()  # show the fresh transcript/candidate immediately
else:
    # Physical keyboard: type the whole guess and press Enter/Submit.
    # A form batches the keystrokes into one rerun instead of one rerun
    # per character, and clear_on_submit empties the box automatically
    # -- both of which sidestep the "widget can't be modified after
    # it's instantiated" error you'd hit trying to clear it by hand.
    with st.form("typed_guess_form", clear_on_submit=True):
        typed_cols = st.columns([3, 1])
        typed_word = typed_cols[0].text_input(
            "Type your guess and press Enter",
            max_chars=engine.word_length,
            disabled=engine.game_over,
            label_visibility="collapsed",
            placeholder=f"Type a {engine.word_length}-letter word…",
        )
        typed_submitted = typed_cols[1].form_submit_button("⌨ Submit", use_container_width=True, disabled=engine.game_over)
    if typed_submitted:
        # Even an empty submission should show "Please enter a
        # 5-letter word." rather than silently doing nothing.
        _submit_word(typed_word)

    # On-screen keyboard: build the guess letter by letter (also what
    # ASL mode will drive in Phase 8 -- same _add_letter/_submit_word).
    render_keyboard(
        engine,
        on_letter=_add_letter,
        on_backspace=_backspace,
        on_clear=_clear,
        on_submit=lambda: _submit_word(st.session_state.current_guess),
    )

st.markdown("<br>", unsafe_allow_html=True)
restart_cols = st.columns([1, 1, 1])
if restart_cols[1].button("🔄 Restart Game", use_container_width=True):
    _restart()

# --- NOW that every widget above has had a chance to mutate state,
# render the grid/attempts/status/current-guess with up-to-date data.
with grid_placeholder.container():
    render_grid(engine, st.session_state.current_guess)
with attempts_placeholder.container():
    render_attempt_counter(engine)
with status_placeholder.container():
    render_status_message(engine)
if st.session_state.input_mode in ("Keyboard", "ASL"):
    with guess_line_placeholder.container():
        padded = st.session_state.current_guess.ljust(engine.word_length, "_")
        st.markdown(f'<div class="current-guess">Current Guess: {padded}</div>', unsafe_allow_html=True)

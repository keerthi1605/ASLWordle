"""
Central place for constants and paths used across SignWordle.

Keeping these in one small file means:
- No "magic numbers" scattered through game/vision/speech/UI code.
- One place to tweak game difficulty, model paths, or thresholds.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"

ASL_MODEL_PATH = MODELS_DIR / "asl_classifier.pkl"
HAND_LANDMARKER_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
ASL_LANDMARKS_CSV_PATH = DATA_DIR / "asl_landmarks.csv"
WORD_LIST_PATH = DATA_DIR / "words.txt"
STATS_PATH = DATA_DIR / "stats.json"

# --- Wordle rules --------------------------------------------------------
WORD_LENGTH = 5
MAX_ATTEMPTS = 6

# --- ASL recognition (used starting Phase 6/7) --------------------------
ASL_LABELS = [chr(c) for c in range(ord("A"), ord("Z") + 1) if chr(c) not in ("J", "Z")]
# J and Z require motion (a "drawn" gesture) which a single-frame, static
# landmark classifier cannot capture. This is a documented limitation,
# not a bug. See docs/learning/06_asl_classifier.md.

STABILIZATION_MIN_AGREEMENT = 0.7  # fraction of the window that must agree
CONFIDENCE_THRESHOLD = 0.6         # minimum classifier confidence to even count a frame

# Frames kept in the rolling prediction buffer. Sized against the
# REAL measured tick rate of the live ASL camera loop in app.py
# (~1 frame/sec -- Streamlit's st.fragment(run_every=...) has a
# practical ~1s floor in this version regardless of the requested
# interval, confirmed by direct measurement; the per-tick work itself
# -- camera read + MediaPipe detection + encode -- only takes ~50-80ms
# in isolation, so the fragment scheduler is the bottleneck, not the
# pipeline). A window of 15 at ~10 FPS (the original assumption) would
# be ~1.5s to accept a letter; at ~1 FPS that same window would take
# 15 SECONDS, which is unusable. 3 frames at ~1 FPS -> ~3s to hold a
# sign steady, a reasonable trade-off. See docs/learning/07_realtime_prediction.md.
STABILIZATION_WINDOW = 3

# --- Voice (used starting Phase 9) --------------------------------------
WHISPER_MODEL_SIZE = "base"  # small/fast enough for CPU-only laptops

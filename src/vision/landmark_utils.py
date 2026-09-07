"""
Pure data transforms on hand landmarks.

Deliberately has NO MediaPipe or OpenCV import: it only deals with
plain (x, y, z) numbers and index tuples (plus NumPy for the vector
math below). That keeps it trivially testable, and it's the shared
foundation for both drawing (Phase 4) and feature engineering
(Phase 5) — see docs/learning/05_feature_engineering.md.
"""

import numpy as np

# MediaPipe's hand model always returns exactly 21 landmarks per hand,
# indexed like this (see docs/learning/04_hand_landmarks.md for the
# picture):
#   0        wrist
#   1 -  4   thumb:  CMC, MCP, IP, TIP
#   5 -  8   index finger:  MCP, PIP, DIP, TIP
#   9 - 12   middle finger: MCP, PIP, DIP, TIP
#  13 - 16   ring finger:   MCP, PIP, DIP, TIP
#  17 - 20   pinky:         MCP, PIP, DIP, TIP
NUM_LANDMARKS = 21

# Which landmark indices are connected by a visible "bone" — this is
# MediaPipe's standard hand topology. Used for drawing (Phase 4) and
# is the same topology Phase 5's feature engineering reasons about.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index finger
    (5, 9), (9, 10), (10, 11), (11, 12),      # middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),    # ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (0, 17),                                  # palm base (wrist-pinky)
]


def to_pixel_coords(landmarks, image_width: int, image_height: int):
    """Convert a list of normalized (x, y, z) landmarks — each in
    [0, 1], as MediaPipe returns them — into integer pixel (x, y)
    coordinates for drawing on an image of the given size. z (depth)
    isn't meaningful for a 2D drawing, so it's dropped here."""
    return [(int(x * image_width), int(y * image_height)) for x, y, _z in landmarks]


# --- Phase 5: feature engineering -----------------------------------
#
# Raw landmark (x, y, z) values depend on WHERE the hand is in the
# frame and HOW BIG it appears (closer to the camera = bigger). Two
# people signing the same letter -- or the same person moving their
# hand slightly -- would otherwise produce very different numbers for
# what should be the "same" shape. Normalizing removes both effects
# before the numbers ever reach a classifier. See
# docs/learning/05_feature_engineering.md for the full explanation.

WRIST_INDEX = 0        # landmark 0 -- the normalization origin
MIDDLE_MCP_INDEX = 9   # middle finger's base knuckle -- the "ruler"
NUM_FEATURES = NUM_LANDMARKS * 3  # 21 landmarks x (x, y, z) = 63


def normalize_landmarks(landmarks) -> np.ndarray:
    """
    Normalize 21 (x, y, z) landmarks for translation and scale
    invariance. Returns a NumPy array of shape (21, 3).

    Translation invariance: subtract the wrist's coordinates from
    every point, so the wrist becomes the origin (0, 0, 0) no matter
    where the hand physically is in the camera frame.

    Scale invariance: divide every (now wrist-relative) coordinate by
    the distance from the wrist to the middle finger's base knuckle
    (landmark 9). That distance is a stable "ruler" for hand size --
    it shrinks/grows consistently whether the hand is small, large,
    close to the camera, or far away, so dividing by it makes the
    result the same regardless of hand size or distance from camera.
    """
    points = np.asarray(landmarks, dtype=np.float64)  # shape (21, 3)
    wrist = points[WRIST_INDEX]
    translated = points - wrist  # every point now relative to the wrist

    scale = float(np.linalg.norm(translated[MIDDLE_MCP_INDEX]))
    if scale < 1e-6:
        # Degenerate input (e.g. every landmark at the same point) --
        # fall back to a tiny epsilon instead of dividing by zero.
        scale = 1e-6

    return translated / scale


def landmarks_to_feature_vector(landmarks) -> np.ndarray:
    """
    Convert 21 raw (x, y, z) landmarks into the flat 63-value feature
    vector (`NUM_FEATURES`) that Phase 6's classifier is trained on
    and predicts from: normalize for translation/scale, then flatten
    the (21, 3) array into a single 1D vector of length 63.
    """
    normalized = normalize_landmarks(landmarks)
    return normalized.flatten()

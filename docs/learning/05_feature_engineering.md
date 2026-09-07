# 05 — Feature Engineering

## 1. What it does

Turns MediaPipe's 21 raw `(x, y, z)` hand landmarks into a single,
**normalized 63-value feature vector** — the exact input Phase 6's
classifier is trained on and predicts from. See
[landmark_utils.py](../../src/vision/landmark_utils.py)'s
`normalize_landmarks()` and `landmarks_to_feature_vector()`.

## 2. What "features" and "feature vectors" mean

A **feature** is a single measurable number that (hopefully) helps
tell classes apart — here, one landmark's x, y, or z coordinate. A
**feature vector** is just all of those numbers for one example,
packed into a fixed-length list/array. Every ASL letter this project
will ever classify gets represented the exact same way: one 63-number
vector (21 landmarks × 3 coordinates each — `NUM_FEATURES` in
`landmark_utils.py`). A classifier never sees an "image" or a "hand";
it only ever sees 63 floats and has to learn which patterns of those
floats correspond to which letter.

## 3. Why raw coordinates aren't good enough

MediaPipe's raw landmarks are normalized to the *frame*, not to the
*hand*. That means two genuinely different things can change the raw
numbers even though the hand shape (the letter being signed) hasn't
changed at all:

- **Position** — sign the same letter in the top-left of the frame vs.
  the center, and every (x, y) shifts, even though it's the same
  letter.
- **Distance from the camera** — lean closer and the hand takes up
  more of the frame (bigger spread between landmarks); lean back and
  it shrinks. Same letter, very different raw numbers.

A classifier trained on raw coordinates would have to separately learn
every combination of "letter A near the top" vs. "letter A near the
bottom" vs. "letter A big" vs. "letter A small" — effectively
multiplying how much it has to learn for no good reason. Normalizing
removes both effects *before* the classifier ever sees the numbers.

## 4. Translation invariance

**Translation invariance** = "doesn't matter where in the frame the
hand is." Achieved by picking one landmark as a fixed reference point
and subtracting it from every other landmark. This project uses the
**wrist (landmark 0)**: every landmark's coordinates become "how far
is this point from the wrist," instead of "where is this point in the
frame." After this step, the wrist itself always ends up at exactly
`(0, 0, 0)` — moving your whole hand around the room no longer changes
any of the numbers, only the actual finger *shape* does.

## 5. Scale invariance

**Scale invariance** = "doesn't matter how big the hand appears" (near
vs. far from camera, or naturally larger/smaller hands). Achieved by
dividing every (already wrist-relative) coordinate by a **reference
distance** measured on the same hand — here, the distance from the
wrist to the middle finger's base knuckle (landmark 9). That distance
grows and shrinks *exactly in proportion* with the rest of the hand's
apparent size, so dividing by it cancels out the size difference:
sign the same letter close-up or far away and, after this step, the
numbers come out the same.

## 6. Putting it together: `normalize_landmarks()`

```python
points = np.asarray(landmarks)       # (21, 3)
wrist = points[0]
translated = points - wrist          # translation invariance
scale = norm(translated[9])          # wrist-to-middle-knuckle distance
normalized = translated / scale      # scale invariance
```

Two safety details worth knowing:

- **Division-by-zero guard**: if every landmark were somehow identical
  (a degenerate detection), `scale` would be `0`. The code clamps it
  to a tiny epsilon (`1e-6`) instead of dividing by zero, so a bad
  frame produces a harmless all-zero-ish vector instead of crashing
  the whole pipeline with a `NaN`/`inf`.
- **z is normalized the same way as x/y** (divided by the same scale),
  even though it's a noisier signal (see `docs/learning/04_hand_landmarks.md`)
  — kept for consistency and because the classifier can learn to weight
  it less if it's not useful.

## 7. Input / Output

- Input: 21 `(x, y, z)` tuples (normalized-to-frame, as MediaPipe
  returns them).
- `normalize_landmarks()` output: a `(21, 3)` NumPy array, translation-
  and scale-normalized.
- `landmarks_to_feature_vector()` output: that array flattened to a
  flat `(63,)` vector — `NUM_FEATURES = 21 × 3 = 63`.

## 8. Connection to other components

Sits between Phase 4 (`HandDetector` — produces raw landmarks) and
Phase 6 (`asl_recognizer.py`'s Random Forest, which is trained on and
predicts from these 63-value vectors). Like every other vision-layer
file, it has zero knowledge of Wordle or the UI.

## 9. Important functions

- `normalize_landmarks(landmarks) -> np.ndarray` shape `(21, 3)`
- `landmarks_to_feature_vector(landmarks) -> np.ndarray` shape `(63,)`
- Constants: `WRIST_INDEX = 0`, `MIDDLE_MCP_INDEX = 9`, `NUM_FEATURES = 63`

## 10. Common errors

- **Forgetting to normalize before training/predicting** — a
  classifier trained on normalized vectors will perform badly (or
  nonsensically) on raw, un-normalized ones, and vice versa. Always
  use the same function on both training data and live predictions.
- **Picking a scale reference that isn't stable** — e.g. dividing by
  frame width instead of a hand-relative distance would reintroduce
  the "closer to camera = different numbers" problem this step exists
  to remove.
- **Not guarding the divide** — a real (if rare) input can make the
  scale reference distance 0; skipping the epsilon guard would crash
  on that input instead of degrading gracefully.

## 11. Testing

`tests/test_landmark_utils.py` checks all of this with plain numbers
(no camera or model needed):

- Output shapes are exactly `(21, 3)` and `(63,)`.
- The wrist lands exactly at `(0, 0, 0)` after normalization.
- **Translation invariance**: shifting an entire synthetic hand by a
  constant offset produces an *identical* normalized result.
- **Scale invariance**: scaling an entire synthetic hand outward from
  the wrist by 3× produces an *identical* normalized result.
- Degenerate input (21 identical points) doesn't produce `NaN`/`inf`.

## 12. Likely viva questions

- **"Why divide by wrist-to-middle-knuckle distance instead of, say,
  hand width in pixels?"** — Pixel-based measurements depend on frame
  resolution and camera distance; a landmark-to-landmark distance on
  the hand itself scales consistently with the hand's actual apparent
  size in the image, making it a much more stable reference.
- **"What would happen if you skipped normalization?"** — The
  classifier would need far more training data to learn that the same
  letter looks different at every position/distance, and would likely
  generalize poorly to a new person or setup.
- **"Why 63 numbers specifically?"** — 21 landmarks × 3 coordinates
  (x, y, z) each; see `docs/learning/04_hand_landmarks.md`.
- **"What happens on a degenerate or empty landmark set?"** — Handled
  explicitly: an all-identical-point input is guarded against
  division by zero rather than crashing; an empty hand list (Phase 4)
  never reaches this function at all — "no hand" and "a hand with a
  computable feature vector" are always kept distinct.

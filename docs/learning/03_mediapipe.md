# 03 — MediaPipe

## 1. What it does

MediaPipe is a library (originally by Google) for running pretrained
computer-vision models efficiently, especially the "detect body
parts/objects from a live camera in real time on a normal CPU" kind of
task. In SignWordle it's used for exactly one thing: finding a hand in
a webcam frame and returning 21 (x, y, z) points that describe its
shape and pose. See [hand_detector.py](../../src/vision/hand_detector.py).

## 2. Why it's needed

Recognizing an ASL letter from a raw photo (millions of pixels) is a
much harder, noisier problem than recognizing it from 21 clean (x, y,
z) numbers that already describe the hand's shape. MediaPipe does the
hard "find the hand and its joints" work; Phase 5/6 only need to
reason about the resulting 21 points, not pixels.

## 3. A version gotcha worth knowing (and explaining in a viva)

MediaPipe has two different Python APIs across its history:

- **Legacy "Solutions" API** (`mp.solutions.hands`) — what almost
  every older tutorial online shows. Simple, no separate model file
  needed (bundled into the package).
- **New "Tasks" API** (`mp.tasks.python.vision.HandLandmarker`) — what
  the `mediapipe` version this project installed (1.0.1) actually
  ships. It needs an explicit **model file** you download separately
  (a `.task` bundle), passed via `BaseOptions(model_asset_path=...)`.

This project uses the **Tasks API** because that's what was actually
available when `pip install mediapipe` ran. This is a good, honest
example of a real "the tutorial doesn't match the installed version"
problem — always check what's actually importable
(`dir(mp.tasks.python.vision)`) rather than trusting an old
Stack Overflow snippet.

## 4. The model file

`models/hand_landmarker.task` is Google's official, publicly
published MediaPipe Hand Landmarker model
(`hand_landmarker/float16`, from
`storage.googleapis.com/mediapipe-models/...`, Apache-2.0 licensed
like the rest of MediaPipe's model zoo). It's a ~7.8 MB TensorFlow
Lite model bundle — not something we trained, and not something we
fabricated; it's the standard pretrained asset every MediaPipe hand
tutorial uses. It's excluded from git (see `.gitignore`) because it's
a large binary; **download it yourself** before running hand tracking:

```bash
curl -L -o models/hand_landmarker.task "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
```

If the file is missing, `HandDetector.__init__` raises a clear
`FileNotFoundError` instead of a confusing crash deep inside
MediaPipe, and `app.py` catches that and falls back to a plain camera
preview with a visible warning — see `docs/learning/10_hci_design.md`
on honest degradation instead of faking a feature.

## 5. How detection works, step by step

1. Convert the RGB NumPy frame into MediaPipe's own `mp.Image`
   wrapper: `mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)`.
2. Call `landmarker.detect(mp_image)` — runs the neural network.
3. The result (`HandLandmarkerResult`) has:
   - `hand_landmarks`: a list of hands, each a list of 21 landmark
     objects (each with `.x`, `.y`, `.z`, normalized 0-1).
   - `handedness`: a matching list telling you "Left" or "Right" (from
     the camera's point of view — see the mirroring note below).
4. `HandDetector.detect()` converts these into plain Python tuples —
   `(x, y, z)` — so nothing downstream needs to import `mediapipe`
   directly (see the architecture rule in `docs/learning/01_project_architecture.md`).

## 6. `num_hands` and confidence thresholds

`HandLandmarkerOptions(num_hands=1, min_hand_detection_confidence=0.5)`
— we only ever look for **one** hand (fingerspelling only needs one),
and only accept a detection the model is at least 50% confident about.
Raising this threshold means fewer false positives but might miss a
hand in bad lighting; lowering it does the opposite. 0.5 is MediaPipe's
own sensible default.

## 7. Input / Output

- Input: one RGB frame (from `src/vision/camera.py`).
- Output: `(hands, handedness)` — `hands` is `list[list[(x,y,z)]]`
  (empty list if no hand found), `handedness` a matching list of
  strings. Plain data, easy to test without a camera (see
  `tests/test_hand_detector.py`'s blank-frame tests).

## 8. Connection to other components

- Takes frames from `Camera` (Phase 3).
- Feeds normalized (x, y, z) landmarks into `landmark_utils.py`
  (drawing now, feature vectors in Phase 5).
- Still knows nothing about ASL letters or Wordle — same boundary
  rule as every other vision-layer file.

## 9. Common errors

- **Forgetting the model file** — `HandDetector()` raises
  `FileNotFoundError` with the exact expected path; download it (§4).
- **Passing a BGR frame** — MediaPipe's `SRGB` image format expects
  RGB; if you feed it a raw OpenCV BGR frame, detection still *runs*
  but is less accurate since the model was trained on RGB images. Our
  `Camera.read_frame()` already converts to RGB, so this is a
  non-issue as long as detection only ever consumes its output.
- **Assuming `mp.solutions` exists** — it doesn't in this installed
  version; see §3.

## 10. Likely viva questions

- **"What does MediaPipe actually give you?"** — 21 (x, y, z) points
  per detected hand, normalized to the frame size, plus a Left/Right
  label. Nothing about letters or gestures — that's built on top.
- **"Why not just feed the raw image into your own classifier?"** —
  Two reasons: (1) a 21-point skeleton is a far smaller, cleaner
  representation than a full image, which makes a simple classifier
  (Phase 6's Random Forest) actually work well; (2) it's inherently
  more robust to background, lighting, and skin tone than raw pixels.
- **"What happens if MediaPipe can't find a hand?"** — `detect()`
  returns `([], [])`. No crash, no guess — Phase 6+ treats "no hand"
  as "no letter to add," never fabricating a prediction.
- **"Legacy `mp.solutions` vs. new Tasks API — why does it matter?"**
  — Because the installed `mediapipe` version only ships the Tasks
  API; code written against the older tutorials' `mp.solutions.hands`
  simply doesn't run. Always verify against the actually-installed
  package.

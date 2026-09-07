# 04 — The 21 Hand Landmarks

## 1. What they are

MediaPipe's hand model always returns exactly **21 points** per
detected hand — one for the wrist and four for each finger (base,
two middle joints, tip). Together they form a simplified "skeleton"
of the hand. See [landmark_utils.py](../../src/vision/landmark_utils.py)
for the exact index numbering used everywhere in this project.

```
        8   12   16   20      <- fingertips
        |    |    |    |
        7   11   15   19
        |    |    |    |
        6   10   14   18
        |    |    |    |
        5    9   13   17      <- finger bases (knuckles)
         \   |    |   /
          \  |    |  /
       4   \ |    | /
        \   \|    |/
         3   +----+
          \ /      \
           2         0        <- 0 = wrist
            \       /
             1-----/           <- thumb base
```

(Rough ASCII sketch — the real layout is a hand shape, not a grid;
see `docs/learning/03_mediapipe.md` for the official landmark diagram
reference, and just run Phase 4's live view to see it directly.)

Index groups:

| Indices | Finger |
|---|---|
| 0 | Wrist |
| 1–4 | Thumb (CMC, MCP, IP, TIP) |
| 5–8 | Index finger (MCP, PIP, DIP, TIP) |
| 9–12 | Middle finger |
| 13–16 | Ring finger |
| 17–20 | Pinky |

## 2. x / y / z coordinates

Each landmark is `(x, y, z)`:

- **x, y** — position within the frame, **normalized** to `[0, 1]`
  (not pixels). `x=0` is the left edge, `x=1` the right edge; same
  idea for `y` top-to-bottom. Multiply by the frame's width/height to
  get pixel coordinates — that's exactly what
  `landmark_utils.to_pixel_coords()` does.
- **z** — rough relative depth: smaller (more negative) means closer
  to the camera than the wrist, roughly in the same scale as x. It's
  **not** true physical distance, and is generally noisier/less
  reliable than x/y. Phase 5's features use it anyway (all 63 numbers:
  21 × 3), but lean primarily on x/y patterns.

## 3. Handedness

MediaPipe also reports "Left" or "Right" per detected hand. Two things
to know:

1. It's from the **camera's** point of view unless you mirror first —
   since `Camera` in this project already mirrors every frame before
   handing it to the detector, the label matches what the *user* would
   call their own hand (raise your right hand → labeled "Right").
2. SignWordle doesn't currently use handedness for anything — ASL
   fingerspelling works the same with either hand — but it's already
   returned from `HandDetector.detect()` in case a later feature wants
   it (e.g., a Phase-10-style hint like "try your other hand").

## 4. Landmark data as a Python structure

```python
hands, handedness = detector.detect(frame_rgb)
# hands:      [ [ (x0,y0,z0), (x1,y1,z1), ..., (x20,y20,z20) ], ... ]
#               ^ one hand ^   ^ 21 landmarks per hand ^
# handedness: ["Right"]   (same length as `hands`)
```

`hands` is `[]` when no hand is found — never `None`, never a
fabricated guess. Downstream code (Phase 5+) can always safely do
`if hands:` without special-casing `None`.

## 5. Image classification vs. landmark classification

Two very different ways you *could* recognize an ASL letter:

- **Image classification**: feed the whole photo into a big neural
  network (a CNN) and let it learn what an "A" looks like from raw
  pixels. Needs lots of training data, is sensitive to background,
  lighting, skin tone, camera angle, and is a "black box" that's hard
  to explain in a viva.
- **Landmark classification** (what this project does): reduce the
  photo to 21 (x, y, z) numbers *first* using a pretrained hand model
  (MediaPipe), then classify **those numbers** with a much simpler
  model (Phase 6's Random Forest). Far less training data needed,
  robust to background/lighting/skin tone (the hand's shape in 3D is
  what matters, not its pixels), and easy to explain: "here are 63
  numbers describing finger positions, here's a decision-tree-based
  classifier trained on those."

This project deliberately picked landmark classification — see
`docs/learning/06_asl_classifier.md` for why that pairs so well with
scikit-learn's Random Forest specifically.

## 6. Important functions/classes

- `HandDetector.detect(frame_rgb) -> (hands, handedness)` — Phase 4.
- `landmark_utils.to_pixel_coords(landmarks, w, h)` — normalized → pixel,
  used for drawing.
- `landmark_utils.HAND_CONNECTIONS` — the 21-point "skeleton" topology
  (which index pairs are connected by a bone), reused for both drawing
  here and reasoning about the hand structurally in Phase 5.
- `draw_hand_landmarks(frame_rgb, hands)` — plain `cv2.circle`/`cv2.line`
  calls, drawn directly with each landmark's pixel position.

## 7. Common errors

- **Treating (x, y) as pixel coordinates directly** — they're
  normalized 0-1; forgetting to multiply by width/height puts every
  dot in the top-left corner of the image.
- **Assuming z is true depth in centimeters** — it's a rough, relative
  signal, not a calibrated measurement.
- **Expecting landmarks when the hand is off-frame or occluded** —
  `hands` is correctly `[]`; that's not a bug to "fix," it's the
  honest answer.

## 8. Likely viva questions

- **"Why 21 points and not more/fewer?"** — That's MediaPipe's fixed
  hand model output; it was chosen by its designers to capture finger
  joints precisely enough for gesture recognition without excess
  detail.
- **"What's the difference between x/y and z here?"** — x/y are
  reliable normalized image-plane coordinates; z is a coarser relative
  depth cue, less precise, derived rather than directly measured.
- **"Why landmark classification instead of a CNN on raw images?"** —
  Smaller/simpler model, less data needed, more explainable, more
  robust to background and lighting — see §5.
- **"What does the app do if no hand is visible?"** — Nothing gets
  added to the current guess; the UI shows "no hand detected" rather
  than guessing.

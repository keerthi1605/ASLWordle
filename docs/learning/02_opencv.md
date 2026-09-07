# 02 — OpenCV & the Webcam

## 1. What it does

`src/vision/camera.py` wraps OpenCV's `cv2.VideoCapture` to give the
rest of the app a simple, honest interface: start the webcam, pull
one mirrored RGB frame at a time, stop it. See
[camera.py](../../src/vision/camera.py).

## 2. Why it's needed

Everything downstream — hand landmarks (Phase 4), features (Phase 5),
ASL classification (Phase 6) — needs a steady stream of images from a
real camera. Isolating that in one small class means the rest of the
vision pipeline never has to think about `cv2.VideoCapture` quirks
(backends, release(), frames failing to read) directly.

## 3. Key concepts

### 3.1 What is a frame?

A webcam doesn't send video, it sends a rapid sequence of still
images — **frames**. In OpenCV each frame is just a NumPy array of
shape `(height, width, 3)`: one row per pixel row, one column per
pixel column, and 3 numbers per pixel for its color channels. Our
webcam gives `(480, 640, 3)` frames — 480 rows tall, 640 columns wide.

### 3.2 `cv2.VideoCapture`

`cv2.VideoCapture(0)` opens camera device index `0` (the first camera
Windows/Linux/macOS finds — usually a laptop's built-in webcam).
`.read()` grabs the next available frame as `(success: bool, frame)`.
`success` is `False` if the camera disconnected, is in use by another
app, or was never opened — **always check it** before touching
`frame`, which is `None` on failure.

We explicitly pass `cv2.CAP_DSHOW` (DirectShow) as the backend on
Windows, because OpenCV's default auto-detected backend can be slow to
open or unreliable on some Windows webcams; DirectShow is the fast,
well-supported one. Other OSes don't have this backend, so the code
falls back to `cv2.CAP_ANY` there.

### 3.3 Continuous processing (the "video" loop)

A webcam feed is just: **grab a frame, do something with it, repeat**.
Streamlit isn't built around infinite loops (a script that never
returns would freeze the whole UI), so `app.py` gets the "video" look
using a different trick: each run grabs **exactly one** frame,
displays it, and then calls `st.rerun()` at the very end to trigger
another run. Every run is a "tick" of the video loop. A short
`time.sleep(0.03)` before each rerun caps this near 30 frames/sec so
it doesn't peg a CPU core spinning as fast as possible.

**Real bug this caused, and the fix**: the first version called plain
`st.rerun()`, which reruns the **entire script** — meaning the Wordle
grid, on-screen keyboard, and every button on the page were being
fully re-executed and redrawn ~30 times a second too, which looked
like the whole page glitching/flickering, not just the camera image.
The fix was `st.fragment` (Streamlit 1.37+): wrapping the whole ASL
camera block in `@st.fragment` and calling `st.rerun(scope="fragment")`
instead of plain `st.rerun()` means **only that fragment's own
contents** re-execute on each video tick — the title, grid, keyboard,
and mode buttons outside it are completely untouched. This is the
general rule for "live updating something" in a Streamlit app that
also has other, unrelated UI: put the live-updating part in its own
`st.fragment` rather than reaching for a full-page `st.rerun()`.

### 3.4 FPS, and reducing image-swap flicker

We measure real FPS instead of assuming it — timestamp each
successfully-read frame and compute `1 / (time_now - time_of_last_frame)`.
Even after scoping the loop to an `st.fragment` (§3.3), each tick still
does real work — read a frame, run MediaPipe hand detection on it,
JPEG-encode it, send it to the browser — so we intentionally target a
lower, honest **10-15 FPS** (`time.sleep(0.08)`, not `0.03`) rather than
chasing 30. Measured in this project: **~6-10 FPS** depending on
whether a hand is in frame (detection costs a bit more when it finds
one). Fewer swaps per second directly means less visible flicker, and
it's a number this project can actually hit rather than a nominal cap
it never reaches.

**The flicker itself**: even scoped to a fragment, swapping an
`<img>`'s source to a brand-new image every tick causes a brief
decode-and-repaint in the browser — a real, visible artifact of "video
via repeated still images" that a tiny bit of size matters for. By
default, `st.image()` given a raw NumPy array re-encodes it as JPEG at
**quality=100** (verified by reading Streamlit's own source,
`elements/lib/image_utils.py`) — needlessly large for a live-updating
preview. We encode the frame ourselves instead
(`cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])`) and
hand `st.image()` the resulting bytes directly, which it passes through
unchanged. On a representative detailed frame this cut the payload
from **412 KB to 86 KB (≈79% smaller)** — measured, not guessed, by
encoding the same synthetic image both ways and comparing byte counts.
A smaller, faster-to-decode JPEG means a shorter flash on each swap.
Detection is unaffected: MediaPipe still runs on the full, uncompressed
frame — only the copy sent to the browser is compressed.

**Honest limit**: some flicker is inherent to this approach and can't
be fully eliminated without different infrastructure (true streaming
video, e.g. `streamlit-webrtc`), which this project deliberately avoids
per "no unnecessary dependencies." We report the measured
improvement rather than claiming it's now flicker-free — see the
project's "never fabricate metrics" rule, which applies to performance
claims just as much as to accuracy numbers.

### 3.5 BGR vs. RGB

OpenCV reads and stores color images as **BGR** (Blue, Green, Red
channel order) for historical reasons. Almost everything else —
Streamlit's `st.image`, PIL, matplotlib, MediaPipe's drawing utilities
— expects **RGB**. Forgetting to convert gives you a real image with
swapped colors (skin looks blue, sky looks orange). We convert once,
right after reading each frame: `cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)`.

### 3.6 Mirroring

A raw webcam frame shows you as everyone else sees you — if you raise
your right hand, it appears on the *left* side of the frame. That's
disorienting to interact with (you're not looking in a mirror, you're
looking at a security camera). `cv2.flip(frame, 1)` flips the frame
horizontally so moving your hand right moves it right on screen too —
standard for any "look at yourself" camera UI, and it'll matter again
in Phase 4 when landmarks need to visually line up with your real hand.

## 4. Input / Output

- Input: a device index (which physical camera) and a mirror flag.
- Output: `read_frame()` returns `(success, rgb_frame_or_None, message)`
  — plain, testable data, no OpenCV-specific types leak out.

## 5. Connection to other components

Phase 4 (MediaPipe) will take the RGB frames this module produces and
run hand detection on them. `camera.py` doesn't know MediaPipe exists,
same separation-of-concerns rule as the game engine.

## 6. Important functions/classes

- `Camera.start() -> (bool, str)` / `Camera.stop()`
- `Camera.read_frame() -> (bool, frame_or_None, str)`
- `Camera.is_open` (property)
- Also usable as a context manager: `with Camera() as cam: ...`

## 7. Common errors

- **Camera "opens" but every `read()` fails** — usually another app
  (Zoom, Teams, a second browser tab) is holding the camera. Close
  other apps using the webcam and try again.
- **`cap.isOpened()` returns `False` immediately** — wrong device
  index, no camera present, or the OS denied camera permission to the
  terminal/Python process. On Windows: Settings → Privacy & security →
  Camera → make sure desktop apps are allowed access.
- **Colors look wrong (blue skin, orange sky)** — forgot the
  `BGR2RGB` conversion, or converted twice (which flips it back to BGR).
- **The image looks mirror-flipped when you didn't expect it, or not
  flipped when you did** — check the `mirror` flag; it's `True` by
  default in this app.

## 8. Testing the camera

**Automated**: `tests/test_camera.py` runs real hardware tests
(`unittest.skipUnless`) automatically when a webcam is present — on
this development machine that's exactly what happened; run:

```bash
venv/Scripts/python.exe -m unittest tests.test_camera -v
```

On a machine with no webcam, those tests are automatically **skipped**
(not failed) — a missing camera is an environment fact, not a bug.

**Manual (in the app)**: run `streamlit run app.py`, click the
**✋ ASL** mode button, then **▶ Start Camera**. You should see your
own mirrored live video with a green "Camera live — mirrored, ~N FPS"
status line. Click **■ Stop Camera** to release it. If it fails, the
app shows a red error message explaining why (camera in use, not
found, or permission denied) instead of crashing or pretending to see
something it can't.

## 9. Known limitation

Doing "video" in Streamlit by repeatedly swapping a still image (§3.3)
is a real, honest trade-off — the alternative is a library like
`streamlit-webrtc`, which pulls in a whole WebRTC stack that this
project deliberately avoids per "no unnecessary dependencies." Now
that the loop is scoped to an `st.fragment` and each frame is a
smaller, lower-quality JPEG (§3.4), the rest of the page (grid,
keyboard, mode buttons, scroll position) stays completely untouched,
and the camera image's own flicker is measurably smaller — but a
slight flicker between frames can still happen. That's the inherent
cost of "swap a still image ~10 times a second" rather than true
streaming video.

## 10. Likely viva questions

- **"Why BGR instead of RGB in OpenCV?"** — Historical: early camera
  and Windows bitmap APIs OpenCV was built against used BGR byte order.
  It never changed for backwards compatibility.
- **"How do you get 'live video' inside Streamlit, which reruns the
  whole script on every interaction?"** — Grab one frame per script
  run, display it, then call `st.rerun()` yourself at the very end to
  trigger the next run — turning Streamlit's rerun mechanism into a
  manual video loop, capped with a short `sleep()` so it isn't wasteful.
- **"Why is the measured FPS (~10) lower than the sleep cap (~30)?"**
  — Because each `st.rerun()` has real round-trip cost (websocket,
  script re-execution, DOM diff) that this app doesn't try to work
  around — that overhead is the actual bottleneck, not the camera.
- **"What happens if no camera is available?"** — `Camera.start()`
  returns `(False, <clear reason>)` instead of throwing an exception;
  the UI shows that reason and never fakes a video feed.

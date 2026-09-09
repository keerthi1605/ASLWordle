"""
Thin wrapper around OpenCV's VideoCapture.

This module's only job is "give me webcam frames, mirrored, as RGB,
and tell me clearly if that's not possible." It knows nothing about
hand landmarks, ASL letters, or Wordle — see
docs/learning/01_project_architecture.md for why that separation
matters (this file must stay reusable/testable on its own).
"""

import sys

import cv2


class Camera:
    """Start/stop a webcam and pull mirrored RGB frames from it."""

    def __init__(self, device_index: int = 0, mirror: bool = True):
        self.device_index = device_index
        self.mirror = mirror
        self._cap: cv2.VideoCapture | None = None

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def start(self) -> tuple[bool, str]:
        """Open the webcam. Returns (success, message) — never raises,
        so a missing/busy camera degrades to a clear message instead
        of crashing the app (see docs/learning/10_hci_design.md on
        error prevention / honest fallback)."""
        if self.is_open:
            return True, "Camera already running."

        # CAP_DSHOW is the fast, reliable backend on Windows. On other
        # platforms it doesn't exist, so fall back to OpenCV's default
        # auto-detection instead.
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        try:
            cap = cv2.VideoCapture(self.device_index, backend)
        except cv2.error as exc:  # pragma: no cover - defensive
            return False, f"OpenCV error opening camera: {exc}"

        if not cap.isOpened():
            cap.release()
            return False, (
                f"Could not open webcam (device index {self.device_index}). "
                "Check that a camera is connected, not already in use by "
                "another app, and that camera permission is granted."
            )
        self._cap = cap
        return True, "Camera started."

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read_frame(self):
        """Grab one frame. Returns (success, rgb_frame_or_None, message)."""
        if not self.is_open:
            return False, None, "Camera is not running."

        ret, frame_bgr = self._cap.read()
        if not ret or frame_bgr is None:
            return False, None, "Failed to read a frame from the camera."

        if self.mirror:
            # flip code 1 = horizontal flip, i.e. a mirror. Matches how
            # people expect to see themselves (raise your right hand,
            # your mirrored self raises what looks like its right hand
            # from your point of view).
            frame_bgr = cv2.flip(frame_bgr, 1)

        # OpenCV reads/writes BGR by convention; almost everything else
        # (Streamlit's st.image, PIL, matplotlib) expects RGB.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return True, frame_rgb, ""

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

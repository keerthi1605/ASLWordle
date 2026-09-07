"""
Tests for src/vision/camera.py.

Camera access is hardware-dependent, so these tests probe the real
webcam if one is available and SKIP (not fail) if it isn't — a missing
camera is an environment fact, not a code bug. Run:

    venv/Scripts/python.exe -m unittest tests.test_camera -v
"""

import unittest

from src.vision.camera import Camera


def _camera_available(device_index: int = 0) -> bool:
    cam = Camera(device_index)
    ok, _ = cam.start()
    cam.stop()
    return ok


_HAS_CAMERA = _camera_available()


class TestCameraWithoutHardware(unittest.TestCase):
    """These don't need a webcam at all — they check the class's
    error-handling contract."""

    def test_not_open_before_start(self):
        cam = Camera()
        self.assertFalse(cam.is_open)

    def test_read_frame_before_start_fails_cleanly(self):
        cam = Camera()
        ok, frame, message = cam.read_frame()
        self.assertFalse(ok)
        self.assertIsNone(frame)
        self.assertTrue(message)  # a human-readable reason, not empty

    def test_stop_without_start_does_not_raise(self):
        cam = Camera()
        cam.stop()  # should be a no-op, not an exception

    def test_nonexistent_device_index_fails_gracefully(self):
        # Device index 99 shouldn't exist on any normal machine.
        cam = Camera(device_index=99)
        ok, message = cam.start()
        self.assertFalse(ok)
        self.assertIn("Could not open", message)
        cam.stop()


@unittest.skipUnless(_HAS_CAMERA, "No webcam detected on this machine — skipping hardware tests.")
class TestCameraWithHardware(unittest.TestCase):
    """These need an actual webcam and are skipped automatically when
    one isn't present (e.g. a CI runner or headless grading machine)."""

    def test_start_stop(self):
        cam = Camera()
        ok, _ = cam.start()
        self.assertTrue(ok)
        self.assertTrue(cam.is_open)
        cam.stop()
        self.assertFalse(cam.is_open)

    def test_read_frame_shape_and_mirroring(self):
        cam = Camera(mirror=True)
        cam.start()
        ok, frame, _ = cam.read_frame()
        cam.stop()
        self.assertTrue(ok)
        self.assertIsNotNone(frame)
        self.assertEqual(len(frame.shape), 3)  # height, width, channels
        self.assertEqual(frame.shape[2], 3)  # RGB has 3 channels

    def test_context_manager_releases_camera(self):
        with Camera() as cam:
            self.assertTrue(cam.is_open)
        self.assertFalse(cam.is_open)


if __name__ == "__main__":
    unittest.main()

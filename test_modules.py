"""
Comprehensive unit and integration test suite for Hand Gesture Desktop Controller.
"""
import time
import unittest
import numpy as np

from config import (
    CameraConfig,
    InteractionConfig,
    SafetyConfig,
    SmoothingConfig,
    TrackingConfig,
    TrackingMode,
)
from gesture_recognizer import GestureEvent, GestureRecognizer, GestureResult, GestureState
from hand_tracker import HandData, HandLandmarkIndex, HandTracker
from mouse_controller import MouseController
from smoothing import CursorSmoother, LowPassFilter, OneEuroFilter1D


class TestSmoothing(unittest.TestCase):
    def test_low_pass_filter(self):
        lpf = LowPassFilter(alpha=0.5)
        self.assertEqual(lpf.filter(10.0), 10.0)
        self.assertEqual(lpf.filter(20.0), 15.0)

    def test_one_euro_filter_precision(self):
        oef = OneEuroFilter1D(min_cutoff=0.7, beta=0.025)
        v1 = oef.filter(100.0, timestamp=0.0)
        v2 = oef.filter(101.0, timestamp=0.033)
        self.assertAlmostEqual(v1, 100.0)
        self.assertTrue(100.0 <= v2 <= 101.0)

    def test_cursor_smoother_deadzone(self):
        cfg = SmoothingConfig(deadzone_pixels=5.0)
        smoother = CursorSmoother(cfg)
        s1 = smoother.smooth(100.0, 100.0)
        s2 = smoother.smooth(102.0, 102.0)  # Distance < 5.0 -> deadzone active
        self.assertEqual(s1, s2)

        s3 = smoother.smooth(150.0, 150.0)  # Distance > 5.0 -> movement occurs
        self.assertNotEqual(s2, s3)


class TestMouseController(unittest.TestCase):
    def setUp(self):
        self.ctrl = MouseController()
        # Disable actual OS clicks during automated unit testing
        self.ctrl.set_enabled(False)

    def test_coordinate_mapping(self):
        w, h = self.ctrl.screen_width, self.ctrl.screen_height
        cx, cy = self.ctrl.map_to_screen(0.5, 0.5)
        self.assertAlmostEqual(cx, (w - 1) / 2.0, delta=2.0)
        self.assertAlmostEqual(cy, (h - 1) / 2.0, delta=2.0)

    def test_cursor_remains_locked_while_holding_under_2s(self):
        init_pos = self.ctrl.current_screen_pos
        holding_result = GestureResult(
            state=GestureState.HOLDING_TO_MOVE,
            event=GestureEvent.NONE,
            pinch_distance=0.02,
            right_pinch_distance=0.2,
            scroll_delta=0.0,
            is_movement_engaged=False,
            is_left_pinched=True,
            is_right_pinched=False,
            is_scrolling=False,
            pointer_pos=(0.8, 0.8),
            hold_progress=0.5,
            held_duration=1.0,
        )
        pos = self.ctrl.handle_gesture(holding_result, (0.8, 0.8))
        self.assertEqual(pos, init_pos)


class TestGestureRecognizer(unittest.TestCase):
    def _create_mock_hand(self, thumb_pos=(0.1, 0.1), index_pos=(0.5, 0.5), middle_pos=(0.6, 0.6)):
        raw = np.zeros((21, 3), dtype=np.float32)
        pixel = np.zeros((21, 2), dtype=np.float32)
        norm = np.zeros((21, 3), dtype=np.float32)

        norm[HandLandmarkIndex.WRIST] = [0.0, 0.0, 0.0]
        norm[HandLandmarkIndex.MIDDLE_MCP] = [0.0, -1.0, 0.0]

        norm[HandLandmarkIndex.THUMB_TIP, :2] = thumb_pos
        norm[HandLandmarkIndex.INDEX_TIP, :2] = index_pos
        norm[HandLandmarkIndex.MIDDLE_TIP, :2] = middle_pos

        raw[HandLandmarkIndex.THUMB_TIP, :2] = thumb_pos
        raw[HandLandmarkIndex.INDEX_TIP, :2] = index_pos

        return HandData(
            raw_landmarks=raw,
            pixel_landmarks=pixel,
            normalized_landmarks=norm,
            hand_scale=100.0,
            handedness="Right",
            confidence=0.99,
        )

    def test_hold_2_seconds_to_unlock_movement(self):
        cfg = InteractionConfig(tracking_mode=TrackingMode.PINCH_TO_MOVE, hold_to_move_duration=2.0)
        rec = GestureRecognizer(cfg)
        pinched_hand = self._create_mock_hand(thumb_pos=(0.0, 0.0), index_pos=(0.02, 0.02))

        # 1. Start pinch at t=1.0s
        res = rec.process(pinched_hand, timestamp=1.00)
        self.assertEqual(res.state, GestureState.HOLDING_TO_MOVE)
        self.assertFalse(res.is_movement_engaged)

        # 2. At t=2.0s (1.0s elapsed) -> Still holding, not engaged
        res = rec.process(pinched_hand, timestamp=2.00)
        self.assertEqual(res.state, GestureState.HOLDING_TO_MOVE)
        self.assertFalse(res.is_movement_engaged)
        self.assertAlmostEqual(res.hold_progress, 0.5, delta=0.01)

        # 3. At t=3.0s (2.0s elapsed) -> Unlocked & Engaged!
        res = rec.process(pinched_hand, timestamp=3.00)
        self.assertEqual(res.state, GestureState.PINCH_MOVE)
        self.assertTrue(res.is_movement_engaged)
        self.assertEqual(res.event, GestureEvent.PINCH_ENGAGED)

    def test_double_tap_pinch_for_left_click(self):
        cfg = InteractionConfig(tracking_mode=TrackingMode.PINCH_TO_MOVE, tap_max_duration=0.35, double_tap_interval=0.55)
        rec = GestureRecognizer(cfg)
        open_hand = self._create_mock_hand(thumb_pos=(0.0, 0.0), index_pos=(0.2, 0.2))
        pinched_hand = self._create_mock_hand(thumb_pos=(0.0, 0.0), index_pos=(0.02, 0.02))

        # 1. First quick tap (0.1s duration)
        rec.process(pinched_hand, timestamp=1.00)
        res = rec.process(open_hand, timestamp=1.10)
        # Should record first tap and NOT click yet
        self.assertEqual(res.event, GestureEvent.TAP_FIRST)

        # 2. Second quick tap within 0.3s
        rec.process(pinched_hand, timestamp=1.25)
        res = rec.process(open_hand, timestamp=1.35)
        # Second tap triggers LEFT CLICK!
        self.assertEqual(res.event, GestureEvent.LEFT_CLICK)


class TestHandTracker(unittest.TestCase):
    def test_tracker_initialization(self):
        tracker = HandTracker()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        hands = tracker.process_frame(dummy_frame)
        self.assertIsInstance(hands, list)


if __name__ == "__main__":
    unittest.main()

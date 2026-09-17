"""
OS-level mouse controller using PyAutoGUI.
Handles coordinate mapping from camera space to screen space,
applies high-precision One-Euro smoothing, executes clicks/double-taps/drags/scrolls,
and enforces failsafe boundaries.
"""
from typing import Optional, Tuple
import numpy as np
import pyautogui

from config import InteractionConfig, SafetyConfig, SmoothingConfig, TrackingMode
from gesture_recognizer import GestureEvent, GestureResult
from smoothing import CursorSmoother


class MouseController:
    """Controls OS mouse cursor based on gesture events and coordinates with precision smoothing."""

    def __init__(
        self,
        interaction_cfg: Optional[InteractionConfig] = None,
        smoothing_cfg: Optional[SmoothingConfig] = None,
        safety_cfg: Optional[SafetyConfig] = None,
    ):
        self.interaction_cfg = interaction_cfg or InteractionConfig()
        self.smoothing_cfg = smoothing_cfg or SmoothingConfig()
        self.safety_cfg = safety_cfg or SafetyConfig()

        # PyAutoGUI settings for real-time responsiveness
        pyautogui.FAILSAFE = self.safety_cfg.failsafe
        pyautogui.PAUSE = 0.0  # Remove default 0.1s pause

        self.screen_width, self.screen_height = pyautogui.size()
        self.smoother = CursorSmoother(self.smoothing_cfg)
        self.enabled = True
        self.is_mouse_down = False
        self.current_screen_pos = (self.screen_width // 2, self.screen_height // 2)

        # Engagement and relative clutch tracking memory
        self.was_engaged = False
        self.last_pointer_norm: Optional[Tuple[float, float]] = None

    def set_enabled(self, enabled: bool):
        """Enable or disable OS mouse control (e.g. during pause)."""
        self.enabled = enabled
        if not enabled and self.is_mouse_down:
            self.mouse_up()
        if not enabled:
            self.smoother.reset()
            self.was_engaged = False
            self.last_pointer_norm = None

    def map_to_screen(self, norm_x: float, norm_y: float) -> Tuple[float, float]:
        """
        Map normalized camera coordinates [0..1] to full screen coordinates [0..W, 0..H],
        accounting for active interaction margins to prevent arm strain.
        """
        mx = self.interaction_cfg.margin_x
        my = self.interaction_cfg.margin_y

        # Clamp within active area
        clamped_x = np.clip(norm_x, mx, 1.0 - mx)
        clamped_y = np.clip(norm_y, my, 1.0 - my)

        # Scale active area to full screen dimensions
        screen_x = np.interp(clamped_x, [mx, 1.0 - mx], [0, self.screen_width - 1])
        screen_y = np.interp(clamped_y, [my, 1.0 - my], [0, self.screen_height - 1])

        return float(screen_x), float(screen_y)

    def move_cursor(self, raw_screen_x: float, raw_screen_y: float) -> Tuple[int, int]:
        """Apply precision smoothing and move OS cursor."""
        smooth_x, smooth_y = self.smoother.smooth(raw_screen_x, raw_screen_y)
        target_x = int(np.clip(smooth_x, 0, self.screen_width - 1))
        target_y = int(np.clip(smooth_y, 0, self.screen_height - 1))

        self.current_screen_pos = (target_x, target_y)

        if self.enabled:
            try:
                pyautogui.moveTo(target_x, target_y, _pause=False)
            except pyautogui.FailSafeException:
                print("\n[FAILSAFE] Cursor reached screen corner. Pausing mouse control.")
                self.set_enabled(False)

        return target_x, target_y

    def move_relative(self, delta_x: float, delta_y: float) -> Tuple[int, int]:
        """Move cursor relative to its current position (clutch mode)."""
        cur_x, cur_y = self.current_screen_pos
        smooth_dx, smooth_dy = self.smoother.smooth(delta_x, delta_y)

        target_x = int(np.clip(cur_x + smooth_dx, 0, self.screen_width - 1))
        target_y = int(np.clip(cur_y + smooth_dy, 0, self.screen_height - 1))

        self.current_screen_pos = (target_x, target_y)

        if self.enabled:
            try:
                pyautogui.moveTo(target_x, target_y, _pause=False)
            except pyautogui.FailSafeException:
                print("\n[FAILSAFE] Cursor reached screen corner. Pausing mouse control.")
                self.set_enabled(False)

        return target_x, target_y

    def click(self, button: str = "left"):
        """Perform a single click."""
        if not self.enabled:
            return
        try:
            pyautogui.click(button=button, _pause=False)
        except pyautogui.FailSafeException:
            self.set_enabled(False)

    def double_click(self):
        """Perform a double click."""
        if not self.enabled:
            return
        try:
            pyautogui.doubleClick(_pause=False)
        except pyautogui.FailSafeException:
            self.set_enabled(False)

    def mouse_down(self, button: str = "left"):
        """Press and hold mouse button."""
        if not self.enabled or self.is_mouse_down:
            return
        try:
            pyautogui.mouseDown(button=button, _pause=False)
            self.is_mouse_down = True
        except pyautogui.FailSafeException:
            self.set_enabled(False)

    def mouse_up(self, button: str = "left"):
        """Release held mouse button."""
        if not self.is_mouse_down:
            return
        try:
            pyautogui.mouseUp(button=button, _pause=False)
            self.is_mouse_down = False
        except pyautogui.FailSafeException:
            self.set_enabled(False)

    def scroll(self, amount: float):
        """Scroll vertical wheel."""
        if not self.enabled or abs(amount) < 0.1:
            return
        try:
            clicks = int(amount)
            pyautogui.scroll(clicks, _pause=False)
        except pyautogui.FailSafeException:
            self.set_enabled(False)

    def handle_gesture(self, gesture: GestureResult, pointer_norm: Tuple[float, float]) -> Tuple[int, int]:
        """
        Main entry: update cursor position and execute click/double-tap/scroll events.
        Cursor only moves when 2.0s hold is reached (is_movement_engaged is True).
        """
        mode = self.interaction_cfg.tracking_mode

        # 1. Cursor Movement handling (Only active when 2s hold engaged)
        if gesture.is_movement_engaged:
            if not self.was_engaged:
                self.smoother.reset()
                self.was_engaged = True
                self.last_pointer_norm = pointer_norm

            if mode == TrackingMode.RELATIVE_PINCH:
                if self.last_pointer_norm is not None:
                    dx = (pointer_norm[0] - self.last_pointer_norm[0]) * self.screen_width * self.interaction_cfg.relative_sensitivity
                    dy = (pointer_norm[1] - self.last_pointer_norm[1]) * self.screen_height * self.interaction_cfg.relative_sensitivity
                    self.move_relative(dx, dy)
                self.last_pointer_norm = pointer_norm
            else:
                # Absolute Pinch-to-Move or Continuous with precision smoothing
                raw_x, raw_y = self.map_to_screen(pointer_norm[0], pointer_norm[1])
                self.move_cursor(raw_x, raw_y)
        else:
            if self.was_engaged:
                self.was_engaged = False
                self.smoother.reset()
                self.last_pointer_norm = None

        # 2. Event triggers (Double-Tap Left Click, Right Click, Scroll)
        if gesture.event == GestureEvent.LEFT_CLICK:
            self.click(button="left")
        elif gesture.event == GestureEvent.RIGHT_CLICK:
            self.click(button="right")
        elif gesture.event == GestureEvent.SCROLL:
            self.scroll(gesture.scroll_delta)

        return self.current_screen_pos

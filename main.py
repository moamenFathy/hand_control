"""
Hand Gesture Desktop Controller - Main Application
Optimized for instant, natural Pinch-to-Move and responsive Tap-to-Click.
"""
import argparse
import sys
import time
from typing import List, Optional, Tuple
import cv2
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


class CalibrationWizard:
    """Interactive 2-step calibration for user-specific pinch thresholds."""

    def __init__(self, duration_per_step: float = 2.5):
        self.duration_per_step = duration_per_step
        self.is_active = False
        self.step = 0  # 1: Open hand, 2: Closed pinch
        self.step_start_time = 0.0
        self.open_samples: List[float] = []
        self.pinch_samples: List[float] = []

    def start(self):
        self.is_active = True
        self.step = 1
        self.step_start_time = time.perf_counter()
        self.open_samples.clear()
        self.pinch_samples.clear()

    def update(self, pinch_dist: float) -> Tuple[bool, str]:
        """Returns (is_finished, status_message)."""
        if not self.is_active:
            return False, ""

        now = time.perf_counter()
        elapsed = now - self.step_start_time

        if self.step == 1:
            self.open_samples.append(pinch_dist)
            remaining = max(0.0, self.duration_per_step - elapsed)
            msg = f"CALIBRATION (1/2): Hold hand wide OPEN... {remaining:.1f}s"
            if elapsed >= self.duration_per_step:
                self.step = 2
                self.step_start_time = now
            return False, msg

        elif self.step == 2:
            self.pinch_samples.append(pinch_dist)
            remaining = max(0.0, self.duration_per_step - elapsed)
            msg = f"CALIBRATION (2/2): Hold fingers PINCHED together... {remaining:.1f}s"
            if elapsed >= self.duration_per_step:
                self.is_active = False
                return True, "Calibration Complete!"
            return False, msg

        return False, ""

    def get_calibrated_thresholds(self) -> Tuple[float, float]:
        open_avg = float(np.median(self.open_samples)) if self.open_samples else 1.00
        pinch_avg = float(np.median(self.pinch_samples)) if self.pinch_samples else 0.20
        return open_avg, pinch_avg


class ClickRipple:
    """Animated visual ripple effect when a click occurs."""

    def __init__(self, pos: Tuple[int, int], color: Tuple[int, int, int] = (0, 255, 0), duration: float = 0.35):
        self.pos = pos
        self.color = color
        self.start_time = time.perf_counter()
        self.duration = duration

    def is_alive(self) -> bool:
        return (time.perf_counter() - self.start_time) < self.duration

    def draw(self, frame: np.ndarray):
        elapsed = time.perf_counter() - self.start_time
        progress = elapsed / self.duration
        if progress < 1.0:
            radius = int(10 + progress * 40)
            thickness = max(1, int(3 * (1.0 - progress)))
            cv2.circle(frame, self.pos, radius, self.color, thickness, cv2.LINE_AA)


class HandGestureApp:
    """Main application manager coordinating capture, tracking, gestures, and OS mouse."""

    def __init__(
        self,
        camera_cfg: Optional[CameraConfig] = None,
        tracking_cfg: Optional[TrackingConfig] = None,
        interaction_cfg: Optional[InteractionConfig] = None,
        smoothing_cfg: Optional[SmoothingConfig] = None,
        safety_cfg: Optional[SafetyConfig] = None,
        show_window: bool = True,
    ):
        self.camera_cfg = camera_cfg or CameraConfig()
        self.tracking_cfg = tracking_cfg or TrackingConfig()
        self.interaction_cfg = interaction_cfg or InteractionConfig()
        self.smoothing_cfg = smoothing_cfg or SmoothingConfig()
        self.safety_cfg = safety_cfg or SafetyConfig()
        self.show_window = show_window

        self.tracker = HandTracker(self.tracking_cfg)
        self.recognizer = GestureRecognizer(self.interaction_cfg)
        self.mouse = MouseController(
            self.interaction_cfg, self.smoothing_cfg, self.safety_cfg
        )
        self.calibrator = CalibrationWizard()
        self.ripples: List[ClickRipple] = []

        self.is_paused = False
        self.fps = 0.0
        self.prev_frame_time = time.perf_counter()
        self.status_notification = "Ready - Pinch to Move | Tap to Click"
        self.notification_time = time.perf_counter()

    def notify(self, message: str, duration: float = 2.0):
        self.status_notification = message
        self.notification_time = time.perf_counter()

    def toggle_mode(self):
        """Cycle through tracking modes."""
        modes = [
            TrackingMode.PINCH_TO_MOVE,
            TrackingMode.RELATIVE_PINCH,
            TrackingMode.CONTINUOUS,
        ]
        cur_idx = modes.index(self.interaction_cfg.tracking_mode)
        next_mode = modes[(cur_idx + 1) % len(modes)]
        self.interaction_cfg.tracking_mode = next_mode
        self.recognizer.config.tracking_mode = next_mode
        self.mouse.interaction_cfg.tracking_mode = next_mode
        mode_names = {
            TrackingMode.PINCH_TO_MOVE: "Pinch-to-Move (Absolute)",
            TrackingMode.RELATIVE_PINCH: "Pinch-to-Move (Relative Clutch)",
            TrackingMode.CONTINUOUS: "Continuous Pointing",
        }
        name = mode_names[next_mode]
        print(f"Tracking mode changed to: {name}")
        self.notify(f"Mode: {name}", duration=2.5)

    def adjust_region_size(self, delta: float):
        """Expand (negative margin) or shrink (positive margin) active screen box."""
        new_mx = np.clip(self.interaction_cfg.margin_x + delta, 0.01, 0.35)
        new_my = np.clip(self.interaction_cfg.margin_y + delta, 0.01, 0.35)
        self.interaction_cfg.margin_x = float(new_mx)
        self.interaction_cfg.margin_y = float(new_my)
        self.mouse.interaction_cfg.margin_x = float(new_mx)
        self.mouse.interaction_cfg.margin_y = float(new_my)
        w_pct = int((1.0 - 2 * new_mx) * 100)
        h_pct = int((1.0 - 2 * new_my) * 100)
        self.notify(f"Active Region: {w_pct}% x {h_pct}%", duration=1.5)

    def run(self):
        """Main camera processing and interaction loop."""
        print("=" * 65)
        print(" Hand Gesture Desktop Controller")
        print(" Gestures:")
        print("   - Pinch & Move: Smooth Cursor Movement (0ms lag)")
        print("   - Quick Pinch Tap: Left Click")
        print("   - Double Pinch Tap: Double Click")
        print("   - Thumb + Middle Pinch: Right Click")
        print("   - Two Fingers Up/Down: Scroll")
        print(" Controls:")
        print("   [SPACE]   - Pause / Resume mouse control")
        print("   [M]       - Toggle Mode (Pinch-to-Move / Relative / Continuous)")
        print("   [+] / [-] - Expand / Shrink active screen region")
        print("   [C]       - Start live calibration wizard")
        print("   [ESC/Q]   - Quit application")
        print(" Failsafe: Slam cursor into any screen corner to pause.")
        print("=" * 65)

        cap = cv2.VideoCapture(self.camera_cfg.camera_index, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.camera_cfg.camera_index)

        if not cap.isOpened():
            print(f"ERROR: Unable to open webcam index {self.camera_cfg.camera_index}.")
            print("Please ensure your webcam is connected and not in use by another app.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_cfg.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_cfg.frame_height)
        cap.set(cv2.CAP_PROP_FPS, self.camera_cfg.fps)

        window_name = "Hand Gesture Desktop Controller"
        if self.show_window:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success or frame is None:
                    print("Waiting for camera frame...")
                    time.sleep(0.05)
                    continue

                now = time.perf_counter()
                dt = now - self.prev_frame_time
                self.prev_frame_time = now
                if dt > 0:
                    self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)

                if self.camera_cfg.flip_horizontal:
                    frame = cv2.flip(frame, 1)

                h, w, _ = frame.shape

                # 1. Process hand tracking
                hands = self.tracker.process_frame(frame)
                primary_hand = hands[0] if hands else None

                # 2. Process gestures
                gesture = self.recognizer.process(primary_hand, timestamp=now)

                # Visual click ripples and notifications
                px = int(gesture.pointer_pos[0] * w)
                py = int(gesture.pointer_pos[1] * h)

                if gesture.event == GestureEvent.CLICK:
                    self.notify("👆 LEFT CLICK!", duration=1.0)
                    self.ripples.append(ClickRipple((px, py), color=(0, 255, 0)))
                elif gesture.event == GestureEvent.DOUBLE_CLICK:
                    self.notify("⚡ DOUBLE CLICK!", duration=1.2)
                    self.ripples.append(ClickRipple((px, py), color=(0, 255, 255)))
                elif gesture.event == GestureEvent.RIGHT_CLICK:
                    self.notify("👉 RIGHT CLICK!", duration=1.0)
                    self.ripples.append(ClickRipple((px, py), color=(255, 150, 0)))

                # 3. Handle calibration if active
                if self.calibrator.is_active:
                    finished, cal_msg = self.calibrator.update(gesture.pinch_distance)
                    self.notify(cal_msg, duration=0.5)
                    if finished:
                        open_val, pinch_val = self.calibrator.get_calibrated_thresholds()
                        self.recognizer.calibrate(open_val, pinch_val)
                        self.notify(
                            f"Calibrated: Open={open_val:.2f}, Pinch={pinch_val:.2f}",
                            duration=3.0,
                        )

                # 4. OS Mouse Control
                screen_pos = self.mouse.current_screen_pos
                if primary_hand is not None and not self.is_paused and not self.calibrator.is_active:
                    screen_pos = self.mouse.handle_gesture(gesture, gesture.pointer_pos)
                else:
                    if primary_hand is None:
                        self.recognizer.reset()
                        self.mouse.smoother.reset()

                # 5. Render HUD Overlay & Click Ripples
                if self.show_window:
                    self._draw_hud(frame, primary_hand, gesture, screen_pos, w, h)
                    self.ripples = [r for r in self.ripples if r.is_alive()]
                    for ripple in self.ripples:
                        ripple.draw(frame)
                    cv2.imshow(window_name, frame)

                # 6. Key bindings
                key = cv2.waitKey(1) & 0xFF
                if key in (self.safety_cfg.emergency_key, ord('q'), ord('Q')):
                    print("\nExiting Hand Gesture Controller.")
                    break
                elif key == self.safety_cfg.toggle_pause_key:
                    self.is_paused = not self.is_paused
                    self.mouse.set_enabled(not self.is_paused)
                    state_str = "PAUSED" if self.is_paused else "RESUMED"
                    print(f"Tracking {state_str}")
                    self.notify(f"Mouse Control {state_str}")
                elif key == self.safety_cfg.toggle_mode_key:
                    self.toggle_mode()
                elif key in (ord('+'), ord('=')):
                    self.adjust_region_size(-0.02)
                elif key in (ord('-'), ord('_')):
                    self.adjust_region_size(+0.02)
                elif key == self.safety_cfg.calibrate_key:
                    print("Starting calibration wizard...")
                    self.calibrator.start()

        finally:
            cap.release()
            if self.show_window:
                cv2.destroyAllWindows()
            if self.mouse.is_mouse_down:
                self.mouse.mouse_up()

    def _draw_hud(
        self,
        frame: np.ndarray,
        hand: Optional[HandData],
        gesture: GestureResult,
        screen_pos: Tuple[int, int],
        w: int,
        h: int,
    ):
        """Draw active boundary, hand skeleton, gestures, and status overlay."""
        # 1. Active Interaction Area Box
        mx = int(self.interaction_cfg.margin_x * w)
        my = int(self.interaction_cfg.margin_y * h)
        box_color = (0, 255, 0) if gesture.is_movement_engaged else (90, 90, 90)
        thickness = 2 if gesture.is_movement_engaged else 1
        cv2.rectangle(frame, (mx, my), (w - mx, h - my), box_color, thickness, cv2.LINE_AA)

        region_w_pct = int((1.0 - 2 * self.interaction_cfg.margin_x) * 100)
        region_h_pct = int((1.0 - 2 * self.interaction_cfg.margin_y) * 100)
        cv2.putText(
            frame,
            f"Active Region ({region_w_pct}% x {region_h_pct}%) [+/- resize]",
            (mx + 5, my - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            box_color,
            1,
            cv2.LINE_AA,
        )

        # 2. Draw Hand Skeleton & Landmarks
        if hand is not None:
            is_pinching = gesture.is_left_pinched or gesture.is_right_pinched
            HandTracker.draw_hand(
                frame,
                hand,
                highlight_index_tip=True,
                pinch_active=is_pinching,
            )

            # Draw target cursor point
            px = int(gesture.pointer_pos[0] * w)
            py = int(gesture.pointer_pos[1] * h)

            if gesture.is_movement_engaged:
                # Green active reticle
                cv2.circle(frame, (px, py), 16, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.circle(frame, (px, py), 4, (0, 255, 0), -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (px, py), 8, (150, 150, 150), 1, cv2.LINE_AA)

        # 3. Top Info Bar (Semi-transparent Header)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 54), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # FPS
        cv2.putText(
            frame,
            f"FPS: {self.fps:.0f}",
            (15, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 200),
            1,
            cv2.LINE_AA,
        )

        # Mode Badge
        mode_abbr = {
            TrackingMode.PINCH_TO_MOVE: "PINCH-TO-MOVE",
            TrackingMode.RELATIVE_PINCH: "RELATIVE CLUTCH",
            TrackingMode.CONTINUOUS: "CONTINUOUS",
        }.get(self.interaction_cfg.tracking_mode, "UNKNOWN")
        cv2.putText(
            frame,
            f"Mode: {mode_abbr}",
            (15, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (180, 220, 255),
            1,
            cv2.LINE_AA,
        )

        # State Badge
        if self.is_paused:
            badge_text = "PAUSED [SPACE to resume]"
            badge_color = (0, 0, 255)
        elif self.calibrator.is_active:
            badge_text = "CALIBRATING..."
            badge_color = (0, 165, 255)
        elif hand is None:
            badge_text = "NO HAND DETECTED"
            badge_color = (120, 120, 120)
        elif gesture.state == GestureState.PINCH_MOVE:
            badge_text = "🟢 PINCH ACTIVE: MOVING CURSOR"
            badge_color = (0, 255, 0)
        elif gesture.state == GestureState.RIGHT_PINCHING:
            badge_text = "RIGHT PINCH"
            badge_color = (255, 150, 0)
        elif gesture.state == GestureState.SCROLLING:
            badge_text = "SCROLLING"
            badge_color = (255, 255, 0)
        else:
            badge_text = "RELAXED (Pinch to Move | Tap to Click)"
            badge_color = (180, 180, 180)

        cv2.putText(
            frame,
            badge_text,
            (180, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            badge_color,
            2,
            cv2.LINE_AA,
        )

        # Cursor Coordinates on Screen
        cv2.putText(
            frame,
            f"Cursor: ({screen_pos[0]}, {screen_pos[1]})",
            (w - 210, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        # 4. Pinch Distance Meter (Bottom-Left)
        if hand is not None:
            bar_x, bar_y, bar_w, bar_h = 15, h - 54, 150, 14
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
            # Scale 0.0 to 1.0 ratio
            fill_w = int(np.clip(gesture.pinch_distance / 1.0, 0.0, 1.0) * bar_w)
            meter_color = (0, 255, 0) if gesture.is_left_pinched else (180, 180, 180)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), meter_color, -1)
            
            # Trigger line (Green at pinch_start_threshold)
            start_x = int((self.interaction_cfg.pinch_start_threshold / 1.0) * bar_w)
            cv2.line(frame, (bar_x + start_x, bar_y - 2), (bar_x + start_x, bar_y + bar_h + 2), (0, 255, 0), 2)
            
            # Release line (Red at pinch_release_threshold)
            rel_x = int((self.interaction_cfg.pinch_release_threshold / 1.0) * bar_w)
            cv2.line(frame, (bar_x + rel_x, bar_y - 2), (bar_x + rel_x, bar_y + bar_h + 2), (0, 0, 255), 2)
            
            status_tag = "PINCHED" if gesture.is_left_pinched else "OPEN"
            cv2.putText(
                frame,
                f"Pinch Ratio: {gesture.pinch_distance:.2f} ({status_tag})",
                (bar_x, bar_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

        # 5. Bottom Notification / Hotkey Tips
        bot_overlay = frame.copy()
        cv2.rectangle(bot_overlay, (0, h - 26), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(bot_overlay, 0.75, frame, 0.25, 0, frame)

        if time.perf_counter() - self.notification_time < 2.5:
            display_msg = self.status_notification
            msg_color = (0, 255, 255)
        else:
            display_msg = "[SPACE] Pause | [M] Mode | [+/-] Resize Box | [C] Calibrate | [ESC/Q] Quit"
            msg_color = (180, 180, 180)

        cv2.putText(
            frame,
            display_msg,
            (15, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            msg_color,
            1,
            cv2.LINE_AA,
        )


def main():
    parser = argparse.ArgumentParser(description="Hand Gesture Desktop Controller")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Camera width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Camera height (default: 480)")
    parser.add_argument(
        "--mode",
        choices=["pinch_to_move", "relative_pinch", "continuous"],
        default="pinch_to_move",
        help="Initial control mode",
    )
    parser.add_argument("--no-gui", action="store_true", help="Run without preview window")
    args = parser.parse_args()

    mode_map = {
        "pinch_to_move": TrackingMode.PINCH_TO_MOVE,
        "relative_pinch": TrackingMode.RELATIVE_PINCH,
        "continuous": TrackingMode.CONTINUOUS,
    }

    camera_cfg = CameraConfig(
        camera_index=args.camera,
        frame_width=args.width,
        frame_height=args.height,
    )
    interaction_cfg = InteractionConfig(
        tracking_mode=mode_map[args.mode],
    )

    app = HandGestureApp(
        camera_cfg=camera_cfg,
        interaction_cfg=interaction_cfg,
        show_window=not args.no_gui,
    )
    app.run()


if __name__ == "__main__":
    main()

"""
Configuration and settings for the Hand Gesture Desktop Controller.
Optimized with finger-proportional normalization and robust pinch detection.
"""
from dataclasses import dataclass
from enum import Enum


class TrackingMode(Enum):
    PINCH_TO_MOVE = "pinch_to_move"        # Mouse moves when forefinger and thumb are held pinched (Absolute)
    RELATIVE_PINCH = "relative_pinch"      # Relative mouse movement while pinching & holding (Trackpad clutch)
    CONTINUOUS = "continuous"              # Mouse tracks pointing index finger continuously


@dataclass
class CameraConfig:
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    fps: int = 30
    flip_horizontal: bool = True  # Mirror camera feed for intuitive control


@dataclass
class TrackingConfig:
    max_num_hands: int = 1
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.7
    model_complexity: int = 1


@dataclass
class InteractionConfig:
    # Primary control mode
    tracking_mode: TrackingMode = TrackingMode.PINCH_TO_MOVE

    # Margins inside camera frame to map to full screen (0.0 to 0.5)
    margin_x: float = 0.06
    margin_y: float = 0.08

    # Speed / sensitivity multipliers for precision control
    cursor_speed_factor: float = 0.90     # Calm, accurate absolute movement
    relative_sensitivity: float = 1.30    # Steady relative clutch control

    # Pinch Distance Thresholds (Ratio of Thumb-Index distance to Index Finger Length)
    # Touching: ~0.10 - 0.30 | Open hand: ~0.75 - 2.00+
    pinch_start_threshold: float = 0.35   # Trigger pinch below 0.35
    pinch_release_threshold: float = 0.48 # Release pinch above 0.48

    # Right-click pinch (Thumb tip to Middle tip ratio)
    right_pinch_start_threshold: float = 0.35
    right_pinch_release_threshold: float = 0.48

    # Click Timings
    tap_max_duration: float = 0.35        # Pinch duration <= 0.35s registers as a Click
    double_click_interval: float = 0.45   # Time window between taps for Double Click

    # Scrolling
    scroll_threshold_y: float = 0.03      # Minimum vertical movement to trigger scroll
    scroll_speed: int = 15                # Units to scroll per tick (steady, controlled)


@dataclass
class SmoothingConfig:
    smoothing_factor: float = 6.0
    deadzone_pixels: float = 3.0
    use_one_euro: bool = True
    min_cutoff: float = 0.8
    beta: float = 0.030
    d_cutoff: float = 1.0


@dataclass
class SafetyConfig:
    failsafe: bool = True                 # PyAutoGUI failsafe (slam mouse to corner to abort)
    emergency_key: int = 27               # ESC key to exit
    toggle_pause_key: int = ord(' ')      # Spacebar to pause/resume tracking
    toggle_mode_key: int = ord('m')       # 'm' key to toggle tracking modes
    calibrate_key: int = ord('c')         # 'c' key to trigger live calibration
    expand_region_key: int = ord('+')     # '+' or '=' to expand active region
    shrink_region_key: int = ord('-')     # '-' to shrink active region

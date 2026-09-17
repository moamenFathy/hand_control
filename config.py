"""
Configuration and settings for the Hand Gesture Desktop Controller.
"""
from dataclasses import dataclass
from enum import Enum


class TrackingMode(Enum):
    PINCH_TO_MOVE = "pinch_to_move"        # Mouse only moves after holding pinch for 2 seconds (Absolute)
    RELATIVE_PINCH = "relative_pinch"      # Relative mouse movement after 2s hold (Trackpad clutch)
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

    # Hold duration required to engage mouse movement (in seconds)
    hold_to_move_duration: float = 2.0

    # Margins inside camera frame to map to full screen (0.0 to 0.5)
    margin_x: float = 0.06
    margin_y: float = 0.08

    # Speed / sensitivity multipliers for precision control
    cursor_speed_factor: float = 0.85     # Toned down for calm, accurate absolute movement
    relative_sensitivity: float = 1.15    # Lowered for precise, steady relative clutch control

    # Pinch (Thumb tip to Index tip) distance threshold normalized by hand size
    pinch_start_threshold: float = 0.050
    pinch_release_threshold: float = 0.080

    # Right-click pinch (Thumb tip to Middle tip)
    right_pinch_start_threshold: float = 0.050
    right_pinch_release_threshold: float = 0.080

    # Click Timing (Double-tap pinch for left click)
    tap_max_duration: float = 0.35        # Max pinch duration to register as a single tap
    double_tap_interval: float = 0.55     # Max seconds between 2 pinch taps to trigger Left Click

    # Scrolling
    scroll_threshold_y: float = 0.03      # Minimum movement to trigger scroll in two-finger mode
    scroll_speed: int = 15                # Units to scroll per tick (steady, controlled)


@dataclass
class SmoothingConfig:
    # Higher value = smoother and calmer; lower value = more raw/jumpy
    smoothing_factor: float = 7.5

    # Minimum screen pixel movement to apply (eliminates micro-jitter at rest for high precision)
    deadzone_pixels: float = 3.5

    # One-Euro Filter parameters (Tuned for rock-solid precision & accuracy)
    use_one_euro: bool = True
    min_cutoff: float = 0.7               # Lower cutoff for smooth, steady low-speed positioning
    beta: float = 0.025                   # Gentle velocity slope for controlled, accurate movements
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

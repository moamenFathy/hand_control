"""
Gesture recognition engine and state machine.
Analyzes finger-proportional hand landmarks to recognize:
- Instant Pinch-to-Move (Pinch forefinger & thumb to move mouse with 0ms lag)
- Quick Pinch Tap: Left Click
- Double Pinch Tap: Double Click
- Thumb + Middle Finger Pinch: Right Click
- Two-Finger Vertical Drag: Scroll Up / Down
"""
from enum import Enum, auto
import time
from typing import NamedTuple, Optional, Tuple
import numpy as np

from config import InteractionConfig, TrackingMode
from hand_tracker import HandData, HandLandmarkIndex


class GestureState(Enum):
    IDLE = auto()
    DISENGAGED = auto()       # Hand visible, fingers relaxed/open (cursor frozen)
    PINCH_MOVE = auto()       # Forefinger + Thumb pinched (cursor moving in real-time)
    RIGHT_PINCHING = auto()   # Thumb + Middle finger pinch (Right Click)
    SCROLLING = auto()        # Two fingers extended scrolling vertically


class GestureEvent(Enum):
    NONE = auto()
    PINCH_START = auto()      # Moment forefinger and thumb touch
    PINCH_RELEASE = auto()    # Moment sustained pinch hold is released
    CLICK = auto()            # Quick pinch tap -> Left Click
    DOUBLE_CLICK = auto()     # Two quick pinch taps -> Double Click
    RIGHT_CLICK = auto()      # Thumb + Middle pinch -> Right Click
    SCROLL = auto()           # Two-finger vertical scroll


class GestureResult(NamedTuple):
    state: GestureState
    event: GestureEvent
    pinch_distance: float
    right_pinch_distance: float
    scroll_delta: float
    is_movement_engaged: bool   # True when mouse cursor should move
    is_left_pinched: bool       # Forefinger + thumb touching
    is_right_pinched: bool      # Middle + thumb touching
    is_scrolling: bool          # Two-finger scroll active
    pointer_pos: Tuple[float, float]  # Normalized (x, y) coordinates


class GestureRecognizer:
    """State machine for responsive hand gesture recognition and clicking."""

    def __init__(self, config: Optional[InteractionConfig] = None):
        self.config = config or InteractionConfig()

        # State tracking
        self.current_state = GestureState.IDLE
        self.is_left_pinched = False
        self.is_right_pinched = False
        self.is_scrolling = False

        # Timing memories
        self.pinch_start_time: Optional[float] = None
        self.last_click_time: float = 0.0

        # Scrolling reference
        self.last_scroll_y: Optional[float] = None

    def reset(self):
        """Reset state when hand tracking is lost."""
        self.current_state = GestureState.IDLE
        self.is_left_pinched = False
        self.is_right_pinched = False
        self.is_scrolling = False
        self.pinch_start_time = None
        self.last_click_time = 0.0
        self.last_scroll_y = None

    def calibrate(self, open_hand_dist: float, closed_pinch_dist: float):
        """Dynamically calibrate pinch thresholds based on individual user anatomy."""
        gap = max(0.10, open_hand_dist - closed_pinch_dist)
        self.config.pinch_start_threshold = float(closed_pinch_dist + gap * 0.35)
        self.config.pinch_release_threshold = float(closed_pinch_dist + gap * 0.65)
        self.config.right_pinch_start_threshold = self.config.pinch_start_threshold
        self.config.right_pinch_release_threshold = self.config.pinch_release_threshold

    def process(self, hand: Optional[HandData], timestamp: Optional[float] = None) -> GestureResult:
        if timestamp is None:
            timestamp = time.perf_counter()

        if hand is None:
            self.reset()
            return GestureResult(
                state=GestureState.IDLE,
                event=GestureEvent.NONE,
                pinch_distance=1.5,
                right_pinch_distance=1.5,
                scroll_delta=0.0,
                is_movement_engaged=False,
                is_left_pinched=False,
                is_right_pinched=False,
                is_scrolling=False,
                pointer_pos=(0.5, 0.5),
            )

        raw = hand.raw_landmarks

        thumb_tip = raw[HandLandmarkIndex.THUMB_TIP, :2]
        index_tip = raw[HandLandmarkIndex.INDEX_TIP, :2]
        middle_tip = raw[HandLandmarkIndex.MIDDLE_TIP, :2]
        ring_tip = raw[HandLandmarkIndex.RING_TIP, :2]
        pinky_tip = raw[HandLandmarkIndex.PINKY_TIP, :2]

        index_mcp = raw[HandLandmarkIndex.INDEX_MCP, :2]
        middle_mcp = raw[HandLandmarkIndex.MIDDLE_MCP, :2]
        ring_mcp = raw[HandLandmarkIndex.RING_MCP, :2]
        pinky_mcp = raw[HandLandmarkIndex.PINKY_MCP, :2]

        # Anatomical finger lengths for distance normalization
        index_len = max(0.04, float(np.linalg.norm(index_mcp - index_tip)))
        middle_len = max(0.04, float(np.linalg.norm(middle_mcp - middle_tip)))

        # Distance ratios relative to individual finger lengths (Invariant to tilt and distance)
        pinch_dist = float(np.linalg.norm(thumb_tip - index_tip)) / index_len
        right_pinch_dist = float(np.linalg.norm(thumb_tip - middle_tip)) / middle_len

        # Pointer position (Midpoint when pinched, index tip when pointing)
        pinch_midpoint = ((thumb_tip[0] + index_tip[0]) / 2.0, (thumb_tip[1] + index_tip[1]) / 2.0)
        pointer_pos = pinch_midpoint if self.is_left_pinched else (float(index_tip[0]), float(index_tip[1]))

        # Check finger extensions
        index_extended = index_tip[1] < index_mcp[1]
        middle_extended = middle_tip[1] < middle_mcp[1]
        ring_folded = ring_tip[1] > ring_mcp[1]
        pinky_folded = pinky_tip[1] > pinky_mcp[1]

        event = GestureEvent.NONE
        scroll_delta = 0.0

        # --- 1. Two-Finger Scroll Mode ---
        two_finger_dist = float(np.linalg.norm(index_tip - middle_tip)) / index_len
        scroll_pose = (
            index_extended
            and middle_extended
            and ring_folded
            and pinky_folded
            and two_finger_dist < 0.40
            and pinch_dist > self.config.pinch_release_threshold
        )

        if scroll_pose:
            self.is_scrolling = True
            current_y = (index_tip[1] + middle_tip[1]) / 2.0
            if self.last_scroll_y is not None:
                dy = current_y - self.last_scroll_y
                if abs(dy) >= self.config.scroll_threshold_y:
                    scroll_delta = -dy * self.config.scroll_speed * 50
                    event = GestureEvent.SCROLL
                    self.last_scroll_y = current_y
            else:
                self.last_scroll_y = current_y
            return GestureResult(
                state=GestureState.SCROLLING,
                event=event,
                pinch_distance=pinch_dist,
                right_pinch_distance=right_pinch_dist,
                scroll_delta=scroll_delta,
                is_movement_engaged=False,
                is_left_pinched=False,
                is_right_pinched=False,
                is_scrolling=True,
                pointer_pos=pointer_pos,
            )
        else:
            self.is_scrolling = False
            self.last_scroll_y = None

        # --- 2. Right Click (Thumb + Middle Pinch) ---
        if not self.is_left_pinched:
            if not self.is_right_pinched and right_pinch_dist <= self.config.right_pinch_start_threshold:
                self.is_right_pinched = True
            elif self.is_right_pinched and right_pinch_dist >= self.config.right_pinch_release_threshold:
                self.is_right_pinched = False
                event = GestureEvent.RIGHT_CLICK

            if self.is_right_pinched:
                return GestureResult(
                    state=GestureState.RIGHT_PINCHING,
                    event=event,
                    pinch_distance=pinch_dist,
                    right_pinch_distance=right_pinch_dist,
                    scroll_delta=0.0,
                    is_movement_engaged=False,
                    is_left_pinched=False,
                    is_right_pinched=True,
                    is_scrolling=False,
                    pointer_pos=pointer_pos,
                )

        # --- 3. Forefinger + Thumb Pinch & Tap-to-Click ---
        if not self.is_left_pinched:
            if pinch_dist <= self.config.pinch_start_threshold:
                self.is_left_pinched = True
                self.pinch_start_time = timestamp
                event = GestureEvent.PINCH_START
        else:
            if pinch_dist >= self.config.pinch_release_threshold:
                # Pinch released
                self.is_left_pinched = False
                pinch_duration = timestamp - (self.pinch_start_time or timestamp)
                self.pinch_start_time = None

                # If the pinch was a quick tap, trigger click / double-click
                if pinch_duration <= self.config.tap_max_duration:
                    time_since_last_click = timestamp - self.last_click_time
                    if time_since_last_click <= self.config.double_click_interval:
                        event = GestureEvent.DOUBLE_CLICK
                        self.last_click_time = 0.0
                    else:
                        event = GestureEvent.CLICK
                        self.last_click_time = timestamp
                else:
                    # Sustained movement hold finished
                    event = GestureEvent.PINCH_RELEASE
                    self.last_click_time = 0.0

        # Safety: If hand is visibly wide open (distance > release threshold), force disengage
        if pinch_dist >= self.config.pinch_release_threshold:
            self.is_left_pinched = False

        # Determine movement engagement and state
        if self.config.tracking_mode in (TrackingMode.PINCH_TO_MOVE, TrackingMode.RELATIVE_PINCH):
            is_movement_engaged = self.is_left_pinched
            state = GestureState.PINCH_MOVE if self.is_left_pinched else GestureState.DISENGAGED
        else:  # CONTINUOUS
            is_movement_engaged = True
            state = GestureState.PINCH_MOVE if self.is_left_pinched else GestureState.DISENGAGED

        return GestureResult(
            state=state,
            event=event,
            pinch_distance=pinch_dist,
            right_pinch_distance=right_pinch_dist,
            scroll_delta=0.0,
            is_movement_engaged=is_movement_engaged,
            is_left_pinched=self.is_left_pinched,
            is_right_pinched=self.is_right_pinched,
            is_scrolling=False,
            pointer_pos=pointer_pos,
        )

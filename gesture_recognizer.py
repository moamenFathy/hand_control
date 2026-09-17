"""
Gesture recognition engine and state machine.
Analyzes normalized hand landmarks to recognize:
- 2-Second Pinch Hold to Engage Cursor Movement
- Double-Tap Pinch to Left Click (2 quick pinch taps in succession)
- Right Click (Thumb + Middle finger pinch)
- Two-Finger Scroll (Index + Middle extended moving vertically)
"""
from enum import Enum, auto
import time
from typing import NamedTuple, Optional, Tuple
import numpy as np

from config import InteractionConfig, TrackingMode
from hand_tracker import HandData, HandLandmarkIndex


class GestureState(Enum):
    IDLE = auto()
    DISENGAGED = auto()         # Hand visible, but fingers open (cursor frozen)
    HOLDING_TO_MOVE = auto()    # Forefinger + Thumb pinched, counting up to 2.0s hold
    PINCH_MOVE = auto()         # 2.0s hold reached: cursor actively moving
    RIGHT_PINCHING = auto()     # Thumb + Middle finger pinch (Right Click)
    SCROLLING = auto()          # Two fingers extended scrolling vertically


class GestureEvent(Enum):
    NONE = auto()
    PINCH_START = auto()        # Moment forefinger and thumb touch
    PINCH_ENGAGED = auto()      # Moment 2.0s hold completes and cursor unlocks
    PINCH_RELEASE = auto()      # Moment sustained hold is released
    TAP_FIRST = auto()          # First pinch tap recorded
    LEFT_CLICK = auto()         # Second pinch tap completed -> Left Click executed!
    RIGHT_CLICK = auto()        # Thumb + Middle pinch
    SCROLL = auto()             # Two-finger vertical scroll


class GestureResult(NamedTuple):
    state: GestureState
    event: GestureEvent
    pinch_distance: float
    right_pinch_distance: float
    scroll_delta: float
    is_movement_engaged: bool     # True ONLY when 2.0s hold is reached
    is_left_pinched: bool         # Forefinger + thumb touching
    is_right_pinched: bool        # Middle + thumb touching
    is_scrolling: bool            # Two-finger scroll active
    pointer_pos: Tuple[float, float]    # Normalized (x, y) coordinates
    hold_progress: float          # 0.0 to 1.0 progress towards 2.0s move unlock
    held_duration: float          # Current elapsed hold seconds


class GestureRecognizer:
    """State machine for 2-Second Hold-to-Move and Double-Tap Left Click."""

    def __init__(self, config: Optional[InteractionConfig] = None):
        self.config = config or InteractionConfig()

        # State tracking
        self.current_state = GestureState.IDLE
        self.is_left_pinched = False
        self.is_right_pinched = False
        self.is_scrolling = False
        self.is_movement_unlocked = False

        # Timing memories
        self.pinch_start_time: Optional[float] = None
        self.last_tap_time: float = 0.0

        # Scrolling reference
        self.last_scroll_y: Optional[float] = None

    def reset(self):
        """Reset state when hand tracking is lost."""
        self.current_state = GestureState.IDLE
        self.is_left_pinched = False
        self.is_right_pinched = False
        self.is_scrolling = False
        self.is_movement_unlocked = False
        self.pinch_start_time = None
        self.last_tap_time = 0.0
        self.last_scroll_y = None

    def calibrate(self, open_hand_dist: float, closed_pinch_dist: float):
        """Dynamically calibrate pinch thresholds based on individual user anatomy."""
        gap = max(0.01, (open_hand_dist - closed_pinch_dist) * 0.3)
        self.config.pinch_start_threshold = closed_pinch_dist + gap * 0.5
        self.config.pinch_release_threshold = closed_pinch_dist + gap * 1.5
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
                pinch_distance=1.0,
                right_pinch_distance=1.0,
                scroll_delta=0.0,
                is_movement_engaged=False,
                is_left_pinched=False,
                is_right_pinched=False,
                is_scrolling=False,
                pointer_pos=(0.5, 0.5),
                hold_progress=0.0,
                held_duration=0.0,
            )

        norm = hand.normalized_landmarks
        raw = hand.raw_landmarks

        thumb_tip = norm[HandLandmarkIndex.THUMB_TIP, :2]
        index_tip = norm[HandLandmarkIndex.INDEX_TIP, :2]
        middle_tip = norm[HandLandmarkIndex.MIDDLE_TIP, :2]
        ring_tip = norm[HandLandmarkIndex.RING_TIP, :2]
        pinky_tip = norm[HandLandmarkIndex.PINKY_TIP, :2]

        index_mcp = norm[HandLandmarkIndex.INDEX_MCP, :2]
        middle_mcp = norm[HandLandmarkIndex.MIDDLE_MCP, :2]
        ring_mcp = norm[HandLandmarkIndex.RING_MCP, :2]
        pinky_mcp = norm[HandLandmarkIndex.PINKY_MCP, :2]

        # Raw camera normalized points for pointer tracking
        raw_thumb = raw[HandLandmarkIndex.THUMB_TIP, :2]
        raw_index = raw[HandLandmarkIndex.INDEX_TIP, :2]
        pinch_midpoint = ((raw_thumb[0] + raw_index[0]) / 2.0, (raw_thumb[1] + raw_index[1]) / 2.0)
        pointer_pos = pinch_midpoint if self.is_left_pinched else (float(raw_index[0]), float(raw_index[1]))

        # Distances (normalized by hand scale)
        pinch_dist = float(np.linalg.norm(thumb_tip - index_tip))
        right_pinch_dist = float(np.linalg.norm(thumb_tip - middle_tip))

        # Check finger extensions
        index_extended = index_tip[1] < index_mcp[1]
        middle_extended = middle_tip[1] < middle_mcp[1]
        ring_folded = ring_tip[1] > ring_mcp[1]
        pinky_folded = pinky_tip[1] > pinky_mcp[1]

        event = GestureEvent.NONE
        scroll_delta = 0.0

        # --- 1. Two-Finger Scroll Mode ---
        two_finger_dist = float(np.linalg.norm(index_tip - middle_tip))
        scroll_pose = (
            index_extended
            and middle_extended
            and ring_folded
            and pinky_folded
            and two_finger_dist < 0.12
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
                hold_progress=0.0,
                held_duration=0.0,
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
                    hold_progress=0.0,
                    held_duration=0.0,
                )

        # --- 3. Forefinger + Thumb Pinch (2-Second Hold to Move & Double-Tap to Click) ---
        hold_progress = 0.0
        held_duration = 0.0

        if not self.is_left_pinched:
            if pinch_dist <= self.config.pinch_start_threshold:
                self.is_left_pinched = True
                self.pinch_start_time = timestamp
                self.is_movement_unlocked = False
                event = GestureEvent.PINCH_START
        else:
            if pinch_dist >= self.config.pinch_release_threshold:
                # Pinch was released
                self.is_left_pinched = False
                pinch_duration = timestamp - (self.pinch_start_time or timestamp)
                self.pinch_start_time = None
                was_unlocked = self.is_movement_unlocked
                self.is_movement_unlocked = False

                if not was_unlocked and pinch_duration <= self.config.tap_max_duration:
                    # Short pinch tap occurred
                    time_since_last_tap = timestamp - self.last_tap_time
                    if time_since_last_tap <= self.config.double_tap_interval:
                        # 2nd quick pinch tap -> LEFT CLICK!
                        event = GestureEvent.LEFT_CLICK
                        self.last_tap_time = 0.0
                    else:
                        # 1st pinch tap recorded
                        event = GestureEvent.TAP_FIRST
                        self.last_tap_time = timestamp
                else:
                    # Released after holding/moving
                    self.last_tap_time = 0.0
                    event = GestureEvent.PINCH_RELEASE
            else:
                # Still holding pinch - calculate hold duration
                held_duration = timestamp - (self.pinch_start_time or timestamp)
                req_hold = max(0.1, self.config.hold_to_move_duration)
                hold_progress = min(1.0, held_duration / req_hold)

                if held_duration >= req_hold:
                    if not self.is_movement_unlocked:
                        self.is_movement_unlocked = True
                        event = GestureEvent.PINCH_ENGAGED

        # Determine movement engagement and state
        if self.config.tracking_mode in (TrackingMode.PINCH_TO_MOVE, TrackingMode.RELATIVE_PINCH):
            is_movement_engaged = self.is_left_pinched and self.is_movement_unlocked
            if is_movement_engaged:
                state = GestureState.PINCH_MOVE
            elif self.is_left_pinched:
                state = GestureState.HOLDING_TO_MOVE
            else:
                state = GestureState.DISENGAGED
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
            hold_progress=hold_progress,
            held_duration=held_duration,
        )

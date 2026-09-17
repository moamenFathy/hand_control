"""
Hand tracking module using MediaPipe Hand Landmarker.
Handles webcam frames, detects hand landmarks, normalizes coordinates,
and provides visualization drawing utilities.
"""
import os
import urllib.request
from typing import List, Optional, Tuple, NamedTuple
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import TrackingConfig


# Standard MediaPipe Hand Landmark indices
class HandLandmarkIndex:
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


# Hand skeleton connections for visualization
HAND_CONNECTIONS = [
    # Palm
    (0, 1), (0, 5), (0, 17), (5, 9), (9, 13), (13, 17),
    # Thumb
    (1, 2), (2, 3), (3, 4),
    # Index finger
    (5, 6), (6, 7), (7, 8),
    # Middle finger
    (9, 10), (10, 11), (11, 12),
    # Ring finger
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (17, 18), (18, 19), (19, 20)
]


class HandData(NamedTuple):
    # Raw normalized landmarks [21, 3] (x, y, z in 0..1 relative to image)
    raw_landmarks: np.ndarray
    # Pixel coordinates on the image frame [21, 2] (x, y in pixels)
    pixel_landmarks: np.ndarray
    # Wrist-relative & scale-normalized landmarks [21, 3] (invariant to camera distance)
    normalized_landmarks: np.ndarray
    # Hand scale (pixel distance between wrist and middle MCP)
    hand_scale: float
    # Handedness ('Left' or 'Right')
    handedness: str
    # Confidence score
    confidence: float


class HandTracker:
    """Wrapper around MediaPipe Hand Landmarker task."""

    MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    )

    def __init__(self, config: Optional[TrackingConfig] = None, model_path: Optional[str] = None):
        self.config = config or TrackingConfig()
        self.model_path = model_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task"
        )
        self._ensure_model_exists()

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=self.config.max_num_hands,
            min_hand_detection_confidence=self.config.min_detection_confidence,
            min_hand_presence_confidence=self.config.min_tracking_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def _ensure_model_exists(self):
        """Auto-download the MediaPipe hand landmarker model if missing."""
        if not os.path.exists(self.model_path):
            print(f"Downloading hand landmarker model to {self.model_path}...")
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            urllib.request.urlretrieve(self.MODEL_URL, self.model_path)
            print("Model downloaded successfully!")

    def process_frame(self, frame_bgr: np.ndarray) -> List[HandData]:
        """
        Process a BGR image from OpenCV and return a list of detected HandData.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = frame_bgr.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = self.landmarker.detect(mp_image)

        hands: List[HandData] = []
        if not detection_result.hand_landmarks:
            return hands

        for idx, hand_lms in enumerate(detection_result.hand_landmarks):
            # Extract 21 points
            raw = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms], dtype=np.float32)

            # Pixel coords
            pixel = np.zeros((21, 2), dtype=np.float32)
            pixel[:, 0] = raw[:, 0] * w
            pixel[:, 1] = raw[:, 1] * h

            # Handedness
            handedness = "Right"
            confidence = 1.0
            if detection_result.handedness and idx < len(detection_result.handedness):
                first_cat = detection_result.handedness[idx][0]
                handedness = first_cat.category_name
                confidence = first_cat.score

            # Normalize landmarks relative to wrist and scaled by hand size
            norm_lms, hand_scale = self._normalize_landmarks(raw, pixel)

            hands.append(
                HandData(
                    raw_landmarks=raw,
                    pixel_landmarks=pixel,
                    normalized_landmarks=norm_lms,
                    hand_scale=hand_scale,
                    handedness=handedness,
                    confidence=confidence,
                )
            )

        return hands

    def _normalize_landmarks(
        self, raw_landmarks: np.ndarray, pixel_landmarks: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Normalize landmarks:
        1. Shift origin (0, 0, 0) to the Wrist (landmark 0).
        2. Scale by the hand size (distance between wrist and middle MCP landmark 9).
        This guarantees invariant distances regardless of user proximity to webcam.
        """
        wrist_px = pixel_landmarks[HandLandmarkIndex.WRIST]
        middle_mcp_px = pixel_landmarks[HandLandmarkIndex.MIDDLE_MCP]
        hand_scale_px = float(np.linalg.norm(wrist_px - middle_mcp_px))

        # Safeguard against division by zero
        if hand_scale_px < 1e-4:
            hand_scale_px = 1.0

        # Normalization in normalized 3D space
        wrist_raw = raw_landmarks[HandLandmarkIndex.WRIST]
        middle_mcp_raw = raw_landmarks[HandLandmarkIndex.MIDDLE_MCP]
        scale_raw = float(np.linalg.norm(raw_landmarks[HandLandmarkIndex.WRIST, :2] - raw_landmarks[HandLandmarkIndex.MIDDLE_MCP, :2]))

        if scale_raw < 1e-5:
            scale_raw = 1.0

        normalized = (raw_landmarks - wrist_raw) / scale_raw
        return normalized, hand_scale_px

    @staticmethod
    def draw_hand(
        frame_bgr: np.ndarray,
        hand: HandData,
        highlight_index_tip: bool = True,
        pinch_active: bool = False,
    ) -> None:
        """
        Draw hand skeleton, landmarks, and indicators on an OpenCV frame.
        """
        pts = hand.pixel_landmarks.astype(int)

        # Draw bones / connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            pt1 = tuple(pts[start_idx])
            pt2 = tuple(pts[end_idx])
            cv2.line(frame_bgr, pt1, pt2, (200, 200, 200), 2, cv2.LINE_AA)

        # Draw landmark dots
        for i, pt in enumerate(pts):
            # Key tips get slightly larger circles
            if i in (HandLandmarkIndex.THUMB_TIP, HandLandmarkIndex.INDEX_TIP, HandLandmarkIndex.MIDDLE_TIP):
                radius = 6
                color = (0, 255, 255)
            else:
                radius = 4
                color = (255, 100, 50)
            cv2.circle(frame_bgr, tuple(pt), radius, color, -1, cv2.LINE_AA)

        # Highlight index tip (the pointer)
        if highlight_index_tip:
            index_tip = tuple(pts[HandLandmarkIndex.INDEX_TIP])
            thumb_tip = tuple(pts[HandLandmarkIndex.THUMB_TIP])

            # Draw pointer ring
            ring_color = (0, 255, 0) if pinch_active else (255, 200, 0)
            cv2.circle(frame_bgr, index_tip, 12, ring_color, 2, cv2.LINE_AA)

            # Draw pinch connection line
            cv2.line(frame_bgr, thumb_tip, index_tip, ring_color, 2, cv2.LINE_AA)

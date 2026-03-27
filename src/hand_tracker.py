"""MediaPipe wrapper for hand detection and landmark extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp

from .config import (
    DRAW_LANDMARKS_BY_DEFAULT,
    HAND_DETECTION_CONFIDENCE,
    HAND_TRACKING_CONFIDENCE,
    LANDMARK_CONNECTION_COLOR,
    LANDMARK_CONNECTION_THICKNESS,
    LANDMARK_POINT_COLOR,
    LANDMARK_POINT_RADIUS,
    MAX_NUM_HANDS,
)


@dataclass
class HandData:
    """Processed hand landmark information for a single detected hand."""

    pixel_landmarks: List[Tuple[int, int]]
    normalized_landmarks: List[Tuple[float, float]]
    handedness: str
    confidence: float


class HandTracker:
    """Detect a hand and convert landmarks into convenient coordinate lists."""

    def __init__(
        self,
        max_num_hands: int = MAX_NUM_HANDS,
        min_detection_confidence: float = HAND_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = HAND_TRACKING_CONFIDENCE,
        draw_landmarks: bool = DRAW_LANDMARKS_BY_DEFAULT,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._mp_draw = mp.solutions.drawing_utils
        self._connection_spec = self._mp_draw.DrawingSpec(
            color=LANDMARK_CONNECTION_COLOR,
            thickness=LANDMARK_CONNECTION_THICKNESS,
            circle_radius=LANDMARK_POINT_RADIUS,
        )
        self._point_spec = self._mp_draw.DrawingSpec(
            color=LANDMARK_POINT_COLOR,
            thickness=LANDMARK_CONNECTION_THICKNESS,
            circle_radius=LANDMARK_POINT_RADIUS,
        )
        self.draw_landmarks = draw_landmarks
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame, draw_landmarks: Optional[bool] = None):
        """Return an optionally annotated frame and the first detected hand."""
        draw_hand_overlay = self.draw_landmarks if draw_landmarks is None else draw_landmarks
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self._hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        hand_data: Optional[HandData] = None
        if results.multi_hand_landmarks and results.multi_handedness:
            hand_landmarks = results.multi_hand_landmarks[0]
            classification = results.multi_handedness[0].classification[0]
            frame_height, frame_width = frame.shape[:2]

            pixel_landmarks: List[Tuple[int, int]] = []
            normalized_landmarks: List[Tuple[float, float]] = []
            for landmark in hand_landmarks.landmark:
                pixel_landmarks.append((int(landmark.x * frame_width), int(landmark.y * frame_height)))
                normalized_landmarks.append((landmark.x, landmark.y))

            if draw_hand_overlay:
                self._mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self._mp_hands.HAND_CONNECTIONS,
                    self._connection_spec,
                    self._point_spec,
                )

            hand_data = HandData(
                pixel_landmarks=pixel_landmarks,
                normalized_landmarks=normalized_landmarks,
                handedness=classification.label,
                confidence=classification.score,
            )

        return frame, hand_data

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._hands.close()

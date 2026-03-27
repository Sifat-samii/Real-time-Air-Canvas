"""MediaPipe wrapper for hand detection and landmark extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp


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
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.6,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._mp_draw = mp.solutions.drawing_utils
        self._drawing_spec = self._mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame):
        """Return an annotated frame and the first detected hand, if any."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self._hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        hand_data: Optional[HandData] = None
        if results.multi_hand_landmarks and results.multi_handedness:
            hand_landmarks = results.multi_hand_landmarks[0]
            classification = results.multi_handedness[0].classification[0]
            handedness = classification.label
            confidence = classification.score
            frame_height, frame_width = frame.shape[:2]

            pixel_landmarks: List[Tuple[int, int]] = []
            normalized_landmarks: List[Tuple[float, float]] = []

            for landmark in hand_landmarks.landmark:
                pixel_x = int(landmark.x * frame_width)
                pixel_y = int(landmark.y * frame_height)
                pixel_landmarks.append((pixel_x, pixel_y))
                normalized_landmarks.append((landmark.x, landmark.y))

            self._mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                self._mp_hands.HAND_CONNECTIONS,
                self._drawing_spec,
                self._drawing_spec,
            )
            hand_data = HandData(
                pixel_landmarks=pixel_landmarks,
                normalized_landmarks=normalized_landmarks,
                handedness=handedness,
                confidence=confidence,
            )

        return frame, hand_data

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._hands.close()

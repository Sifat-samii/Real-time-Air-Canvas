"""Canvas state and stroke rendering helpers."""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Tuple

import cv2
import numpy as np

from .config import (
    BACKGROUND_COLOR,
    DRAW_DEADZONE_PIXELS,
    DRAW_THICKNESS,
    ERASER_THICKNESS,
    FAST_INTERPOLATION_STEPS,
    FAST_MOVEMENT_DISTANCE,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    INTERPOLATION_STEPS,
    SMOOTHING_ALPHA,
    SMOOTHING_WINDOW,
)
from .utils import average_point, clamp_point, exponential_smooth, interpolate_points, point_distance


Point = Tuple[int, int]


class CanvasManager:
    """Manage the persistent drawing canvas and stroke smoothing."""

    def __init__(self, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT) -> None:
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self._history: Deque[Point] = deque(maxlen=SMOOTHING_WINDOW)
        self._filtered_point: Optional[Point] = None
        self._previous_draw_point: Optional[Point] = None

    def clear(self) -> None:
        """Reset the canvas and stroke state."""
        self.canvas[:] = BACKGROUND_COLOR
        self._history.clear()
        self._filtered_point = None
        self._previous_draw_point = None

    def apply_gesture(self, gesture_state, pointer: Optional[Point], color: Tuple[int, int, int]) -> None:
        """Apply gesture-driven drawing or canvas actions."""
        if gesture_state.gesture == "clear":
            self.clear()
            return

        if gesture_state.gesture != "draw" or pointer is None:
            self._reset_stroke()
            return

        clamped_point = clamp_point(pointer, self.width, self.height)
        self._history.append(clamped_point)
        averaged_point = average_point(self._history)
        if averaged_point is None:
            return

        filtered_point = exponential_smooth(self._filtered_point, averaged_point, SMOOTHING_ALPHA)
        self._filtered_point = filtered_point
        thickness = ERASER_THICKNESS if gesture_state.eraser_enabled else DRAW_THICKNESS

        if self._previous_draw_point is None:
            self._previous_draw_point = filtered_point
            return

        if point_distance(self._previous_draw_point, filtered_point) < DRAW_DEADZONE_PIXELS:
            return

        movement_distance = point_distance(self._previous_draw_point, filtered_point)
        interpolation_steps = FAST_INTERPOLATION_STEPS if movement_distance > FAST_MOVEMENT_DISTANCE else INTERPOLATION_STEPS

        for interpolated_point in interpolate_points(self._previous_draw_point, filtered_point, interpolation_steps):
            cv2.line(self.canvas, self._previous_draw_point, interpolated_point, color, thickness, cv2.LINE_AA)
            self._previous_draw_point = interpolated_point

    def _reset_stroke(self) -> None:
        self._history.clear()
        self._filtered_point = None
        self._previous_draw_point = None

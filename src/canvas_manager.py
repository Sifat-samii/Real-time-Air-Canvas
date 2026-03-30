"""Canvas state, structured strokes, and rendering helpers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

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
    MIN_STROKE_POINTS,
    SMOOTHING_ALPHA,
    SMOOTHING_WINDOW,
)
from .utils import average_point, clamp_point, exponential_smooth, interpolate_points, make_transparent_canvas, point_distance


Point = Tuple[int, int]


@dataclass(slots=True)
class Stroke:
    """A structured stroke stored for undo/redo and export-friendly state."""

    points: List[Point] = field(default_factory=list)
    color: Tuple[int, int, int] = BACKGROUND_COLOR
    thickness: int = DRAW_THICKNESS
    is_eraser: bool = False


class CanvasManager:
    """Manage the persistent drawing canvas and stroke history."""

    def __init__(self, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT) -> None:
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.strokes: List[Stroke] = []
        self.redo_stack: List[Stroke] = []
        self.has_visible_content = False
        self._history: Deque[Point] = deque(maxlen=SMOOTHING_WINDOW)
        self._filtered_point: Optional[Point] = None
        self._previous_draw_point: Optional[Point] = None
        self._active_stroke: Optional[Stroke] = None

    def clear(self) -> None:
        """Reset the canvas and remove all history."""
        self.canvas[:] = BACKGROUND_COLOR
        self.strokes.clear()
        self.redo_stack.clear()
        self.has_visible_content = False
        self._active_stroke = None
        self._reset_stroke_tracking()

    def update_drawing(
        self,
        gesture_state,
        pointer: Optional[Point],
        color: Tuple[int, int, int],
        brush_thickness: int,
        eraser_thickness: int,
    ) -> None:
        """Apply gesture-driven drawing to the current stroke history."""
        if gesture_state.gesture == "clear":
            self.clear()
            return

        if gesture_state.gesture != "draw" or pointer is None:
            self.end_stroke()
            return

        thickness = eraser_thickness if gesture_state.eraser_enabled else brush_thickness
        stroke_color = BACKGROUND_COLOR if gesture_state.eraser_enabled else color
        self._ensure_active_stroke(color=stroke_color, thickness=thickness, is_eraser=gesture_state.eraser_enabled)
        self._append_point(pointer)

    def end_stroke(self) -> None:
        """Finalize the active stroke if it has meaningful content."""
        if self._active_stroke and len(self._active_stroke.points) < MIN_STROKE_POINTS:
            if self.strokes and self.strokes[-1] is self._active_stroke:
                self.strokes.pop()
        self._active_stroke = None
        self._reset_stroke_tracking()

    def undo(self) -> bool:
        """Undo the most recent stroke."""
        self.end_stroke()
        if not self.strokes:
            return False
        self.redo_stack.append(self.strokes.pop())
        self._rebuild_canvas()
        return True

    def redo(self) -> bool:
        """Redo the most recently undone stroke."""
        self.end_stroke()
        if not self.redo_stack:
            return False
        self.strokes.append(self.redo_stack.pop())
        self._rebuild_canvas()
        return True

    def save_transparent(self, output_path: str) -> bool:
        """Save the current canvas as a transparent PNG."""
        return bool(cv2.imwrite(output_path, make_transparent_canvas(self.canvas)))

    def _ensure_active_stroke(self, color: Tuple[int, int, int], thickness: int, is_eraser: bool) -> None:
        if self._active_stroke is None:
            self._active_stroke = Stroke(color=color, thickness=thickness, is_eraser=is_eraser)
            self.strokes.append(self._active_stroke)
            self.redo_stack.clear()
            return

        if (
            self._active_stroke.color != color
            or self._active_stroke.thickness != thickness
            or self._active_stroke.is_eraser != is_eraser
        ):
            self.end_stroke()
            self._active_stroke = Stroke(color=color, thickness=thickness, is_eraser=is_eraser)
            self.strokes.append(self._active_stroke)
            self.redo_stack.clear()

    def _append_point(self, pointer: Point) -> None:
        clamped_point = clamp_point(pointer, self.width, self.height)
        self._history.append(clamped_point)
        averaged_point = average_point(self._history)
        if averaged_point is None or self._active_stroke is None:
            return

        filtered_point = exponential_smooth(self._filtered_point, averaged_point, SMOOTHING_ALPHA)
        self._filtered_point = filtered_point

        if self._previous_draw_point is None:
            self._previous_draw_point = filtered_point
            self._active_stroke.points.append(filtered_point)
            return

        movement_distance = point_distance(self._previous_draw_point, filtered_point)
        if movement_distance < DRAW_DEADZONE_PIXELS:
            return

        interpolation_steps = FAST_INTERPOLATION_STEPS if movement_distance > FAST_MOVEMENT_DISTANCE else INTERPOLATION_STEPS
        for interpolated_point in interpolate_points(self._previous_draw_point, filtered_point, interpolation_steps):
            self._render_segment(self._previous_draw_point, interpolated_point, self._active_stroke)
            self._active_stroke.points.append(interpolated_point)
            self._previous_draw_point = interpolated_point

    def _render_segment(self, start: Point, end: Point, stroke: Stroke) -> None:
        self.has_visible_content = True
        line_type = cv2.LINE_8 if stroke.is_eraser else cv2.LINE_AA
        cv2.line(self.canvas, start, end, stroke.color, stroke.thickness, line_type)
        if stroke.is_eraser:
            cv2.circle(self.canvas, start, stroke.thickness // 2, BACKGROUND_COLOR, -1, cv2.LINE_8)
            cv2.circle(self.canvas, end, stroke.thickness // 2, BACKGROUND_COLOR, -1, cv2.LINE_8)

    def _rebuild_canvas(self) -> None:
        self.canvas[:] = BACKGROUND_COLOR
        self.has_visible_content = False
        for stroke in self.strokes:
            if len(stroke.points) < 2:
                continue
            previous_point = stroke.points[0]
            for point in stroke.points[1:]:
                self._render_segment(previous_point, point, stroke)
                previous_point = point
        self._active_stroke = None
        self._reset_stroke_tracking()

    def _reset_stroke_tracking(self) -> None:
        self._history.clear()
        self._filtered_point = None
        self._previous_draw_point = None

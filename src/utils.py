"""Helper utilities shared across the air canvas application."""

from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Deque, Iterable, Optional, Tuple

import cv2
import numpy as np


Point = Tuple[int, int]
Rect = Tuple[int, int, int, int]


def clamp_point(point: Point, width: int, height: int) -> Point:
    """Keep a point within frame boundaries."""
    x = max(0, min(point[0], width - 1))
    y = max(0, min(point[1], height - 1))
    return x, y


def average_point(history: Deque[Point]) -> Optional[Point]:
    """Return the mean point for a short history of tracked positions."""
    if not history:
        return None

    xs = [point[0] for point in history]
    ys = [point[1] for point in history]
    return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))


def exponential_smooth(previous: Optional[Point], current: Point, alpha: float) -> Point:
    """Apply a simple exponential moving average to a tracked point."""
    if previous is None:
        return current

    x = int((1 - alpha) * previous[0] + alpha * current[0])
    y = int((1 - alpha) * previous[1] + alpha * current[1])
    return x, y


def interpolate_points(start: Point, end: Point, steps: int) -> Iterable[Point]:
    """Generate intermediate points between two points for smoother strokes."""
    if steps <= 1:
        return [end]

    points = []
    for step in range(1, steps + 1):
        alpha = step / steps
        x = int((1 - alpha) * start[0] + alpha * end[0])
        y = int((1 - alpha) * start[1] + alpha * end[1])
        points.append((x, y))
    return points


def blend_canvas(frame: np.ndarray, canvas: np.ndarray) -> np.ndarray:
    """Blend the black-background drawing canvas over the webcam frame."""
    grayscale = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(grayscale, 20, 255, cv2.THRESH_BINARY)
    inverse_mask = cv2.bitwise_not(mask)

    frame_background = cv2.bitwise_and(frame, frame, mask=inverse_mask)
    canvas_foreground = cv2.bitwise_and(canvas, canvas, mask=mask)
    return cv2.addWeighted(frame_background, 1.0, canvas_foreground, 1.0, 0.0)


def point_in_rect(point: Point, rect: Rect) -> bool:
    """Check whether a point lies inside a rectangle."""
    x, y = point
    rect_x, rect_y, width, height = rect
    return rect_x <= x <= rect_x + width and rect_y <= y <= rect_y + height


def point_distance(point_a: Point, point_b: Point) -> float:
    """Return Euclidean distance between two points."""
    return float(np.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1]))


def shrink_rect(rect: Rect, inset: int) -> Rect:
    """Inset a rectangle while keeping a valid positive area."""
    x, y, width, height = rect
    safe_inset = min(inset, max(0, width // 2 - 1), max(0, height // 2 - 1))
    return x + safe_inset, y + safe_inset, width - 2 * safe_inset, height - 2 * safe_inset


def build_save_path(prefix: str, extension: str = "png") -> str:
    """Create a timestamped filename for saved images."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"


def make_transparent_canvas(canvas: np.ndarray) -> np.ndarray:
    """Convert the black-background canvas to a BGRA image with transparency."""
    grayscale = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(grayscale, 20, 255, cv2.THRESH_BINARY)
    return np.dstack((canvas.copy(), alpha))

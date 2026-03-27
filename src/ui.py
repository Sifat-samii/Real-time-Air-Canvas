"""Rendering helpers for the air canvas interface."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2

from .config import (
    COLOR_MAP,
    CURSOR_ERASER_RADIUS,
    CURSOR_RADIUS,
    CURSOR_THICKNESS,
    FRAME_WIDTH,
    MODE_BADGE_HEIGHT,
    MODE_BADGE_WIDTH,
    MODE_BADGE_X,
    MODE_BADGE_Y,
    PALETTE_BOX_HEIGHT,
    PALETTE_BOX_WIDTH,
    PALETTE_HEIGHT,
    PALETTE_MARGIN,
    PALETTE_ORDER,
    PALETTE_TEXT_SCALE,
    STATUS_PANEL_ALPHA,
    STATUS_PANEL_HEIGHT,
    STATUS_PANEL_WIDTH,
    TEXT_COLOR,
)
from .utils import blend_canvas


class UIManager:
    """Draw the palette, status labels, and final composited frame."""

    def __init__(self) -> None:
        self.palette_boxes = self._build_palette_boxes()

    def resolve_color(self, color_name: str) -> Tuple[int, int, int]:
        return COLOR_MAP[color_name]

    def compose(self, frame, canvas, gesture_state):
        """Blend the frame and canvas, then add UI overlays."""
        composed = blend_canvas(frame, canvas)
        self._draw_palette(composed, gesture_state.brush_color_name, gesture_state.hovered_palette_name)
        self._draw_mode_badge(composed, gesture_state)
        self._draw_status(composed, gesture_state)
        self._draw_cursor(composed, gesture_state.pointer, gesture_state.eraser_enabled)
        return composed

    def _build_palette_boxes(self) -> Dict[str, Tuple[int, int, int, int]]:
        boxes: Dict[str, Tuple[int, int, int, int]] = {}
        x = PALETTE_MARGIN
        y = PALETTE_MARGIN

        for color_name in PALETTE_ORDER:
            boxes[color_name] = (x, y, PALETTE_BOX_WIDTH, PALETTE_BOX_HEIGHT)
            x += PALETTE_BOX_WIDTH + PALETTE_MARGIN

        return boxes

    def _draw_palette(
        self,
        frame,
        selected_color_name: str,
        hovered_palette_name: Optional[str],
    ) -> None:
        cv2.rectangle(frame, (0, 0), (FRAME_WIDTH, PALETTE_HEIGHT), (40, 40, 40), -1)

        for color_name, (x, y, width, height) in self.palette_boxes.items():
            color = COLOR_MAP[color_name]
            is_selected = color_name == selected_color_name
            is_hovered = color_name == hovered_palette_name
            border_color = COLOR_MAP["white"] if is_selected else (180, 180, 180)
            border_thickness = 3 if is_selected else 2

            cv2.rectangle(frame, (x, y), (x + width, y + height), color, -1)
            cv2.rectangle(frame, (x, y), (x + width, y + height), border_color, border_thickness)

            if is_hovered:
                cv2.rectangle(frame, (x - 4, y - 4), (x + width + 4, y + height + 4), (255, 255, 255), 1)

            cv2.putText(
                frame,
                color_name.upper(),
                (x + 10, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                PALETTE_TEXT_SCALE,
                COLOR_MAP["white"] if color_name != "yellow" else (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

    def _draw_status(self, frame, gesture_state) -> None:
        panel = frame.copy()
        top_left = (15, PALETTE_HEIGHT + 15)
        bottom_right = (15 + STATUS_PANEL_WIDTH, PALETTE_HEIGHT + 15 + STATUS_PANEL_HEIGHT)
        cv2.rectangle(panel, top_left, bottom_right, (20, 20, 20), -1)
        cv2.addWeighted(panel, STATUS_PANEL_ALPHA, frame, 1 - STATUS_PANEL_ALPHA, 0, frame)

        labels = [
            f"Tool: {'Eraser' if gesture_state.eraser_enabled else 'Pen'}",
            f"Brush: {gesture_state.brush_color_name}",
            f"Mode: {gesture_state.mode}",
            gesture_state.status_text,
            "Keys: C clear | S save | M mirror | E exit",
        ]

        start_y = PALETTE_HEIGHT + 43
        for index, label in enumerate(labels):
            cv2.putText(
                frame,
                label,
                (30, start_y + index * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62 if index < 4 else 0.58,
                TEXT_COLOR,
                2,
                cv2.LINE_AA,
            )

    def _draw_mode_badge(self, frame, gesture_state) -> None:
        """Draw a compact badge for the current interaction state."""
        if gesture_state.eraser_enabled:
            badge_color = (30, 30, 30)
            label = "ERASER"
        elif gesture_state.gesture == "draw":
            badge_color = (35, 110, 35)
            label = "DRAW"
        elif gesture_state.gesture in {"palette", "palette_hover"}:
            badge_color = (140, 90, 20)
            label = "PALETTE"
        elif gesture_state.gesture == "clear":
            badge_color = (40, 40, 160)
            label = "CLEARED"
        else:
            badge_color = (80, 80, 80)
            label = "IDLE"

        cv2.rectangle(
            frame,
            (MODE_BADGE_X, MODE_BADGE_Y),
            (MODE_BADGE_X + MODE_BADGE_WIDTH, MODE_BADGE_Y + MODE_BADGE_HEIGHT),
            badge_color,
            -1,
        )
        cv2.rectangle(
            frame,
            (MODE_BADGE_X, MODE_BADGE_Y),
            (MODE_BADGE_X + MODE_BADGE_WIDTH, MODE_BADGE_Y + MODE_BADGE_HEIGHT),
            COLOR_MAP["white"],
            2,
        )
        cv2.putText(
            frame,
            label,
            (MODE_BADGE_X + 18, MODE_BADGE_Y + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            COLOR_MAP["white"],
            2,
            cv2.LINE_AA,
        )

    def _draw_cursor(self, frame, pointer, eraser_enabled: bool) -> None:
        if pointer is None:
            return

        radius = CURSOR_ERASER_RADIUS if eraser_enabled else CURSOR_RADIUS
        cv2.circle(frame, pointer, radius, COLOR_MAP["white"], CURSOR_THICKNESS)

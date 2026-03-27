"""Rendering helpers for the air canvas interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2

from .config import (
    COLOR_MAP,
    CURSOR_ACTIVE_COLOR,
    CURSOR_COLOR,
    CURSOR_ERASER_RADIUS,
    CURSOR_FILL_RADIUS,
    CURSOR_RADIUS,
    CURSOR_THICKNESS,
    FRAME_WIDTH,
    HELP_PANEL_HEIGHT,
    HELP_PANEL_WIDTH,
    HELP_PANEL_X,
    HELP_PANEL_Y,
    MODE_BADGE_HEIGHT,
    MODE_BADGE_WIDTH,
    MODE_BADGE_X,
    MODE_BADGE_Y,
    PALETTE_BOX_HEIGHT,
    PALETTE_BOX_WIDTH,
    PALETTE_HEIGHT,
    PALETTE_LABEL_Y_OFFSET,
    PALETTE_MARGIN,
    PALETTE_ORDER,
    PALETTE_TEXT_SCALE,
    STATUS_PANEL_ALPHA,
    STATUS_PANEL_HEIGHT,
    STATUS_PANEL_WIDTH,
    STATUS_PANEL_X,
    STATUS_PANEL_Y,
    TEXT_ACCENT_COLOR,
    TEXT_COLOR,
    TEXT_MUTED_COLOR,
    TOAST_HEIGHT,
    TOAST_PADDING_X,
    TOAST_Y_OFFSET,
)
from .utils import Rect, blend_canvas


@dataclass
class OverlayState:
    """UI-focused state passed into the renderer each frame."""

    brush_thickness: int
    eraser_thickness: int
    mirror_enabled: bool
    landmarks_visible: bool
    help_visible: bool
    fps_visible: bool
    fps_value: float
    can_undo: bool
    can_redo: bool
    toast_message: str = ""


class UIManager:
    """Draw the palette, HUD, help overlay, and composited frame."""

    def __init__(self) -> None:
        self.palette_boxes = self._build_palette_boxes()

    def resolve_color(self, color_name: str) -> Tuple[int, int, int]:
        return COLOR_MAP[color_name]

    def compose(self, frame, canvas, gesture_state, overlay_state: OverlayState):
        """Blend the frame and canvas, then add UI overlays."""
        composed = blend_canvas(frame, canvas)
        self._draw_palette(composed, gesture_state.brush_color_name, gesture_state.hovered_palette_name)
        self._draw_mode_badge(composed, gesture_state)
        self._draw_status_panel(composed, gesture_state, overlay_state)
        if overlay_state.help_visible:
            self._draw_help_panel(composed)
        if overlay_state.toast_message:
            self._draw_toast(composed, overlay_state.toast_message)
        self._draw_cursor(composed, gesture_state.pointer, gesture_state.eraser_enabled, gesture_state.gesture == "draw")
        return composed

    def _build_palette_boxes(self) -> Dict[str, Rect]:
        boxes: Dict[str, Rect] = {}
        x = PALETTE_MARGIN
        y = PALETTE_MARGIN
        for color_name in PALETTE_ORDER:
            boxes[color_name] = (x, y, PALETTE_BOX_WIDTH, PALETTE_BOX_HEIGHT)
            x += PALETTE_BOX_WIDTH + PALETTE_MARGIN
        return boxes

    def _draw_palette(self, frame, selected_color_name: str, hovered_palette_name: Optional[str]) -> None:
        cv2.rectangle(frame, (0, 0), (FRAME_WIDTH, PALETTE_HEIGHT), (24, 24, 24), -1)
        for color_name, (x, y, width, height) in self.palette_boxes.items():
            color = COLOR_MAP[color_name]
            is_selected = color_name == selected_color_name
            is_hovered = color_name == hovered_palette_name
            border_color = COLOR_MAP["white"] if is_selected else (115, 115, 115)
            border_thickness = 3 if is_selected else 1

            cv2.rectangle(frame, (x, y), (x + width, y + height), color, -1)
            cv2.rectangle(frame, (x, y), (x + width, y + height), border_color, border_thickness)
            if is_hovered:
                cv2.rectangle(frame, (x - 4, y - 4), (x + width + 4, y + height + 4), TEXT_ACCENT_COLOR, 1)

            cv2.putText(
                frame,
                color_name.upper(),
                (x + 12, y + PALETTE_LABEL_Y_OFFSET),
                cv2.FONT_HERSHEY_SIMPLEX,
                PALETTE_TEXT_SCALE,
                COLOR_MAP["white"] if color_name != "yellow" else (20, 20, 20),
                2,
                cv2.LINE_AA,
            )

    def _draw_status_panel(self, frame, gesture_state, overlay_state: OverlayState) -> None:
        panel = frame.copy()
        top_left = (STATUS_PANEL_X, STATUS_PANEL_Y)
        bottom_right = (STATUS_PANEL_X + STATUS_PANEL_WIDTH, STATUS_PANEL_Y + STATUS_PANEL_HEIGHT)
        cv2.rectangle(panel, top_left, bottom_right, (18, 18, 18), -1)
        cv2.addWeighted(panel, STATUS_PANEL_ALPHA, frame, 1 - STATUS_PANEL_ALPHA, 0, frame)

        lines = [
            f"Tool: {'Eraser' if gesture_state.eraser_enabled else 'Pen'}",
            f"Brush: {gesture_state.brush_color_name}  Size: {overlay_state.brush_thickness}px",
            f"Eraser Size: {overlay_state.eraser_thickness}px",
            f"Mode: {gesture_state.mode}  Tracking: {gesture_state.hand_confidence:.2f}",
            gesture_state.status_text,
            self._build_toggles_line(overlay_state),
        ]
        start_y = STATUS_PANEL_Y + 28
        for index, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (STATUS_PANEL_X + 14, start_y + index * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58 if index < 5 else 0.53,
                TEXT_COLOR if index != 5 else TEXT_MUTED_COLOR,
                2,
                cv2.LINE_AA,
            )

    def _build_toggles_line(self, overlay_state: OverlayState) -> str:
        fps_text = f"{overlay_state.fps_value:.1f} FPS" if overlay_state.fps_visible else "FPS hidden"
        undo_state = "Undo ready" if overlay_state.can_undo else "Undo empty"
        redo_state = "Redo ready" if overlay_state.can_redo else "Redo empty"
        return (
            f"{fps_text} | Mirror {'on' if overlay_state.mirror_enabled else 'off'} | "
            f"Landmarks {'on' if overlay_state.landmarks_visible else 'off'} | {undo_state} | {redo_state}"
        )

    def _draw_mode_badge(self, frame, gesture_state) -> None:
        """Draw a compact badge for the current interaction state."""
        if gesture_state.eraser_enabled:
            badge_color = (28, 28, 28)
            label = "ERASER"
        elif gesture_state.gesture == "draw":
            badge_color = (36, 118, 50)
            label = "DRAW"
        elif gesture_state.gesture in {"palette", "palette_hover"}:
            badge_color = (153, 94, 28)
            label = "PALETTE"
        elif gesture_state.gesture == "clear":
            badge_color = (50, 62, 170)
            label = "CLEARED"
        else:
            badge_color = (78, 78, 78)
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

    def _draw_help_panel(self, frame) -> None:
        panel = frame.copy()
        top_left = (HELP_PANEL_X, HELP_PANEL_Y)
        bottom_right = (HELP_PANEL_X + HELP_PANEL_WIDTH, HELP_PANEL_Y + HELP_PANEL_HEIGHT)
        cv2.rectangle(panel, top_left, bottom_right, (14, 14, 14), -1)
        cv2.addWeighted(panel, 0.58, frame, 0.42, 0, frame)

        lines = [
            "Gestures: index only draw | open palm eraser | fist clear",
            "Palette: hover index fingertip over a color box",
            "Keys: [ ] brush size | Z undo | Y redo | N new canvas",
            "Keys: S save | T transparent save | P composite screenshot",
            "Keys: M mirror | L landmarks | H help | F FPS | E exit",
        ]
        for index, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (HELP_PANEL_X + 14, HELP_PANEL_Y + 32 + index * 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                TEXT_MUTED_COLOR if index < 2 else TEXT_COLOR,
                2,
                cv2.LINE_AA,
            )

    def _draw_toast(self, frame, message: str) -> None:
        width = min(FRAME_WIDTH - 40, 16 * len(message) + TOAST_PADDING_X * 2)
        x = (FRAME_WIDTH - width) // 2
        y = frame.shape[0] - TOAST_Y_OFFSET - TOAST_HEIGHT
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + TOAST_HEIGHT), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (x, y), (x + width, y + TOAST_HEIGHT), TEXT_ACCENT_COLOR, 1)
        cv2.putText(
            frame,
            message,
            (x + TOAST_PADDING_X, y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )

    def _draw_cursor(self, frame, pointer, eraser_enabled: bool, active: bool) -> None:
        if pointer is None:
            return

        radius = CURSOR_ERASER_RADIUS if eraser_enabled else CURSOR_RADIUS
        outline = CURSOR_ACTIVE_COLOR if active else CURSOR_COLOR
        cv2.circle(frame, pointer, radius, outline, CURSOR_THICKNESS)
        cv2.circle(frame, pointer, CURSOR_FILL_RADIUS, outline, -1)

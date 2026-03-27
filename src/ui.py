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
    HELP_PANEL_HEIGHT,
    HELP_PANEL_WIDTH,
    HELP_PANEL_X,
    MODE_BADGE_HEIGHT,
    MODE_BADGE_WIDTH,
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
    fullscreen_enabled: bool
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
        self.palette_boxes: Dict[str, Rect] = {}
        self._frame_width = 0
        self._frame_height = 0

    def resolve_color(self, color_name: str) -> Tuple[int, int, int]:
        return COLOR_MAP[color_name]

    def compose(self, frame, canvas, gesture_state, overlay_state: OverlayState):
        """Blend the frame and canvas, then add UI overlays."""
        self.update_layout(frame.shape[1], frame.shape[0])
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

    def update_layout(self, frame_width: int, frame_height: int) -> None:
        """Update cached UI layout for the current frame size."""
        if frame_width == self._frame_width and frame_height == self._frame_height and self.palette_boxes:
            return

        self._frame_width = frame_width
        self._frame_height = frame_height
        self.palette_boxes = self._build_palette_boxes(frame_width)

    def _build_palette_boxes(self, frame_width: int) -> Dict[str, Rect]:
        boxes: Dict[str, Rect] = {}
        total_width = len(PALETTE_ORDER) * PALETTE_BOX_WIDTH + (len(PALETTE_ORDER) - 1) * PALETTE_MARGIN
        x = max(PALETTE_MARGIN, (frame_width - total_width) // 2)
        y = PALETTE_MARGIN
        for color_name in PALETTE_ORDER:
            boxes[color_name] = (x, y, PALETTE_BOX_WIDTH, PALETTE_BOX_HEIGHT)
            x += PALETTE_BOX_WIDTH + PALETTE_MARGIN
        return boxes

    def _draw_palette(self, frame, selected_color_name: str, hovered_palette_name: Optional[str]) -> None:
        cv2.rectangle(frame, (0, 0), (self._frame_width, PALETTE_HEIGHT), (24, 24, 24), -1)
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
            f"{fps_text} | Fullscreen {'on' if overlay_state.fullscreen_enabled else 'off'} | "
            f"Mirror {'on' if overlay_state.mirror_enabled else 'off'} | "
            f"Landmarks {'on' if overlay_state.landmarks_visible else 'off'} | {undo_state} | {redo_state}"
        )

    def _draw_mode_badge(self, frame, gesture_state) -> None:
        """Draw a compact badge for the current interaction state."""
        badge_x = max(24, self._frame_width - MODE_BADGE_WIDTH - 24)
        badge_y = STATUS_PANEL_Y - 6
        if gesture_state.eraser_enabled:
            badge_color = (28, 28, 28)
            label = "ERASER"
        elif gesture_state.gesture == "draw":
            badge_color = (36, 118, 50)
            label = "BRUSH"
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
            (badge_x, badge_y),
            (badge_x + MODE_BADGE_WIDTH, badge_y + MODE_BADGE_HEIGHT),
            badge_color,
            -1,
        )
        cv2.rectangle(
            frame,
            (badge_x, badge_y),
            (badge_x + MODE_BADGE_WIDTH, badge_y + MODE_BADGE_HEIGHT),
            COLOR_MAP["white"],
            2,
        )
        cv2.putText(
            frame,
            label,
            (badge_x + 18, badge_y + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            COLOR_MAP["white"],
            2,
            cv2.LINE_AA,
        )

    def _draw_help_panel(self, frame) -> None:
        help_y = min(self._frame_height - HELP_PANEL_HEIGHT - 20, STATUS_PANEL_Y + STATUS_PANEL_HEIGHT + 18)
        panel = frame.copy()
        top_left = (HELP_PANEL_X, help_y)
        bottom_right = (HELP_PANEL_X + HELP_PANEL_WIDTH, help_y + HELP_PANEL_HEIGHT)
        cv2.rectangle(panel, top_left, bottom_right, (14, 14, 14), -1)
        cv2.addWeighted(panel, 0.58, frame, 0.42, 0, frame)

        lines = [
            "Gestures: index only draw | open palm eraser | fist clear",
            "Palette: hover index fingertip over a color box",
            "Keys: [ ] brush size | Z undo | Y redo | N new canvas",
            "Keys: S save | T transparent save | P composite screenshot",
            "Keys: F fullscreen | G FPS | M mirror | L landmarks | H help | E exit",
        ]
        for index, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (HELP_PANEL_X + 14, help_y + 32 + index * 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                TEXT_MUTED_COLOR if index < 2 else TEXT_COLOR,
                2,
                cv2.LINE_AA,
            )

    def _draw_toast(self, frame, message: str) -> None:
        width = min(self._frame_width - 40, 16 * len(message) + TOAST_PADDING_X * 2)
        x = (self._frame_width - width) // 2
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
        if eraser_enabled:
            cv2.line(frame, (pointer[0] - 8, pointer[1]), (pointer[0] + 8, pointer[1]), outline, 2, cv2.LINE_AA)
            cv2.line(frame, (pointer[0], pointer[1] - 8), (pointer[0], pointer[1] + 8), outline, 2, cv2.LINE_AA)

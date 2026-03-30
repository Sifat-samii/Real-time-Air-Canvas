"""Gesture detection, finger-state analysis, and debounce logic."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .config import (
    CLEAR_COOLDOWN,
    CLEAR_HOLD_TIME,
    ERASER_COLOR_NAME,
    ERASER_HOLD_TIME,
    ERASER_TOGGLE_COOLDOWN,
    FINGER_MCP_IDS,
    FINGER_PIP_IDS,
    FINGERTIP_IDS,
    INDEX_FINGER_NAME,
    PALETTE_COOLDOWN,
    PALETTE_HIT_INSET,
    PALETTE_HOVER_TIME,
)
from .utils import point_distance, point_in_rect, shrink_rect


@dataclass(slots=True)
class GestureState:
    """Current interaction state derived from the tracked hand."""

    gesture: str
    mode: str
    finger_states: Dict[str, bool]
    pointer: Optional[Tuple[int, int]]
    brush_color_name: str
    active_color_name: str
    eraser_enabled: bool
    status_text: str
    hovered_palette_name: Optional[str]
    hand_confidence: float


class GestureInterpreter:
    """Track gesture state and enforce hold, hover, and cooldown rules."""

    def __init__(self) -> None:
        self.brush_color_name = "red"
        self.eraser_enabled = False
        self._last_clear_time = 0.0
        self._last_eraser_toggle_time = 0.0
        self._last_palette_time = 0.0
        self._clear_hold_start: Optional[float] = None
        self._eraser_hold_start: Optional[float] = None
        self._clear_latched = False
        self._eraser_latched = False
        self._hovered_palette_name: Optional[str] = None
        self._palette_hover_start: Optional[float] = None

    def idle_state(self) -> GestureState:
        """Return the idle gesture state when no hand is visible."""
        self._clear_hold_start = None
        self._eraser_hold_start = None
        self._hovered_palette_name = None
        self._palette_hover_start = None
        self._clear_latched = False
        self._eraser_latched = False
        return self._build_state(
            gesture="idle",
            mode="eraser" if self.eraser_enabled else "idle",
            finger_states={},
            pointer=None,
            status_text="Show one hand to begin",
            hovered_palette_name=None,
            hand_confidence=0.0,
        )

    def update(self, hand_data, palette_boxes) -> GestureState:
        """Interpret finger states and map them into app actions."""
        finger_states = self._detect_finger_states(hand_data)
        pointer = hand_data.pixel_landmarks[FINGERTIP_IDS[INDEX_FINGER_NAME]]
        hovered_color = self._palette_hit(pointer, palette_boxes)
        now = time.monotonic()

        clear_state = self._handle_clear(finger_states, pointer, now, hand_data.confidence)
        if clear_state is not None:
            return clear_state

        eraser_state = self._handle_eraser_toggle(finger_states, pointer, now, hand_data.confidence)
        if eraser_state is not None:
            return eraser_state

        palette_state = self._handle_palette_hover(finger_states, pointer, hovered_color, now, hand_data.confidence)
        if palette_state is not None:
            return palette_state

        if self._is_draw_gesture(finger_states) and hovered_color is None:
            return self._build_state(
                gesture="draw",
                mode="eraser" if self.eraser_enabled else "draw",
                finger_states=finger_states,
                pointer=pointer,
                status_text="Erasing" if self.eraser_enabled else "Drawing",
                hovered_palette_name=None,
                hand_confidence=hand_data.confidence,
            )

        return self._build_state(
            gesture="idle",
            mode="eraser" if self.eraser_enabled else "idle",
            finger_states=finger_states,
            pointer=pointer,
            status_text="Move to draw or hover over palette",
            hovered_palette_name=hovered_color,
            hand_confidence=hand_data.confidence,
        )

    def current_color_name(self) -> str:
        """Return the active color name, respecting eraser mode."""
        return ERASER_COLOR_NAME if self.eraser_enabled else self.brush_color_name

    def _detect_finger_states(self, hand_data) -> Dict[str, bool]:
        """Estimate whether each finger is extended using hand-size-aware thresholds."""
        landmarks = hand_data.pixel_landmarks
        hand_size = self._estimate_hand_size(landmarks)
        vertical_margin = max(12.0, hand_size * 0.18)
        horizontal_margin = max(10.0, hand_size * 0.15)

        return {
            "thumb": self._thumb_is_open(landmarks, horizontal_margin),
            "index": self._finger_is_extended(landmarks, "index", vertical_margin),
            "middle": self._finger_is_extended(landmarks, "middle", vertical_margin),
            "ring": self._finger_is_extended(landmarks, "ring", vertical_margin),
            "pinky": self._finger_is_extended(landmarks, "pinky", vertical_margin),
        }

    def _estimate_hand_size(self, landmarks) -> float:
        wrist = landmarks[0]
        middle_mcp = landmarks[FINGER_MCP_IDS["middle"]]
        index_mcp = landmarks[FINGER_MCP_IDS["index"]]
        pinky_mcp = landmarks[FINGER_MCP_IDS["pinky"]]
        palm_height = point_distance(wrist, middle_mcp)
        palm_width = point_distance(index_mcp, pinky_mcp)
        return max(palm_height, palm_width)

    def _finger_is_extended(self, landmarks, finger_name: str, margin: float) -> bool:
        tip = landmarks[FINGERTIP_IDS[finger_name]]
        pip = landmarks[FINGER_PIP_IDS[finger_name]]
        mcp = landmarks[FINGER_MCP_IDS[finger_name]]
        return tip[1] < pip[1] < mcp[1] and (mcp[1] - tip[1]) > margin

    def _thumb_is_open(self, landmarks, margin: float) -> bool:
        thumb_tip = landmarks[FINGERTIP_IDS["thumb"]]
        thumb_ip = landmarks[FINGER_PIP_IDS["thumb"]]
        thumb_mcp = landmarks[FINGER_MCP_IDS["thumb"]]
        horizontal_span = abs(thumb_tip[0] - thumb_mcp[0])
        vertical_compactness = abs(thumb_tip[1] - thumb_ip[1])
        return horizontal_span > margin and horizontal_span > vertical_compactness

    def _is_draw_gesture(self, finger_states: Dict[str, bool]) -> bool:
        return finger_states.get("index", False) and not any(
            finger_states[finger] for finger in ("middle", "ring", "pinky", "thumb")
        )

    def _all_fingers_up(self, finger_states: Dict[str, bool]) -> bool:
        return all(finger_states.values())

    def _is_fist(self, finger_states: Dict[str, bool]) -> bool:
        return not any(finger_states.values())

    def _palette_hit(self, pointer, palette_boxes) -> Optional[str]:
        for color_name, rect in palette_boxes.items():
            if point_in_rect(pointer, shrink_rect(rect, PALETTE_HIT_INSET)):
                return color_name
        return None

    def _handle_clear(self, finger_states, pointer, now: float, hand_confidence: float) -> Optional[GestureState]:
        if self._is_fist(finger_states):
            self._hovered_palette_name = None
            self._palette_hover_start = None

            if self._clear_latched:
                return self._build_state(
                    gesture="idle",
                    mode="eraser" if self.eraser_enabled else "idle",
                    finger_states=finger_states,
                    pointer=pointer,
                    status_text="Release fist to re-arm clear",
                    hovered_palette_name=None,
                    hand_confidence=hand_confidence,
                )

            if self._clear_hold_start is None:
                self._clear_hold_start = now

            if now - self._last_clear_time < CLEAR_COOLDOWN:
                return self._build_state(
                    gesture="idle",
                    mode="eraser" if self.eraser_enabled else "idle",
                    finger_states=finger_states,
                    pointer=pointer,
                    status_text="Clear cooling down",
                    hovered_palette_name=None,
                    hand_confidence=hand_confidence,
                )

            remaining = max(0.0, CLEAR_HOLD_TIME - (now - self._clear_hold_start))
            if remaining > 0:
                return self._build_state(
                    gesture="idle",
                    mode="eraser" if self.eraser_enabled else "idle",
                    finger_states=finger_states,
                    pointer=pointer,
                    status_text=f"Hold fist {remaining:.1f}s to clear",
                    hovered_palette_name=None,
                    hand_confidence=hand_confidence,
                )

            self._clear_latched = True
            self._last_clear_time = now
            return self._build_state(
                gesture="clear",
                mode="clear",
                finger_states=finger_states,
                pointer=pointer,
                status_text="Canvas cleared",
                hovered_palette_name=None,
                hand_confidence=hand_confidence,
            )

        self._clear_hold_start = None
        self._clear_latched = False
        return None

    def _handle_eraser_toggle(self, finger_states, pointer, now: float, hand_confidence: float) -> Optional[GestureState]:
        if self._all_fingers_up(finger_states):
            self._hovered_palette_name = None
            self._palette_hover_start = None

            if self._eraser_latched:
                return self._build_state(
                    gesture="idle",
                    mode="eraser" if self.eraser_enabled else "idle",
                    finger_states=finger_states,
                    pointer=pointer,
                    status_text="Release open palm to re-arm toggle",
                    hovered_palette_name=None,
                    hand_confidence=hand_confidence,
                )

            if self._eraser_hold_start is None:
                self._eraser_hold_start = now

            if now - self._last_eraser_toggle_time < ERASER_TOGGLE_COOLDOWN:
                return self._build_state(
                    gesture="idle",
                    mode="eraser" if self.eraser_enabled else "idle",
                    finger_states=finger_states,
                    pointer=pointer,
                    status_text="Tool toggle cooling down",
                    hovered_palette_name=None,
                    hand_confidence=hand_confidence,
                )

            remaining = max(0.0, ERASER_HOLD_TIME - (now - self._eraser_hold_start))
            if remaining > 0:
                return self._build_state(
                    gesture="idle",
                    mode="eraser" if self.eraser_enabled else "idle",
                    finger_states=finger_states,
                    pointer=pointer,
                    status_text=f"Hold open palm {remaining:.1f}s to toggle eraser",
                    hovered_palette_name=None,
                    hand_confidence=hand_confidence,
                )

            self.eraser_enabled = not self.eraser_enabled
            self._eraser_latched = True
            self._last_eraser_toggle_time = now
            return self._build_state(
                gesture="toggle_eraser",
                mode="eraser" if self.eraser_enabled else "draw",
                finger_states=finger_states,
                pointer=pointer,
                status_text="Eraser enabled" if self.eraser_enabled else "Pen restored",
                hovered_palette_name=None,
                hand_confidence=hand_confidence,
            )

        self._eraser_hold_start = None
        self._eraser_latched = False
        return None

    def _handle_palette_hover(
        self,
        finger_states,
        pointer,
        hovered_color,
        now: float,
        hand_confidence: float,
    ) -> Optional[GestureState]:
        if hovered_color is None or not self._is_draw_gesture(finger_states):
            self._hovered_palette_name = None
            self._palette_hover_start = None
            return None

        if hovered_color != self._hovered_palette_name:
            self._hovered_palette_name = hovered_color
            self._palette_hover_start = now

        if self._palette_hover_start is None:
            self._palette_hover_start = now

        if now - self._last_palette_time < PALETTE_COOLDOWN:
            return self._build_state(
                gesture="palette_hover",
                mode="palette",
                finger_states=finger_states,
                pointer=pointer,
                status_text="Palette cooling down",
                hovered_palette_name=hovered_color,
                hand_confidence=hand_confidence,
            )

        hover_elapsed = now - self._palette_hover_start
        if hover_elapsed < PALETTE_HOVER_TIME:
            return self._build_state(
                gesture="palette_hover",
                mode="palette",
                finger_states=finger_states,
                pointer=pointer,
                status_text=f"Hold on {hovered_color} {PALETTE_HOVER_TIME - hover_elapsed:.1f}s",
                hovered_palette_name=hovered_color,
                hand_confidence=hand_confidence,
            )

        self.brush_color_name = hovered_color
        self.eraser_enabled = False
        self._last_palette_time = now
        self._palette_hover_start = now
        return self._build_state(
            gesture="palette",
            mode="palette",
            finger_states=finger_states,
            pointer=pointer,
            status_text=f"Brush set to {hovered_color}",
            hovered_palette_name=hovered_color,
            hand_confidence=hand_confidence,
        )

    def _build_state(
        self,
        gesture: str,
        mode: str,
        finger_states: Dict[str, bool],
        pointer: Optional[Tuple[int, int]],
        status_text: str,
        hovered_palette_name: Optional[str],
        hand_confidence: float,
    ) -> GestureState:
        return GestureState(
            gesture=gesture,
            mode=mode,
            finger_states=finger_states,
            pointer=pointer,
            brush_color_name=self.brush_color_name,
            active_color_name=self.current_color_name(),
            eraser_enabled=self.eraser_enabled,
            status_text=status_text,
            hovered_palette_name=hovered_palette_name,
            hand_confidence=hand_confidence,
        )

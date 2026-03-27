"""Application entry point for the Real-Time AI Air Canvas."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time

import cv2

from src.canvas_manager import CanvasManager
from src.config import (
    BRUSH_STEP,
    CAMERA_BACKEND,
    CAMERA_FALLBACK_INDEXES,
    CAMERA_INDEX,
    COMPOSITE_PREFIX,
    DEFAULT_MIRROR_FRAME,
    DRAW_LANDMARKS_BY_DEFAULT,
    DRAW_THICKNESS,
    ERASER_THICKNESS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MAX_BRUSH_THICKNESS,
    MAX_CAMERA_READ_FAILURES,
    MIN_BRUSH_THICKNESS,
    SAVE_PREFIX,
    SHOW_FPS_BY_DEFAULT,
    SHOW_HELP_BY_DEFAULT,
    TOAST_DURATION_SECONDS,
    TRANSPARENT_PREFIX,
    WINDOW_NAME,
)
from src.gesture_logic import GestureInterpreter
from src.hand_tracker import HandTracker
from src.ui import OverlayState, UIManager
from src.utils import build_save_path


@dataclass
class AppState:
    """Mutable runtime state for app controls and user-facing toggles."""

    brush_thickness: int = DRAW_THICKNESS
    eraser_thickness: int = ERASER_THICKNESS
    mirror_frame: bool = DEFAULT_MIRROR_FRAME
    landmarks_visible: bool = DRAW_LANDMARKS_BY_DEFAULT
    help_visible: bool = SHOW_HELP_BY_DEFAULT
    fps_visible: bool = SHOW_FPS_BY_DEFAULT
    toast_message: str = ""
    toast_expires_at: float = 0.0

    def set_toast(self, message: str) -> None:
        self.toast_message = message
        self.toast_expires_at = time.monotonic() + TOAST_DURATION_SECONDS

    def current_toast(self) -> str:
        return self.toast_message if time.monotonic() < self.toast_expires_at else ""


def open_camera():
    """Open the first available webcam from the configured camera indices."""
    candidate_indexes = [CAMERA_INDEX, *[index for index in CAMERA_FALLBACK_INDEXES if index != CAMERA_INDEX]]
    last_error = "No camera indexes were attempted."

    for camera_index in candidate_indexes:
        capture = cv2.VideoCapture(camera_index, CAMERA_BACKEND)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        if capture.isOpened():
            return capture, camera_index

        last_error = f"Unable to open camera index {camera_index}."
        capture.release()

    return None, last_error


def estimate_fps(frame_times: deque[float]) -> float:
    """Estimate FPS from a small sliding window of timestamps."""
    if len(frame_times) < 2:
        return 0.0

    elapsed = frame_times[-1] - frame_times[0]
    if elapsed <= 0:
        return 0.0

    return (len(frame_times) - 1) / elapsed


def handle_keypress(key: int, app_state: AppState, canvas: CanvasManager, latest_frame) -> bool:
    """Handle keyboard shortcuts. Return True when the app should exit."""
    if key in (27, ord("e"), ord("E")):
        return True
    if key in (ord("c"), ord("C"), ord("n"), ord("N")):
        canvas.clear()
        app_state.set_toast("Started a new blank canvas")
    elif key in (ord("m"), ord("M")):
        app_state.mirror_frame = not app_state.mirror_frame
        app_state.set_toast(f"Mirror {'enabled' if app_state.mirror_frame else 'disabled'}")
    elif key in (ord("l"), ord("L")):
        app_state.landmarks_visible = not app_state.landmarks_visible
        app_state.set_toast(f"Landmarks {'shown' if app_state.landmarks_visible else 'hidden'}")
    elif key in (ord("h"), ord("H")):
        app_state.help_visible = not app_state.help_visible
        app_state.set_toast(f"Help overlay {'shown' if app_state.help_visible else 'hidden'}")
    elif key in (ord("f"), ord("F")):
        app_state.fps_visible = not app_state.fps_visible
        app_state.set_toast(f"FPS {'shown' if app_state.fps_visible else 'hidden'}")
    elif key == ord("["):
        app_state.brush_thickness = max(MIN_BRUSH_THICKNESS, app_state.brush_thickness - BRUSH_STEP)
        app_state.set_toast(f"Brush size {app_state.brush_thickness}px")
    elif key == ord("]"):
        app_state.brush_thickness = min(MAX_BRUSH_THICKNESS, app_state.brush_thickness + BRUSH_STEP)
        app_state.set_toast(f"Brush size {app_state.brush_thickness}px")
    elif key in (ord("z"), ord("Z")):
        app_state.set_toast("Undo stroke" if canvas.undo() else "Nothing to undo")
    elif key in (ord("y"), ord("Y")):
        app_state.set_toast("Redo stroke" if canvas.redo() else "Nothing to redo")
    elif key in (ord("s"), ord("S")):
        save_path = build_save_path(SAVE_PREFIX)
        app_state.set_toast(f"Saved canvas to {save_path}" if cv2.imwrite(save_path, canvas.canvas) else "Failed to save canvas image")
    elif key in (ord("t"), ord("T")):
        save_path = build_save_path(TRANSPARENT_PREFIX)
        app_state.set_toast(
            f"Saved transparent PNG to {save_path}" if canvas.save_transparent(save_path) else "Failed to save transparent PNG"
        )
    elif key in (ord("p"), ord("P")) and latest_frame is not None:
        save_path = build_save_path(COMPOSITE_PREFIX)
        app_state.set_toast(
            f"Saved composite screenshot to {save_path}" if cv2.imwrite(save_path, latest_frame) else "Failed to save composite screenshot"
        )
    return False


def main() -> int:
    """Run the webcam drawing application."""
    capture, camera_result = open_camera()
    if capture is None:
        print("Error: Unable to open webcam.")
        print(camera_result)
        print("Check camera permissions, close other camera apps, or change CAMERA_INDEX.")
        return 1

    tracker = HandTracker()
    gestures = GestureInterpreter()
    canvas = CanvasManager(FRAME_WIDTH, FRAME_HEIGHT)
    ui = UIManager()
    app_state = AppState()
    frame_times: deque[float] = deque(maxlen=30)
    read_failures = 0
    latest_composed_frame = None
    app_state.set_toast(f"Camera {camera_result} ready")

    try:
        while True:
            success, frame = capture.read()
            if not success:
                read_failures += 1
                if read_failures >= MAX_CAMERA_READ_FAILURES:
                    print("Error: Repeated camera read failures. Exiting.")
                    break
                continue

            read_failures = 0
            frame_times.append(cv2.getTickCount() / cv2.getTickFrequency())

            if app_state.mirror_frame:
                frame = cv2.flip(frame, 1)

            tracked_frame, hand_data = tracker.process(frame, draw_landmarks=app_state.landmarks_visible)
            gesture_state = gestures.update(hand_data, ui.palette_boxes) if hand_data else gestures.idle_state()
            fps = estimate_fps(frame_times)
            if app_state.fps_visible:
                gesture_state.status_text = f"{gesture_state.status_text} | FPS {fps:.1f}"

            selected_color = ui.resolve_color(gesture_state.active_color_name)
            canvas.update_drawing(
                gesture_state=gesture_state,
                pointer=gesture_state.pointer,
                color=selected_color,
                brush_thickness=app_state.brush_thickness,
                eraser_thickness=app_state.eraser_thickness,
            )

            overlay_state = OverlayState(
                brush_thickness=app_state.brush_thickness,
                eraser_thickness=app_state.eraser_thickness,
                mirror_enabled=app_state.mirror_frame,
                landmarks_visible=app_state.landmarks_visible,
                help_visible=app_state.help_visible,
                fps_visible=app_state.fps_visible,
                fps_value=fps,
                can_undo=bool(canvas.strokes),
                can_redo=bool(canvas.redo_stack),
                toast_message=app_state.current_toast(),
            )
            latest_composed_frame = ui.compose(
                frame=tracked_frame,
                canvas=canvas.canvas,
                gesture_state=gesture_state,
                overlay_state=overlay_state,
            )

            cv2.imshow(WINDOW_NAME, latest_composed_frame)
            key = cv2.waitKey(1) & 0xFF
            if handle_keypress(key, app_state, canvas, latest_composed_frame):
                break

    finally:
        canvas.end_stroke()
        tracker.close()
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Application entry point for the Real-Time AI Air Canvas."""

from __future__ import annotations

from collections import deque

import cv2

from src.canvas_manager import CanvasManager
from src.config import (
    CAMERA_FALLBACK_INDEXES,
    CAMERA_INDEX,
    DEFAULT_MIRROR_FRAME,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MAX_CAMERA_READ_FAILURES,
    WINDOW_NAME,
)
from src.gesture_logic import GestureInterpreter
from src.hand_tracker import HandTracker
from src.ui import UIManager
from src.utils import build_save_path


def open_camera():
    """Open the first available webcam from the configured camera indices."""
    candidate_indexes = [CAMERA_INDEX, *[index for index in CAMERA_FALLBACK_INDEXES if index != CAMERA_INDEX]]
    last_error = "No camera indexes were attempted."

    for camera_index in candidate_indexes:
        capture = cv2.VideoCapture(camera_index)
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


def main() -> int:
    """Run the webcam drawing application."""
    capture, camera_result = open_camera()
    if capture is None:
        print("Error: Unable to open webcam.")
        print(camera_result)
        print("Check camera permissions, close other camera apps, or change CAMERA_INDEX.")
        return 1

    tracker = HandTracker(max_num_hands=1)
    gestures = GestureInterpreter()
    canvas = CanvasManager(FRAME_WIDTH, FRAME_HEIGHT)
    ui = UIManager()
    mirror_frame = DEFAULT_MIRROR_FRAME
    frame_times: deque[float] = deque(maxlen=30)
    read_failures = 0

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

            if mirror_frame:
                frame = cv2.flip(frame, 1)
            tracked_frame, hand_data = tracker.process(frame)

            if hand_data:
                gesture_state = gestures.update(hand_data, ui.palette_boxes)
            else:
                gesture_state = gestures.idle_state()

            fps = estimate_fps(frame_times)
            gesture_state.status_text = f"{gesture_state.status_text} | FPS {fps:.1f}"
            current_point = gesture_state.pointer
            selected_color = ui.resolve_color(gesture_state.active_color_name)
            canvas.apply_gesture(
                gesture_state=gesture_state,
                pointer=current_point,
                color=selected_color,
            )

            composed = ui.compose(
                frame=tracked_frame,
                canvas=canvas.canvas,
                gesture_state=gesture_state,
            )

            cv2.imshow(WINDOW_NAME, composed)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("e"), ord("E")):
                break
            if key in (ord("c"), ord("C")):
                canvas.clear()
            if key in (ord("m"), ord("M")):
                mirror_frame = not mirror_frame
            if key in (ord("s"), ord("S")):
                save_path = build_save_path()
                if cv2.imwrite(save_path, canvas.canvas):
                    print(f"Saved canvas to {save_path}")
                else:
                    print("Error: Failed to save the canvas image.")

    finally:
        tracker.close()
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

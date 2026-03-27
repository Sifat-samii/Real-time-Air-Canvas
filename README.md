# Real-Time AI Air Canvas

A polished local desktop computer-vision project that turns a webcam into an air-drawing tool. It uses MediaPipe Hands for live hand tracking, OpenCV for rendering and interaction, and NumPy for fast canvas operations. The app stays lightweight and easy to run while adding enough reliability and UX polish to feel like a serious mini product.

## Features

- Live webcam hand tracking with MediaPipe Hands
- Smooth index-finger air drawing with moving average smoothing, exponential smoothing, and adaptive interpolation
- Clean palette interaction with hover confirmation and safer hitboxes
- Eraser mode, clear gesture, new blank canvas, undo, and redo
- Adjustable brush thickness with separate eraser thickness
- Save opaque canvas, transparent export, and composite screenshot
- Mirror toggle, landmark toggle, help toggle, and FPS toggle
- Structured stroke history for better state management and undo/redo
- Clean HUD with tool, brush size, mode, status, and shortcut hints

## Tech Stack

- Python
- OpenCV
- MediaPipe Hands
- NumPy

## Project Structure

```text
air_canvas/
  main.py
  requirements.txt
  README.md
  src/
    __init__.py
    config.py
    hand_tracker.py
    gesture_logic.py
    canvas_manager.py
    ui.py
    utils.py
```

## Architecture Summary

- `main.py`: app loop, camera startup, keyboard controls, runtime toggles, save/export handling
- `src/config.py`: central settings for camera, smoothing, gestures, brush sizes, UI defaults, and colors
- `src/hand_tracker.py`: MediaPipe wrapper that returns clean hand landmark data and supports landmark visibility toggling
- `src/gesture_logic.py`: finger-state detection, gesture classification, hover timing, and debounce/cooldown rules
- `src/canvas_manager.py`: structured stroke storage, canvas rendering, undo/redo, and transparent export
- `src/ui.py`: palette, HUD, help panel, toast feedback, and cursor rendering
- `src/utils.py`: geometry helpers, smoothing helpers, save-path helpers, blending, and transparent conversion

## Installation

Use Python 3.10 or 3.11. MediaPipe is the dependency most likely to fail on unsupported Python versions.

1. Open the `air_canvas` folder directly in VS Code.
2. Open a terminal in that folder.
3. Create a virtual environment:

```powershell
python -m venv .venv
```

4. Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once in the same terminal and try again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

5. Upgrade `pip` and install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## How To Run

From the VS Code terminal, inside the `air_canvas` folder:

```powershell
python main.py
```

## Controls

### Gestures

- `Only index finger up`: draw on the canvas
- `Hold index fingertip over a palette box`: select a color
- `Open palm for about 0.45 seconds`: toggle eraser mode
- `Fist for about 0.6 seconds`: clear the canvas

### Keyboard

- `[` and `]`: decrease or increase brush thickness
- `C`: clear the current canvas
- `N`: start a new blank canvas
- `Z`: undo the last stroke
- `Y`: redo the last undone stroke
- `S`: save the current canvas as a PNG
- `T`: save the current canvas as a transparent PNG
- `P`: save a screenshot of the composite output
- `M`: toggle webcam mirroring
- `L`: toggle landmark visibility
- `H`: toggle help overlay
- `F`: toggle FPS display
- `E` or `Esc`: exit the application

## Test Checklist

1. Run the app and confirm the webcam opens.
2. Raise only your index finger and draw slow and fast strokes.
3. Hover over a palette box and confirm the color changes only after a short hold.
4. Show an open palm long enough to toggle eraser mode.
5. Make a fist long enough to clear the canvas.
6. Draw multiple strokes, then test `Z` and `Y`.
7. Press `S`, `T`, and `P` to confirm files are written locally.
8. Toggle `M`, `L`, `H`, and `F` to verify the HUD updates correctly.

## Behavior Notes

- Drawing is disabled while interacting with the palette.
- Palette hitboxes are slightly inset to reduce accidental switches near box edges.
- Clear and eraser gestures require both hold time and release-to-rearm behavior.
- Undo and redo operate on complete strokes, not partial in-progress segments.
- Transparent export contains only the drawing, not the webcam frame.

## Known Limitations

- The project is optimized for one hand first.
- Thumb detection remains the most angle-sensitive part of the gesture model.
- Hand tracking quality still depends on lighting, camera quality, and background clutter.
- Gesture timings and smoothing thresholds may need local tuning for different webcams and laptops.
- Redo history is cleared when you draw a new stroke after undoing, which is expected behavior.

## Manual Tuning Areas

If you want to tune behavior further, adjust values in `src/config.py`:

- smoothing and interpolation values
- palette hover timing
- gesture hold durations
- brush and eraser thickness defaults
- camera resolution and preferred camera index
- UI defaults for mirror, help, FPS, and landmark visibility

## Future Improvements

- Add runtime controls for eraser thickness
- Add saved session replay or time-lapse export
- Add handedness-aware calibration and thumb tuning
- Add a small settings panel instead of keyboard-only controls
- Add optional multi-hand tools or custom gesture mapping

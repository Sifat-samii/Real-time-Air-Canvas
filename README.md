# Real-Time AI Air Canvas

A local Python computer-vision project that turns a laptop webcam into an air-drawing tool. The app tracks one hand in real time with MediaPipe Hands, interprets a focused set of gestures, and renders smooth strokes onto a separate drawing canvas blended over the webcam feed.

## Features

- Real-time webcam feed with live hand landmark detection
- Smooth index-finger drawing with averaging, exponential smoothing, and interpolation
- Safer palette hit detection with hover confirmation and inset hitboxes
- Open-palm hold to toggle eraser mode
- Fist hold to clear the canvas with cooldown and release-to-rearm logic
- Separate persistent drawing canvas blended over the live camera frame
- Clear on-screen UI for tool, brush color, mode, status, and controls
- Save the current drawing canvas to a PNG with `S`
- Toggle webcam mirroring with `M`
- Lightweight FPS display in the overlay

## Tech Stack

- Python
- OpenCV
- MediaPipe Hands
- NumPy

## Folder Structure

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

## Setup

Use Python 3.10 or 3.11. MediaPipe is the dependency most likely to fail on unsupported Python versions.

1. Open the `air_canvas` folder directly in VS Code.
2. Open the integrated terminal in that folder.
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

## Run

From the VS Code terminal, inside the `air_canvas` folder:

```powershell
python main.py
```

## Controls

### Gestures

- `Only index finger up`: draw on the canvas
- `Hold index fingertip over a palette box`: select a brush color
- `Open palm for about 0.45 seconds`: toggle eraser mode
- `Fist for about 0.6 seconds`: clear the canvas

### Keyboard

- `C`: clear the canvas
- `S`: save the current drawing canvas as a PNG
- `M`: toggle mirrored webcam view
- `E`: exit the application

## Behavior Notes

- Drawing happens only when the classifier detects the index-only gesture.
- Palette selection requires a short hover, so passing through the top bar does not instantly switch colors.
- Palette hit detection uses slightly inset hitboxes, so edge grazing is less likely to trigger a color switch.
- Clear and eraser gestures require both a hold duration and a release before they can trigger again.
- Stroke smoothing uses a short moving average, an exponential smoother, a deadzone, and extra interpolation during faster movement.
- Saved images contain the drawing canvas only, not the webcam frame.

## Foolproof Setup Checklist

- Run the app from the `air_canvas` directory, not the parent workspace.
- Make sure no other app is currently using the webcam.
- If `mediapipe` fails to install, switch to Python 3.10 or 3.11.
- If the app cannot open the webcam, edit `CAMERA_INDEX` in `src/config.py` and try another value.
- The app will also try fallback camera indexes automatically.
- If the webcam window opens but stays black, close Zoom, Teams, Discord, browser tabs, or camera utilities.
- If gestures feel unstable, improve lighting and keep your palm facing the camera more directly.
- If the webcam appears backwards for your use case, press `M` to toggle mirroring.

## Known Limitations

- The project is tuned for one-hand use first.
- Thumb detection is still the least stable part because thumb orientation changes a lot with camera angle.
- Hand tracking quality depends heavily on lighting, camera quality, and background clutter.
- MediaPipe can still lose the hand during fast motion or occlusion.
- Gesture timing and smoothing thresholds may need local tuning depending on webcam resolution and latency.

## Future Improvements

- Save the composited webcam-plus-canvas view as an optional export
- Add undo and redo support
- Add thickness controls and a proper tool tray
- Add handedness-aware calibration or per-user gesture tuning
- Add multi-hand support
- Add optional gesture customization

## Camera And OS Notes

- If the webcam does not open, check OS camera permissions for Python and VS Code.
- On some laptops, OpenCV works better after other camera apps are fully closed.
- If your machine has multiple cameras, the preferred camera index may not be `0`.

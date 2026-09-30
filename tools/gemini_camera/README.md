# Gemini RGB-D Camera Utilities

This directory keeps the local Gemini depth-camera utilities used to acquire
RGB-D examples for the paper workflow.

Main files:

- `gemini_camera.py`: OpenNI/Orbbec capture and frame export helper.
- `openni_gemini.py`: lower-level OpenNI access helper.
- `rgb_uvc_tune.py`: RGB UVC brightness/exposure tuning helper.
- `uvc_control.ps1`: Windows UVC control wrapper.
- `requirements.txt`: camera-specific Python dependencies from the local workspace.

Notes from the local setup:

- Connect the Gemini camera through a USB 3 port.
- Verify the device with OrbbecViewer before running Python capture scripts.
- If RGB is too dark, tune exposure/gain through `rgb_uvc_tune.py` or the vendor viewer.
- Some USB enumeration warnings can appear even when the RGB camera is visible through a normal webcam path. Depth access still requires the Orbbec/OpenNI device stack.


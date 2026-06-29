# Hand Gesture Control (Computer Vision + Automation)

## Install
```bash
pip install -r requirements.txt
```

## Run
```bash
python gesture_control.py
```

## How it works
- OpenCV captures webcam frames
- MediaPipe Hands detects hand landmarks
- Heuristics determine which fingers are up
- Gestures trigger automation via `pyautogui`

## Gesture mapping
| Gesture | Action |
|---|---|
| 👍 Thumb Up | Volume Up |
| 👎 Thumb Down | Volume Down |
| ✌️ Index + Middle | Open browser (Ctrl+L → google.com) |
| ☝️ Index Finger | Move mouse (small relative move) |
| ✊ Fist | Pause media (play/pause + fallback) |
| 🖐️ Open Palm | Screenshot (Win+Shift+S) |

## Notes / Troubleshooting
- If volume/media keys do not work, adjust the hotkeys/press strings in `GestureController.act()`.
- Webcam should be index `0`. Change `cv2.VideoCapture(0)` if needed.


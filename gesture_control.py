import time
import sys
import math

import cv2
import mediapipe as mp
import pyautogui


# ----------------------------
# Gesture detection helpers
# ----------------------------

def _landmark_to_xy(lm):
    # lm is a mediapipe landmark with normalized coords
    return lm.x, lm.y


def _finger_is_up(hand_landmarks, finger_tip_id: int, finger_pip_id: int) -> bool:
    """Heuristic: finger tip is above (smaller y) than PIP joint (in image coords)."""
    lm = hand_landmarks.landmark
    tip = lm[finger_tip_id]
    pip = lm[finger_pip_id]
    return tip.y < pip.y


def _thumb_is_up(hand_landmarks, handedness_label: str) -> bool:
    """Heuristic: thumb tip is away from palm toward the outside.

    Works reasonably well for mirrored webcam setups.
    """
    lm = hand_landmarks.landmark

    # Thumb: tip=4, ip=3, mcp=2
    thumb_tip = lm[4]
    thumb_ip = lm[3]
    thumb_mcp = lm[2]

    # For Right hand: thumb points left in normalized x; for Left hand: points right.
    # Compare tip.x relative to thumb_mcp.x and also compare vertical-ish.
    if handedness_label.lower() == "right":
        return thumb_tip.x < thumb_mcp.x and thumb_tip.y < thumb_ip.y
    else:
        return thumb_tip.x > thumb_mcp.x and thumb_tip.y < thumb_ip.y


def _finger_is_extended(hand_landmarks, finger_tip_id: int, finger_mcp_id: int, finger_pip_id: int) -> bool:
    """More robust finger extension check.

    Uses a combination of:
    - tip vs pip (vertical-ish extension)
    - tip vs mcp distance (helps when hand is tilted)
    """
    lm = hand_landmarks.landmark
    tip = lm[finger_tip_id]
    pip = lm[finger_pip_id]
    mcp = lm[finger_mcp_id]

    # In image coordinates: y decreases upward
    vertical_ok = tip.y < pip.y

    # Distance-based heuristic to reduce false positives when hand is rotated
    # (normalized coords -> compare z too)
    dx = tip.x - mcp.x
    dy = tip.y - mcp.y
    dz = tip.z - mcp.z
    dist_tip_mcp = math.sqrt(dx * dx + dy * dy + dz * dz)

    # Heuristic threshold: extended fingers tend to have larger tip-mcp distance.
    # Tuneable; chosen to work reasonably across 640x480 webcam with default MediaPipe.
    return vertical_ok and dist_tip_mcp > 0.12


def detect_gesture(hand_landmarks, handedness_label: str) -> tuple[str | None, dict[str, bool]]:
    """Returns (gesture, debug_states).

    gesture is one of:
    thumb_up, thumb_down, peace, index, fist, open_palm, or None.

    debug_states contains per-finger booleans so you can tune thresholds.
    """

    # MediaPipe Hands landmark IDs

    # Index: tip=8, pip=6, mcp=5
    # Middle: tip=12, pip=10, mcp=9
    # Ring: tip=16, pip=14, mcp=13
    # Pinky: tip=20, pip=18, mcp=17

    index_up = _finger_is_extended(hand_landmarks, 8, 5, 6)
    middle_up = _finger_is_extended(hand_landmarks, 12, 9, 10)
    ring_up = _finger_is_extended(hand_landmarks, 16, 13, 14)
    pinky_up = _finger_is_extended(hand_landmarks, 20, 17, 18)

    # Thumb: keep the original heuristic (it works better in mirrored setups)
    # but gate it with whether thumb is actually separated from palm a bit.
    thumb_up = _thumb_is_up(hand_landmarks, handedness_label)

    # Gate thumb decision with basic separation from thumb MCP
    lm = hand_landmarks.landmark
    thumb_tip = lm[4]
    thumb_mcp = lm[2]
    dx = thumb_tip.x - thumb_mcp.x
    dy = thumb_tip.y - thumb_mcp.y
    dz = thumb_tip.z - thumb_mcp.z
    thumb_dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    if thumb_dist < 0.08:
        thumb_up = False

    debug_states = {
        "index_up": index_up,
        "middle_up": middle_up,
        "ring_up": ring_up,
        "pinky_up": pinky_up,
        "thumb_up": thumb_up,
    }

    # Fist: all non-thumb fingers down and thumb down-ish
    if (not index_up) and (not middle_up) and (not ring_up) and (not pinky_up) and (not thumb_up):
        return "fist", debug_states

    # Open palm: thumb up + all four fingers up
    if thumb_up and index_up and middle_up and ring_up and pinky_up:
        return "open_palm", debug_states

    # Peace: index + middle up, ring + pinky down, thumb down
    if index_up and middle_up and (not ring_up) and (not pinky_up) and (not thumb_up):
        return "peace", debug_states

    # Index: only index up
    if index_up and (not middle_up) and (not ring_up) and (not pinky_up) and (not thumb_up):
        return "index", debug_states

    # Thumb up only (roughly): thumb up, other fingers down
    if thumb_up and (not index_up) and (not middle_up) and (not ring_up) and (not pinky_up):
        return "thumb_up", debug_states

    # Thumb down: thumb not up, other fingers down
    if (not thumb_up) and (not index_up) and (not middle_up) and (not ring_up) and (not pinky_up):
        return "thumb_down", debug_states

    return None, debug_states



# ----------------------------
# Gesture -> action mapping
# ----------------------------

class GestureController:
    def __init__(self):
        # Basic anti-spam / debouncing
        self.last_gesture = None
        self.last_time = 0.0
        self.cooldown_sec = 1.0

        # Extra cooldown dedicated for right-click (to reduce accidental touches)
        self.right_click_last_time = 0.0
        self.right_click_cooldown_sec = 1.8

        # pyautogui settings
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.02


    def _can_trigger(self, gesture: str) -> bool:
        now = time.time()
        if gesture != self.last_gesture:
            self.last_gesture = gesture
            self.last_time = now
            return True
        return (now - self.last_time) >= self.cooldown_sec

    def act(self, gesture: str) -> None:
        # Special-case right click: use a dedicated cooldown so near-fist doesn't spam.
        if gesture == "fist":
            now = time.time()
            if (now - self.right_click_last_time) < self.right_click_cooldown_sec:
                return
            self.right_click_last_time = now

        if not self._can_trigger(gesture):
            return

        # Map gestures to actions
        # Note: On Windows, pyautogui hotkeys are generally reliable for volume/media.
        # If volume keys don't work, user can switch to Windows key sequences.
        try:
            if gesture == "thumb_up":
                # Volume Up
                pyautogui.press("volumeup")


            elif gesture == "thumb_down":
                # Volume Down
                pyautogui.press("volumedown")

            elif gesture == "peace":
                # Open Browser (Ctrl+L then type a URL) - more robust than relying on OS browser launcher
                pyautogui.hotkey("ctrl", "l")
                pyautogui.typewrite("https://www.google.com", interval=0.01)
                pyautogui.press("enter")

            elif gesture in ("index", "index_move"):
                # Move mouse (absolute): handled in main() by passing index_move.
                # This act() keeps the click mapping only.
                pass

            elif gesture == "index_click":
                # Left click
                pyautogui.click()

            elif gesture == "double_click":
                # Double click
                pyautogui.doubleClick()



            elif gesture == "fist":
                # Right click
                pyautogui.click(button="right")

            # open_palm previously triggered screenshot (Win+Shift+S).
            # Disabled to prevent accidental interruptions.
            elif gesture == "open_palm":
                pass

        except Exception:

            # Avoid crashing on input/OS issues
            pass


# ----------------------------
# Main application
# ----------------------------

def main():
    # Uncomment for debugging cursor automation
    # pyautogui._pyautogui_win._display = 'something'

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        sys.exit(1)

    # Improve stability
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    controller = GestureController()

    gesture_label_map = {
        "thumb_up": "👍 Thumb Up (Vol+)",
        "thumb_down": "👎 Thumb Down (Vol-)",
        "peace": "✌️ Peace (Open Browser)",
        "index": "☝️ Index",
        "index_click": "🖱️ Click (Pinch)",
        "fist": "✊ Fist (Right Click)",
        "open_palm": "🖐️ Open Palm (Screenshot)",
        None: "",
    }


    # --- gesture smoothing (reduces flicker / improves accuracy) ---
    # Require the same detected gesture for a few consecutive frames.
    stable_window = 6
    stable_counts: dict[str | None, int] = {}

    # Absolute cursor mapping smoothing
    cursor_smooth = 0.25  # 0..1 (higher = follows hand more closely)
    last_cursor_x: float | None = None
    last_cursor_y: float | None = None

    # Click gating: avoid multiple clicks while keeping pinch held
    last_click_time = 0.0
    click_cooldown_sec = 0.6

    while True:

        success, frame = cap.read()
        if not success:
            break

        # Flip for more natural interaction (and to match handedness heuristics)
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        detected = None

        hand_index_tip = None
        hand_index_mcp = None
        pinch = False

        if results.multi_hand_landmarks:
            # Use first hand
            hand_landmarks = results.multi_hand_landmarks[0]
            handedness_label = "Right"
            if results.multi_handedness and len(results.multi_handedness) > 0:
                handedness_label = results.multi_handedness[0].classification[0].label

            detected, debug_states = detect_gesture(hand_landmarks, handedness_label)

            # --- Cursor position + click pinch (thumb + index distance) ---
            # Index tip = 8, Thumb tip = 4
            lm = hand_landmarks.landmark
            hand_index_tip = lm[8]
            # pinch if thumb tip is close to index tip in 3D (normalized space)
            thumb_tip = lm[4]
            dx = thumb_tip.x - hand_index_tip.x
            dy = thumb_tip.y - hand_index_tip.y
            dz = thumb_tip.z - hand_index_tip.z
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            pinch = dist < 0.08

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)


        # Apply smoothing before triggering actions
        if detected is not None:
            stable_counts[detected] = stable_counts.get(detected, 0) + 1
        else:
            # reset when no hand / no gesture
            stable_counts.clear()

        # choose gesture with max count in window
        stable_gesture = None
        if stable_counts:
            stable_gesture = max(stable_counts, key=lambda k: stable_counts[k])

        # --- Absolute cursor mapping using index fingertip ---
        if hand_index_tip is not None:
            screen_w, screen_h = pyautogui.size()
            # MediaPipe coords are normalized [0..1] with origin at top-left of image.
            target_x = hand_index_tip.x * screen_w
            target_y = hand_index_tip.y * screen_h

            # Expand mapping range so the cursor can reach the screen edges.
            # Without this, small camera movement often never maps to the full 0..screen range.
            # If cursor still can't reach edges, increase this.
            # tune: 1.25..2.0
            expand = 1.6

            target_x = (target_x - screen_w / 2) * expand + screen_w / 2
            target_y = (target_y - screen_h / 2) * expand + screen_h / 2

            # Clamp (in case landmarks go outside frame)
            target_x = max(0, min(screen_w - 1, target_x))
            target_y = max(0, min(screen_h - 1, target_y))


            if last_cursor_x is None or last_cursor_y is None:
                last_cursor_x = target_x
                last_cursor_y = target_y
            else:
                # Smooth movement
                last_cursor_x = (1 - cursor_smooth) * last_cursor_x + cursor_smooth * target_x
                last_cursor_y = (1 - cursor_smooth) * last_cursor_y + cursor_smooth * target_y

            pyautogui.moveTo(last_cursor_x, last_cursor_y, duration=0.0)

        # Click: pinch thumb+index
        now = time.time()
        if pinch and (now - last_click_time) >= click_cooldown_sec:
            # Pinch & hold -> double click
            controller.act("double_click")
            last_click_time = now


        # Trigger other actions only when stable for N frames
        # Extra gating for right-click to avoid accidental touches:
        # - Require strong "fist" stability (more frames)
        # - Require that all non-thumb fingers are actually folded (tip closer to MCP)
        if stable_gesture is not None:
            # While pinching (left click), avoid triggering other actions
            # (right-click, volume, etc.) because it can interfere with cursor placement.
            if pinch:
                stable_frames = 0
            else:
                stable_frames = stable_counts.get(stable_gesture, 0)



            if stable_gesture == "fist":
                right_click_stable_window = 10
                # Only allow right-click when the left-click pinch is NOT active.
                # This prevents mixed/accidental click gestures from firing the wrong button.
                if pinch:
                    pass
                elif stable_frames >= right_click_stable_window and hand_landmarks is not None:

                    lm = hand_landmarks.landmark

                    # Finger tip ids: index 8, middle 12, ring 16, pinky 20
                    # Finger MCP ids: index 5, middle 9, ring 13, pinky 17
                    # If finger is folded, distance tip->mcp will be smaller.
                    def tip_to_mcp_dist(tip_id: int, mcp_id: int) -> float:
                        t = lm[tip_id]
                        m = lm[mcp_id]
                        dx = t.x - m.x
                        dy = t.y - m.y
                        dz = t.z - m.z
                        return math.sqrt(dx * dx + dy * dy + dz * dz)

                    index_folded = tip_to_mcp_dist(8, 5) < 0.14
                    middle_folded = tip_to_mcp_dist(12, 9) < 0.14
                    ring_folded = tip_to_mcp_dist(16, 13) < 0.14
                    pinky_folded = tip_to_mcp_dist(20, 17) < 0.14

                    if index_folded and middle_folded and ring_folded and pinky_folded:
                        controller.act("fist")
                        stable_counts.clear()
            else:
                if stable_frames >= stable_window:
                    controller.act(stable_gesture)
                    stable_counts.clear()



        # Overlay
        label = gesture_label_map.get(detected, "")
        if label:
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(frame, "ESC: Quit", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Gesture Control", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


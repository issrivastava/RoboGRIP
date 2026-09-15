"""
RoboGrip Step 3: Static Gesture Classifier (rule-based baseline)
Turns 21 landmarks -> finger count -> gesture -> robot command.

Gestures:
  OPEN (5 fingers)       -> STOP
  FIST (0)               -> GRIP
  PEACE (index+middle)   -> RELEASE
  POINT (index only)     -> MOVE (direction from hand position)
  THREE (3 fingers)      -> UP
  FOUR (4 fingers)       -> DOWN
  THUMB_UP (thumb only)  -> UP (alt)

Run:
  venv\\Scripts\\python.exe src\\03_gesture_classifier.py
Press 'q' to quit.
"""
import cv2
import time
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "hand_landmarker.task")
HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS

# Landmark indices
TIP = [4, 8, 12, 16, 20]
PIP = [3, 6, 10, 14, 18]
NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]


def fingers_up(hand_lms, handedness="Right"):
    """Return [thumb,index,middle,ring,pinky] 1=open, 0=closed.
    Scale-invariant: thresholds relative to hand size (wrist->middle_mcp),
    so far/close hands work the same. Fixes FIST->UP bug.
    """
    import math
    fingers = []

    def dist(a, b):
        return math.hypot(hand_lms[a].x - hand_lms[b].x,
                          hand_lms[a].y - hand_lms[b].y)
    try:
        hand_size = dist(0, 9)
        if hand_size < 1e-6:
            hand_size = 0.2

        # Thumb: tip(4)-IP(3) + distance to index-base(5) to separate
        # FIST vs THUMB-UP. Your fist up varies 0.18-0.37, so up alone overlaps.
        # FIST: thumb tip folded near index-base => d45 small.
        # THUMB-UP/OPEN: tip far from index-base => d45 large.
        dx = hand_lms[4].x - hand_lms[3].x
        dy = hand_lms[4].y - hand_lms[3].y
        spread = abs(dx) / hand_size
        up = -dy / hand_size  # positive when tip above IP (thumbs-up)
        down = dy / hand_size  # positive when tip below IP (thumbs-down)
        d45 = dist(4, 5) / hand_size
        # Your real data:
        # THUMB-UP: spread 0.028, up 0.344, d45 0.637 -> OPEN
        # FIST: spread 0.102, up 0.319, d45 0.232 -> CLOSED
        # up alone overlaps (0.344 vs 0.319), but d45 separates (0.637 vs 0.232).
        # Added down for THUMB-DOWN (tip below IP).
        thumb_open = ((spread > 0.25) or (up > 0.30) or (down > 0.30)) and (d45 > 0.40)
    except Exception:
        thumb_open = False
        hand_size = 0.2
    fingers.append(1 if thumb_open else 0)

    # Other 4 fingers: tip above PIP by >12% of hand size = open
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        open_ = hand_lms[tip].y < hand_lms[pip].y - (0.12 * hand_size)
        fingers.append(1 if open_ else 0)
    return fingers


def classify(hand_lms, handedness="Right"):
    import math
    # Strong THUMB-DOWN shortcut BEFORE finger checks:
    # Sideways hand breaks y-based finger logic, so use thumb-only signal.
    # Your data: THUMB_DOWN down=0.386 d45=1.222 (very far) vs FIST d45=0.23.
    try:
        def _d(a, b):
            return math.hypot(hand_lms[a].x-hand_lms[b].x, hand_lms[a].y-hand_lms[b].y)
        _hs = max(_d(0, 9), 1e-6)
        _down = (hand_lms[4].y - hand_lms[3].y) / _hs
        _d45 = _d(4, 5) / _hs
        if _down > 0.30 and _d45 > 0.80:
            return "THUMB_DOWN", "DOWN", 1, [1, 0, 0, 0, 0]
    except Exception:
        pass

    f = fingers_up(hand_lms, handedness)
    count = sum(f)
    idx, mid, ring, pinky, thumb = f[1], f[2], f[3], f[4], f[0]

    # Specific patterns first (more reliable than raw count)
    if f == [0, 0, 0, 0, 0]:
        return "FIST", "GRIP", count, f
    # Tolerant OPEN: 4 fingers open = STOP even if thumb tucked.
    # Fixes OPEN showing as FOUR/DOWN when thumb spread is small.
    if f[1:] == [1, 1, 1, 1]:
        return "OPEN_PALM", "STOP", count, f
    if f == [0, 1, 1, 0, 0]:
        return "PEACE", "RELEASE", count, f
    if f == [0, 1, 0, 0, 0]:
        return "POINT", "MOVE", count, f
    if f == [1, 0, 0, 0, 0]:
        # Thumb-only: direction decides UP vs DOWN (intuitive!)
        try:
            # tip(4) above IP(3) = thumbs-up, below = thumbs-down
            if hand_lms[4].y < hand_lms[3].y:
                return "THUMB_UP", "UP", count, f
            else:
                return "THUMB_DOWN", "DOWN", count, f
        except Exception:
            return "THUMB_UP", "UP", count, f
    if f == [0, 0, 0, 0, 1]:
        return "PINKY", "DOWN", count, f
    if f == [0, 1, 1, 1, 0]:
        return "THREE", "UP", count, f
    if count == 3:
        return "THREE", "UP", count, f
    if count == 1 and thumb == 1:
        return "THUMB_UP", "UP", count, f
    if count == 1:
        return "ONE", "LEFT", count, f
    if count == 2:
        return "TWO", "RIGHT", count, f
    return "UNKNOWN", "HOLD", count, f


def draw_hand(frame, hand_lms, color=(0, 255, 0)):
    h, w, _ = frame.shape
    for conn in HAND_CONNECTIONS:
        x0, y0 = int(hand_lms[conn.start].x * w), int(hand_lms[conn.start].y * h)
        x1, y1 = int(hand_lms[conn.end].x * w), int(hand_lms[conn.end].y * h)
        cv2.line(frame, (x0, y0), (x1, y1), color, 2)
    for lm in hand_lms:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)


def main():
    base = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base, num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.VIDEO
    )
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    import time as _t
    _t.sleep(1.0)

    print("Gestures: FIST=GRIP | OPEN=STOP | PEACE=RELEASE | POINT=MOVE | THUMB-UP=UP | THUMB-DOWN=PINKY=DOWN")
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect_for_video(mp_img, int(time.time() * 1000))

            y0 = 30
            if res.hand_landmarks:
                colors = [(0, 255, 0), (255, 0, 0)]
                for i, hand_lms in enumerate(res.hand_landmarks):
                    handed = "Right"
                    if res.handedness and i < len(res.handedness):
                        handed = res.handedness[i][0].category_name
                    gesture, cmd, cnt, f = classify(hand_lms, handed)
                    draw_hand(frame, hand_lms, colors[i % 2])
                    h, w, _ = frame.shape
                    wx = int(hand_lms[0].x * w)
                    wy = int(hand_lms[0].y * h) - 25
                    cv2.putText(frame, f"H{i+1} {handed} {gesture} -> {cmd} ({cnt})",
                                (max(5, wx-80), max(20, wy)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i % 2], 2)
                    # Side panel
                    cv2.putText(frame, f"H{i+1}: {gesture} => {cmd}",
                                (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i % 2], 2)
                    y0 += 30
                    # Debug fingers
                    cv2.putText(frame, f"  fingers {f}",
                                (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    y0 += 25
            else:
                cv2.putText(frame, "Show hand: FIST / OPEN / PEACE / POINT",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.putText(frame, "FIST=GRIP OPEN=STOP PEACE=RELEASE POINT=MOVE THUMBup=UP THUMBdown=DOWN",
                        (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.imshow("RoboGrip - Gesture Classifier", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

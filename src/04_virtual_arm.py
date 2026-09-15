"""
RoboGrip Step 4: Temporal Validation + Confidence + Virtual Arm + Safety
- Hold gesture 8 frames (~0.4s) to accept (temporal validation)
- Require handedness score >0.6 (confidence-aware)
- Virtual 2-link arm with workspace limits + E-STOP
- Controls:
    FIST -> GRIP (close), PEACE -> RELEASE (open)
    THUMB_UP / THREE -> shoulder UP, THUMB_DOWN / PINKY -> shoulder DOWN
    ONE (1 finger) -> elbow LEFT, TWO (2 fingers) -> elbow RIGHT
    OPEN -> STOP (freeze), POINT -> small forward nudge
    Both hands OPEN at once OR key 'e' -> E-STOP (latch). 'r' to reset. 'q' quit.

Run:
  venv\\Scripts\\python.exe src\\04_virtual_arm.py
"""
import cv2
import time
import os
import math
import collections
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Reuse your tuned classifier from Step 3
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "gc", os.path.join(os.path.dirname(__file__), "03_gesture_classifier.py"))
gc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gc)

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "hand_landmarker.task")

HOLD_FRAMES = 5       # relaxed from 8: 5x in a row (~0.25s) for easier demo
CONF_THRESH = 0.5     # relaxed from 0.6 for typical room light
REQUIRED_SAME = 4     # majority: 4/5 same = accept (tolerates 1 flicker frame)

# Virtual arm limits (workspace constraints = safety chapter)
A1_MIN, A1_MAX = 20, 160    # shoulder deg
A2_MIN, A2_MAX = 10, 150    # elbow deg

def draw_virtual_arm(panel, a1_deg, a2_deg, grip_closed, estop, blocked_msg=""):
    h, w, _ = panel.shape
    panel[:] = (30, 30, 30)
    bx, by = w // 2, h - 40
    L1, L2 = 120, 100
    a1 = math.radians(a1_deg)
    a2 = math.radians(a2_deg)
    # shoulder -> elbow -> wrist
    ex = int(bx + L1 * math.cos(math.pi - a1))
    ey = int(by - L1 * math.sin(math.pi - a1))
    # elbow angle relative
    wx = int(ex + L2 * math.cos(math.pi - a1 - (math.pi - math.radians(180 - a2)) + math.pi/2*0))
    # Simpler 2-link FK:
    # Use standard: joint1 at base, link1 at a1 from vertical, joint2 bends by a2
    # Recompute cleanly:
    th1 = math.radians(90 - a1_deg + 45)  # map for nice screen pose
    # Actually keep simple visual: shoulder rotates, elbow rotates relative
    x1 = bx + int(L1 * math.cos(a1))
    y1 = by - int(L1 * math.sin(a1))
    x2 = x1 + int(L2 * math.cos(a1 - math.radians(a2_deg)))
    y2 = y1 - int(L2 * math.sin(a1 - math.radians(a2_deg)))

    col = (0, 0, 255) if estop else (0, 255, 0)
    cv2.circle(panel, (bx, by), 12, (200, 200, 200), -1)
    cv2.line(panel, (bx, by), (x1, y1), col, 8)
    cv2.circle(panel, (x1, y1), 8, (255, 255, 0), -1)
    cv2.line(panel, (x1, y1), (x2, y2), col, 6)
    # gripper
    g = 12 if grip_closed else 25
    ang = a1 - math.radians(a2_deg)
    gx = math.cos(ang); gy = -math.sin(ang)
    px, py = -gy, gx  # perpendicular
    cv2.line(panel, (x2, y2), (int(x2+20*gx+px*g), int(y2+20*gy+py*g)), (0, 165, 255), 4)
    cv2.line(panel, (x2, y2), (int(x2+20*gx-px*g), int(y2+20*gy-py*g)), (0, 165, 255), 4)

    cv2.putText(panel, f"A1 shoulder {a1_deg:.0f} [{A1_MIN}-{A1_MAX}]", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(panel, f"A2 elbow {a2_deg:.0f} [{A2_MIN}-{A2_MAX}]", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(panel, f"Grip {'CLOSED' if grip_closed else 'OPEN'}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    if estop:
        cv2.putText(panel, "E-STOP LATCHED press r", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    if blocked_msg:
        cv2.putText(panel, blocked_msg, (10, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    # workspace box
    cv2.rectangle(panel, (20, 100), (w-20, h-20), (80, 80, 80), 1)
    return (x2, y2)


def main():
    base = python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = vision.HandLandmarkerOptions(base_options=base, num_hands=2,
        min_hand_detection_confidence=0.5, running_mode=vision.RunningMode.VIDEO)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 640); cap.set(4, 480)
    time.sleep(1.0)

    hist = collections.deque(maxlen=HOLD_FRAMES)
    stable_cmd = "HOLD"
    stable_gesture = "BOOT"
    a1_deg, a2_deg = 90, 60
    grip_closed = False
    estop = False
    blocked_msg = ""
    blocked_until = 0

    print(f"Hold each gesture ~0.4s ({HOLD_FRAMES} frames) + confidence>{CONF_THRESH} to fire.")
    with vision.HandLandmarker.create_from_options(opts) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(time.time()*1000))

            raw_gesture, raw_cmd, conf = "NO_HAND", "HOLD", 0.0
            if res.hand_landmarks:
                # Use first hand for control, second hand for e-stop check
                i = 0
                hand_lms = res.hand_landmarks[0]
                handed = res.handedness[0][0].category_name if res.handedness else "Right"
                conf = float(res.handedness[0][0].score) if res.handedness else 0.0
                g, c, cnt, f = gc.classify(hand_lms, handed)
                # Confidence gate
                if conf < CONF_THRESH:
                    raw_gesture, raw_cmd = "LOW_CONF", "HOLD"
                else:
                    raw_gesture, raw_cmd = g, c
                # Draw both hands
                for j, hl in enumerate(res.hand_landmarks):
                    gc.draw_hand(frame, hl, [(0,255,0),(255,0,0)][j%2])

                # Two-hand OPEN e-stop (safety demo)
                if len(res.hand_landmarks) == 2:
                    try:
                        g2, c2, _, _ = gc.classify(res.hand_landmarks[1],
                            res.handedness[1][0].category_name if len(res.handedness)>1 else "Left")
                        if c == "STOP" and c2 == "STOP":
                            estop = True
                    except Exception:
                        pass

            hist.append(raw_gesture)
            # Temporal validation (majority vote, tolerates 1 flicker):
            if len(hist) == HOLD_FRAMES:
                from collections import Counter
                most, votes = Counter(hist).most_common(1)[0]
                if votes >= REQUIRED_SAME and most not in ("HOLD", "NO_HAND", "LOW_CONF", "UNKNOWN"):
                    if most != stable_gesture:
                        stable_gesture = most
                        m = {"FIST":"GRIP","OPEN_PALM":"STOP","PEACE":"RELEASE",
                             "POINT":"MOVE","THUMB_UP":"UP","THUMB_DOWN":"DOWN",
                             "PINKY":"DOWN","THREE":"UP","ONE":"LEFT","TWO":"RIGHT"}
                        stable_cmd = m.get(most, "HOLD")
                        print(f"FIRED {most}->{stable_cmd} conf {conf:.2f}")

            # Apply stable command to virtual arm (unless e-stop)
            blocked_msg = "" if time.time() > blocked_until else blocked_msg
            if not estop:
                step = 2.5
                if stable_cmd == "UP":
                    na = min(A1_MAX, a1_deg + step)
                    if na == a1_deg: blocked_msg = "BLOCKED: shoulder max (workspace)"
                    blocked_until = time.time()+0.5 if na==a1_deg else blocked_until
                    a1_deg = na
                elif stable_cmd == "DOWN":
                    na = max(A1_MIN, a1_deg - step)
                    if na == a1_deg: blocked_msg = "BLOCKED: shoulder min (workspace)"
                    blocked_until = time.time()+0.5 if na==a1_deg else blocked_until
                    a1_deg = na
                elif stable_cmd == "LEFT":
                    a2_deg = max(A2_MIN, a2_deg - step)
                elif stable_cmd == "RIGHT":
                    a2_deg = min(A2_MAX, a2_deg + step)
                elif stable_cmd == "MOVE":
                    a1_deg = min(A1_MAX, a1_deg + 1.0)
                    a2_deg = min(A2_MAX, a2_deg + 1.0)
                elif stable_cmd == "GRIP":
                    grip_closed = True
                elif stable_cmd == "RELEASE":
                    grip_closed = False
                # STOP/HOLD = freeze (do nothing)
            # else frozen

            # HUD on camera
            cv2.putText(frame, f"raw {raw_gesture}->{raw_cmd} conf {conf:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if conf>CONF_THRESH else (0,0,255), 2)
            prog = sum(1 for x in hist if x==raw_gesture) if raw_gesture!="NO_HAND" else 0
            cv2.putText(frame, f"hold {prog}/{HOLD_FRAMES} stable {stable_gesture}->{stable_cmd}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            if estop:
                cv2.putText(frame, "E-STOP - press r", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            panel = __import__("numpy").zeros((480, 400, 3), dtype=__import__("numpy").uint8)
            draw_virtual_arm(panel, a1_deg, a2_deg, grip_closed, estop, blocked_msg)
            combo = __import__("numpy").hstack([frame, cv2.resize(panel, (400, 480))])
            cv2.imshow("RoboGrip Step4 - hold to fire + virtual arm (e=E-STOP r=reset q=quit)", combo)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('e'):
                estop = True
            elif k == ord('r'):
                estop = False
                hist.clear()
                stable_cmd = "HOLD"; stable_gesture = "RESET"

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

"""
RoboGrip Step 2: Hand Landmark Detection (MediaPipe 1.x Tasks API)
Shows 21 hand landmarks.
Run from project root:
  venv\\Scripts\\python.exe src\\02_hand_tracking.py
Press 'q' to quit.
"""
import cv2
import time
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "hand_landmarker.task")

# Connections for drawing lines between dots
HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS


def draw_hand(frame, hand_landmarks, color=(0, 255, 0)):
    h, w, _ = frame.shape
    # Draw lines
    for conn in HAND_CONNECTIONS:
        x0 = int(hand_landmarks[conn.start].x * w)
        y0 = int(hand_landmarks[conn.start].y * h)
        x1 = int(hand_landmarks[conn.end].x * w)
        y1 = int(hand_landmarks[conn.end].y * h)
        cv2.line(frame, (x0, y0), (x1, y1), color, 2)
    # Draw 21 dots
    for lm in hand_landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.circle(frame, (x, y), 6, (255, 255, 255), 1)


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Download: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        return

    base = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.VIDEO
    )

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return

    # Warmup for Windows webcam
    import time as _t
    _t.sleep(1.0)

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        print("Show BOTH palms to camera, 40-70cm away, good light. Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int(time.time() * 1000)

            result = landmarker.detect_for_video(mp_img, ts)

            if result.hand_landmarks:
                colors = [(0, 255, 0), (255, 0, 0)]  # hand0=green, hand1=blue
                for i, hand_lms in enumerate(result.hand_landmarks):
                    draw_hand(frame, hand_lms, colors[i % 2])
                    # Label left/right if available
                    label = ""
                    if result.handedness and i < len(result.handedness):
                        label = result.handedness[i][0].category_name
                    h, w, _ = frame.shape
                    x = int(hand_lms[0].x * w)
                    y = int(hand_lms[0].y * h) - 20
                    cv2.putText(frame, f"Hand {i+1} {label}", (x-40, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i % 2], 2)
                cv2.putText(frame, f"Hands: {len(result.hand_landmarks)}/2",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No hand - show palm, good light, closer",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

            cv2.imshow("RoboGrip - Hand Tracking v1", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""
RoboGrip Step 6: Serial control (needs Arduino + arm.ino uploaded).
Sends stable commands to Arduino at 115200 baud.
Install: venv\\Scripts\\python.exe -m pip install pyserial
Run: venv\\Scripts\\python.exe src\\06_serial_control.py --port COM3
Keys: e=E-STOP, r=reset, q=quit. Both palms OPEN = E-STOP.
"""
import time, argparse, collections, os
import cv2, mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import importlib.util
_spec = importlib.util.spec_from_file_location("gc",
    os.path.join(os.path.dirname(__file__), "03_gesture_classifier.py"))
gc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(gc)

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "hand_landmarker.task")

def main(port="COM3"):
    import serial
    ser = serial.Serial(port, 115200, timeout=1)
    time.sleep(2.0)
    print(f"Connected {port}. Upload arduino/arm.ino first!")
    base = python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = vision.HandLandmarkerOptions(base_options=base, num_hands=2,
        min_hand_detection_confidence=0.5, running_mode=vision.RunningMode.VIDEO)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW); cap.set(3,640); cap.set(4,480); time.sleep(1.0)
    hist = collections.deque(maxlen=5)
    stable = "HOLD"
    last_sent = ""
    m = {"FIST":"GRIP","OPEN_PALM":"STOP","PEACE":"RELEASE","POINT":"MOVE",
         "THUMB_UP":"UP","THUMB_DOWN":"DOWN","PINKY":"DOWN","THREE":"UP","ONE":"LEFT","TWO":"RIGHT"}
    with vision.HandLandmarker.create_from_options(opts) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame,1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb), int(time.time()*1000))
            raw="NO_HAND"; conf=0.0
            if res.hand_landmarks:
                conf=float(res.handedness[0][0].score) if res.handedness else 0.0
                g,c,_,_=gc.classify(res.hand_landmarks[0],
                    res.handedness[0][0].category_name if res.handedness else "Right")
                raw=g if conf>=0.5 else "LOW_CONF"
                for j,hl in enumerate(res.hand_landmarks):
                    gc.draw_hand(frame, hl, [(0,255,0),(255,0,0)][j%2])
                if len(res.hand_landmarks)==2:
                    try:
                        _,c2,_,_=gc.classify(res.hand_landmarks[1],"Left")
                        if c=="STOP" and c2=="STOP":
                            ser.write(b"E:1\n"); stable="ESTOP"
                    except Exception: pass
            hist.append(raw)
            if len(hist)==5:
                most,votes=collections.Counter(hist).most_common(1)[0]
                if votes>=4 and most not in ("HOLD","NO_HAND","LOW_CONF","UNKNOWN"):
                    if most!=stable:
                        stable=most; cmd=m.get(most,"HOLD")
                        if cmd!=last_sent and cmd not in ("HOLD",):
                            ser.write((cmd+"\n").encode()); last_sent=cmd
                            print(f"SENT {cmd} ({most})")
            cv2.putText(frame,f"stable {stable} last {last_sent}",(10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            cv2.imshow("RoboGrip Serial - e=estop r=reset q=quit",frame)
            k=cv2.waitKey(1)&0xFF
            if k==ord('q'): break
            elif k==ord('e'): ser.write(b"E:1\n"); stable="ESTOP"
            elif k==ord('r'): ser.write(b"E:0\n"); stable="HOLD"; hist.clear()
    cap.release(); cv2.destroyAllWindows(); ser.close()

if __name__=="__main__":
    import argparse; ap=argparse.ArgumentParser(); ap.add_argument("--port",default="COM3")
    main(ap.parse_args().port)

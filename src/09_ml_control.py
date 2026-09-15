"""
RoboGrip Step 8 (software-only): Live ML control.
Uses models/gesture_rf.pkl instead of rules. Same temporal + safety as Step 4.
Run: venv\\Scripts\\python.exe src\\09_ml_control.py
"""
import cv2, time, os, pickle, collections, math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE, "models", "hand_landmarker.task")
RF_PATH = os.path.join(BASE, "models", "gesture_rf.pkl")
CMD = {"FIST":"GRIP","OPEN_PALM":"STOP","PEACE":"RELEASE","POINT":"MOVE","THUMB_UP":"UP","THUMB_DOWN":"DOWN"}

def feat(lms):
    wx,wy=lms[0].x,lms[0].y
    s=max(math.hypot(lms[9].x-wx,lms[9].y-wy),1e-6)
    f=[]
    for lm in lms:
        f+=[(lm.x-wx)/s,(lm.y-wy)/s,(lm.z-wx)/s if hasattr(lm,'z') else 0.0]
    return [f]

def main():
    import numpy as np
    clf=pickle.load(open(RF_PATH,"rb"))
    print(f"Loaded {RF_PATH} classes {clf.classes_}")
    base=python.BaseOptions(model_asset_path=MODEL_PATH)
    opts=vision.HandLandmarkerOptions(base_options=base,num_hands=1,
        min_hand_detection_confidence=0.5,running_mode=vision.RunningMode.VIDEO)
    cap=cv2.VideoCapture(0,cv2.CAP_DSHOW);cap.set(3,640);cap.set(4,480);time.sleep(1.0)
    hist=collections.deque(maxlen=5)
    stable="HOLD"
    with vision.HandLandmarker.create_from_options(opts) as landmarker:
        while True:
            ret,frame=cap.read()
            if not ret: break
            frame=cv2.flip(frame,1)
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            res=landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb),int(time.time()*1000))
            raw="NO_HAND"; prob=0.0
            if res.hand_landmarks:
                p=clf.predict_proba(feat(res.hand_landmarks[0]))[0]
                i=p.argmax(); raw=str(clf.classes_[i]); prob=float(p[i])
                if prob<0.6: raw="LOW_CONF"
            hist.append(raw)
            if len(hist)==5:
                most,votes=collections.Counter(hist).most_common(1)[0]
                if votes>=4 and most not in ("HOLD","NO_HAND","LOW_CONF","UNKNOWN"):
                    stable=most
            cmd=CMD.get(stable,"HOLD")
            cv2.putText(frame,f"ML {raw} {prob:.2f} stable {stable}->{cmd}",(10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,255,0),2)
            cv2.imshow("RoboGrip ML control q=quit",frame)
            if cv2.waitKey(1)&0xFF==ord('q'): break
    cap.release();cv2.destroyAllWindows()

if __name__=="__main__": main()

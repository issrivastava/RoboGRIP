"""
RoboGrip Debug v2: separates FIST vs THUMB-UP.
Press 's' to log. Make FIST, then THUMB-UP, then OPEN.
"""
import cv2, time, os, math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "hand_landmarker.task")

def dist(lms,a,b):
    return math.hypot(lms[a].x-lms[b].x, lms[a].y-lms[b].y)

base = python.BaseOptions(model_asset_path=MODEL_PATH)
opts = vision.HandLandmarkerOptions(base_options=base, num_hands=1,
    min_hand_detection_confidence=0.5, running_mode=vision.RunningMode.VIDEO)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3,640); cap.set(4,480)
time.sleep(1.0)
with vision.HandLandmarker.create_from_options(opts) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame=cv2.flip(frame,1)
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        img=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
        res=landmarker.detect_for_video(img,int(time.time()*1000))
        if res.hand_landmarks:
            lms=res.hand_landmarks[0]
            hs=max(dist(lms,0,9),1e-6)
            dx=lms[4].x-lms[3].x
            dy=lms[4].y-lms[3].y
            spread=abs(dx)/hs
            up=-dy/hs
            d45=dist(lms,4,5)/hs  # thumb tip to index-base: FIST=close, THUMB-UP=far
            d46=dist(lms,4,6)/hs
            d49=dist(lms,4,9)/hs  # thumb tip to middle-base
            txt=f"spread {spread:.2f} up {up:.2f} d45 {d45:.2f} d46 {d46:.2f}"
            cv2.putText(frame,txt,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,0),2)
            cv2.putText(frame,f"d49 {d49:.2f} hs {hs:.3f}",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)
            h,w,_=frame.shape
            for idx,col in [(3,(255,0,0)),(4,(0,0,255)),(5,(0,255,255))]:
                cv2.circle(frame,(int(lms[idx].x*w),int(lms[idx].y*h)),8,col,-1)
            key=cv2.waitKey(1)&0xFF
            if key==ord('s'):
                line=f"spread={spread:.3f} up={up:.3f} d45={d45:.3f} d46={d46:.3f} d49={d49:.3f} hs={hs:.4f}"
                print(line)
                with open("debug_log.txt","a") as f: f.write(line+"\n")
            if key==ord('q'): break
        else:
            cv2.putText(frame,"Show ONE hand",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
            if cv2.waitKey(1)&0xFF==ord('q'): break
        cv2.imshow("Thumb Debug v2 - s=log q=quit",frame)
cap.release(); cv2.destroyAllWindows()

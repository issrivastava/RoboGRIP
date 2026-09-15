"""
RoboGrip Task Demo (software-only): timed pick-and-place for report.
Move wrist to cube + GRIP to pick, to green target + RELEASE to place.
Metrics: task completion time, success rate, commands count, BLOCKED hits.

Controls (hold 0.4s to fire, same as Step 4):
 FIST=GRIP PEACE=RELEASE OPEN=STOP THUMB_UP=UP THREE=UP THUMB_DOWN/PINKY=DOWN
 ONE=LEFT TWO=RIGHT POINT=nudge. e=E-STOP r=reset n=new cube q=quit.

Run:
  venv\\Scripts\\python.exe src\\10_task_demo.py
Result: task_results.csv
"""
import cv2, time, os, math, random, csv, collections
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import importlib.util
_spec = importlib.util.spec_from_file_location("gc",
    os.path.join(os.path.dirname(__file__), "03_gesture_classifier.py"))
gc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(gc)

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "hand_landmarker.task")
HOLD, CONF = 5, 0.5
A1_MIN,A1_MAX,A2_MIN,A2_MAX = 20,160,10,150
M = {"FIST":"GRIP","OPEN_PALM":"STOP","PEACE":"RELEASE","POINT":"MOVE",
     "THUMB_UP":"UP","THUMB_DOWN":"DOWN","PINKY":"DOWN","THREE":"UP","ONE":"LEFT","TWO":"RIGHT"}

def fk(a1d,a2d,bx,by,L1=120,L2=100):
    a1=math.radians(a1d)
    x1=bx+int(L1*math.cos(a1)); y1=by-int(L1*math.sin(a1))
    x2=x1+int(L2*math.cos(a1-math.radians(a2d))); y2=y1-int(L2*math.sin(a1-math.radians(a2d)))
    return (x1,y1),(x2,y2)

def main():
    import numpy as np
    base=python.BaseOptions(model_asset_path=MODEL_PATH)
    opts=vision.HandLandmarkerOptions(base_options=base,num_hands=2,
        min_hand_detection_confidence=0.5,running_mode=vision.RunningMode.VIDEO)
    cap=cv2.VideoCapture(0,cv2.CAP_DSHOW);cap.set(3,640);cap.set(4,480);time.sleep(1.0)
    hist=collections.deque(maxlen=HOLD)
    stable="HOLD"; a1,a2=90,60; grip=False; estop=False
    W,H=400,480; bx,by=W//2,H-40
    # Easy start: cube near initial wrist (~287,270), target nearby
    cube=[250,280]; target=[320,260]; holding=False
    t0=time.time(); cmds=0; blocked=0; trials=[]; trial_no=1
    print("Task: wrist->cube + GRIP, ->target + RELEASE. n=new cube.")
    with vision.HandLandmarker.create_from_options(opts) as landmarker:
        while True:
            ret,frame=cap.read()
            if not ret: break
            frame=cv2.flip(frame,1)
            res=landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),int(time.time()*1000))
            raw="NO_HAND"; conf=0.0
            if res.hand_landmarks:
                conf=float(res.handedness[0][0].score) if res.handedness else 0.0
                g,c,_,_=gc.classify(res.hand_landmarks[0],
                    res.handedness[0][0].category_name if res.handedness else "Right")
                raw=g if conf>=CONF else "LOW_CONF"
                for j,hl in enumerate(res.hand_landmarks):
                    gc.draw_hand(frame,hl,[(0,255,0),(255,0,0)][j%2])
                if len(res.hand_landmarks)==2:
                    try:
                        _,c2,_,_=gc.classify(res.hand_landmarks[1],"Left")
                        if c=="STOP" and c2=="STOP": estop=True
                    except Exception: pass
            hist.append(raw)
            if len(hist)==HOLD:
                most,votes=collections.Counter(hist).most_common(1)[0]
                if votes>=4 and most not in ("HOLD","NO_HAND","LOW_CONF","UNKNOWN") and most!=stable:
                    stable=most; cmds+=1
            cmd=M.get(stable,"HOLD")
            if not estop:
                st=4.0  # faster for task (was 2.5)
                na1,na2=a1,a2
                if cmd=="UP": na1=min(A1_MAX,a1+st)
                elif cmd=="DOWN": na1=max(A1_MIN,a1-st)
                elif cmd=="LEFT": na2=max(A2_MIN,a2-st)
                elif cmd=="RIGHT": na2=min(A2_MAX,a2+st)
                elif cmd=="MOVE": na1=min(A1_MAX,a1+1.5); na2=min(A2_MAX,a2+1.5)
                elif cmd=="GRIP": grip=True
                elif cmd=="RELEASE": grip=False
                if (na1,na2)==(a1,a2) and cmd in ("UP","DOWN","LEFT","RIGHT"): blocked+=1
                a1,a2=na1,na2
            (x1,y1),(x2,y2)=fk(a1,a2,bx,by)
            # pick/place logic - EASY radii (was 30/35)
            if not holding and grip and math.hypot(x2-cube[0],y2-cube[1])<45:
                holding=True
            if holding:
                cube=[x2,y2]
                if (not grip) and math.hypot(x2-target[0],y2-target[1])<50:
                    dt=time.time()-t0
                    trials.append({"trial":trial_no,"time":round(dt,1),"cmds":cmds,"blocked":blocked,"success":True})
                    print(f"Trial {trial_no} DONE {dt:.1f}s cmds {cmds} blocked {blocked}")
                    trial_no+=1; holding=False; grip=False
                    # next cubes spawn central reachable (not far corners)
                    cube=[random.randint(180,280),random.randint(230,320)]
                    target=[random.randint(200,320),random.randint(230,320)]
                    t0=time.time(); cmds=0; blocked=0
            # draw panel
            panel=np.zeros((H,W,3),dtype=np.uint8); panel[:]=(30,30,30)
            cv2.rectangle(panel,(target[0]-30,target[1]-30),(target[0]+30,target[1]+30),(0,255,0),2)
            cv2.putText(panel,"TARGET",(target[0]-30,target[1]-35),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
            col=(0,165,255) if not holding else (255,0,255)
            cv2.rectangle(panel,(cube[0]-12,cube[1]-12),(cube[0]+12,cube[1]+12),col,-1)
            c2=(0,0,255) if estop else (0,255,0)
            cv2.circle(panel,(bx,by),10,(200,200,200),-1)
            cv2.line(panel,(bx,by),(x1,y1),c2,7); cv2.circle(panel,(x1,y1),7,(255,255,0),-1)
            cv2.line(panel,(x1,y1),(x2,y2),c2,5)
            cv2.circle(panel,(x2,y2),6,(255,255,255),-1)
            d_cube=math.hypot(x2-cube[0],y2-cube[1])
            d_targ=math.hypot(x2-target[0],y2-target[1])
            cv2.putText(panel,f"T{trial_no} {time.time()-t0:.1f}s {stable}->{cmd}",(10,25),
                        cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1)
            cv2.putText(panel,f"holding {holding} grip {'C' if grip else 'O'}", (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1)
            if not holding:
                cv2.putText(panel,f"cube {d_cube:.0f}px need<45+GRIP",(10,75),
                            cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,255) if d_cube<45 else (200,200,200),1)
            else:
                cv2.putText(panel,f"targ {d_targ:.0f}px need<50+RELEASE",(10,75),
                            cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,255) if d_targ<50 else (200,200,200),1)
            cv2.putText(panel,"keys:arrows move g=grip h=release",(10,100),
                        cv2.FONT_HERSHEY_SIMPLEX,0.45,(150,150,150),1)
            if estop: cv2.putText(panel,"E-STOP r=reset",(10,125),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
            cv2.putText(frame,f"raw {raw} hold {sum(1 for x in hist if x==raw)}/{HOLD} stable {stable}->{cmd}",(10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,0),2)
            combo=np.hstack([frame,cv2.resize(panel,(400,480))])
            cv2.imshow("RoboGrip Task CLICK-HERE then g/h/wasd",combo)
            k=cv2.waitKey(15)&0xFF  # longer window to catch keys
            if k!=255 and k!=0:
                print(f"key {k} chr {chr(k) if 32<=k<127 else '?'}")
            if k==ord('q'): break
            elif k==ord('e'): estop=True
            elif k==ord('r'): estop=False; hist.clear(); stable="HOLD"
            # Keyboard fallback: clears gesture history so gestures don't instantly override
            elif k==ord('g'): grip=True; stable="KEY_GRIP"; hist.clear(); cmds+=1; print("KEY GRIP -> should stick if <45px")
            elif k==ord('h'): grip=False; stable="KEY_RELEASE"; hist.clear(); cmds+=1; print("KEY RELEASE")
            elif k==ord('w'): a1=min(A1_MAX,a1+4); print(f"KEY UP a1 {a1}")
            elif k==ord('s'): a1=max(A1_MIN,a1-4); print(f"KEY DOWN a1 {a1}")
            elif k==ord('a'): a2=max(A2_MIN,a2-4); print(f"KEY LEFT a2 {a2}")
            elif k==ord('d'): a2=min(A2_MAX,a2+4); print(f"KEY RIGHT a2 {a2}")
            elif k==ord('n'):
                cube=[random.randint(180,280),random.randint(230,320)]; holding=False
                t0=time.time(); cmds=0; blocked=0
    cap.release(); cv2.destroyAllWindows()
    if trials:
        import csv as _c
        with open("task_results.csv","w",newline="") as f:
            w=_c.DictWriter(f,fieldnames=["trial","time","cmds","blocked","success"]); w.writeheader(); w.writerows(trials)
        avg=sum(t["time"] for t in trials)/len(trials)
        print(f"\nSaved task_results.csv: {len(trials)} trials avg {avg:.1f}s success {len(trials)}/{trial_no-1}")
    else:
        print("No completed trials yet - that's ok, practice run.")

if __name__=="__main__": main()

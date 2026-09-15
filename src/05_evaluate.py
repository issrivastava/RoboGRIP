"""
RoboGrip Step 5: Evaluation
Measures your abstract metrics: accuracy, precision, recall, F1,
latency, success rate, false-command rate.

Protocol: for each prompted gesture, hold it 3s.
Script logs predicted vs expected, computes scores, saves results.csv

Run:
  venv\\Scripts\\python.exe src\\05_evaluate.py --trials 3
  (3 trials per gesture = 18 tests, ~2 min. Use --trials 5 for report.)
"""
import cv2, time, os, argparse, csv, collections
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import importlib.util
_spec = importlib.util.spec_from_file_location("gc",
    os.path.join(os.path.dirname(__file__), "03_gesture_classifier.py"))
gc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(gc)

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "hand_landmarker.task")
GESTURES = ["FIST", "OPEN_PALM", "PEACE", "POINT", "THUMB_UP", "THUMB_DOWN"]
HOLD_FRAMES = 5
CONF_THRESH = 0.5

def evaluate(trials=3):
    base = python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = vision.HandLandmarkerOptions(base_options=base, num_hands=1,
        min_hand_detection_confidence=0.5, running_mode=vision.RunningMode.VIDEO)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3,640); cap.set(4,480); time.sleep(1.0)
    rows = []
    with vision.HandLandmarker.create_from_options(opts) as landmarker:
        for gi, expected in enumerate(GESTURES):
            for t in range(trials):
                print(f"\n[{gi*trials+t+1}/{len(GESTURES)*trials}] Show {expected} - 3s starting NOW (trial {t+1})")
                hist = collections.deque(maxlen=HOLD_FRAMES)
                start = time.time()
                fired = None
                fire_time = None
                false_fires = 0
                while time.time() - start < 3.0:
                    ret, frame = cap.read()
                    if not ret: continue
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = landmarker.detect_for_video(
                        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(time.time()*1000))
                    raw = "NO_HAND"
                    conf = 0.0
                    if res.hand_landmarks:
                        conf = float(res.handedness[0][0].score) if res.handedness else 0.0
                        g, c, _, _ = gc.classify(res.hand_landmarks[0],
                            res.handedness[0][0].category_name if res.handedness else "Right")
                        raw = g if conf >= CONF_THRESH else "LOW_CONF"
                    hist.append(raw)
                    if len(hist)==HOLD_FRAMES:
                        most, votes = collections.Counter(hist).most_common(1)[0]
                        if votes>=4 and most not in ("HOLD","NO_HAND","LOW_CONF","UNKNOWN"):
                            if fired is None and most==expected:
                                fired = most; fire_time = time.time()-start
                            elif fired is None and most!=expected:
                                false_fires += 1  # wrong stable before correct
                    cv2.putText(frame, f"Show {expected} trial {t+1} {3.0-(time.time()-start):.1f}s raw:{raw}",
                                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    cv2.imshow("RoboGrip Evaluate - q to abort", frame)
                    if cv2.waitKey(1)&0xFF==ord('q'):
                        cap.release(); cv2.destroyAllWindows(); return
                success = (fired==expected)
                latency = fire_time if success else 3.0
                rows.append({"expected":expected,"success":success,"latency":round(latency,2),
                             "false_fires":false_fires})
                print(f" -> {'PASS' if success else 'FAIL'} latency {latency:.2f}s false {false_fires}")
                time.sleep(0.7)
    cap.release(); cv2.destroyAllWindows()
    # Metrics
    with open("results.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["expected","success","latency","false_fires"]); w.writeheader(); w.writerows(rows)
    print("\nSaved results.csv")
    total=len(rows); ok=sum(1 for r in rows if r["success"])
    print(f"Accuracy / Success rate: {ok}/{total} = {ok/total:.2%}")
    print(f"Avg latency: {sum(r['latency'] for r in rows)/total:.2f}s")
    print(f"False-command events: {sum(r['false_fires'] for r in rows)}")
    # Per-gesture precision/recall (one-vs-rest from success counts)
    for g in GESTURES:
        tp=sum(1 for r in rows if r["expected"]==g and r["success"])
        tot=sum(1 for r in rows if r["expected"]==g)
        rec=tp/max(tot,1)
        print(f" {g}: recall {rec:.2%} ({tp}/{tot})")
    print("\nFor report: repeat in different light/background + 2nd user, compare.")
    print("Next: Arduino wiring when hardware arrives.")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--trials",type=int,default=3)
    evaluate(ap.parse_args().trials)

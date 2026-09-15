"""
RoboGrip Step 6 (software-only): Collect ML dataset.
Captures normalized 63-D landmarks per gesture for sklearn training.
Needed to honestly claim 'ML-based classification' in report.

Controls: keys 0-5 to select gesture, SPACE to capture 1 sample, q quit.
  0=FIST 1=OPEN_PALM 2=PEACE 3=POINT 4=THUMB_UP 5=THUMB_DOWN
Aim 80-100 samples per gesture, varied distance/light/angle.

Run:
  venv\\Scripts\\python.exe src\\07_collect_dataset.py
Output: data/dataset.csv
"""
import cv2, time, os, csv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "hand_landmarker.task")
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "dataset.csv")
LABELS = ["FIST","OPEN_PALM","PEACE","POINT","THUMB_UP","THUMB_DOWN"]

def normalize(lms):
    # translate to wrist(0), scale by wrist->middle_mcp(9)
    import math
    wx, wy, wz = lms[0].x, lms[0].y, lms[0].z
    s = math.hypot(lms[9].x-wx, lms[9].y-wy)
    s = max(s, 1e-6)
    feat = []
    for lm in lms:
        feat += [(lm.x-wx)/s, (lm.y-wy)/s, (lm.z-wz)/s if hasattr(lm,'z') else 0.0]
    return feat  # 63 values

def main():
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    new = not os.path.exists(DATA_PATH)
    f = open(DATA_PATH, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["label"] + [f"f{i}" for i in range(63)])
    base = python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = vision.HandLandmarkerOptions(base_options=base, num_hands=1,
        min_hand_detection_confidence=0.5, running_mode=vision.RunningMode.VIDEO)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW); cap.set(3,640); cap.set(4,480); time.sleep(1.0)
    cur = 0
    counts = {l:0 for l in LABELS}
    # count existing
    try:
        import csv as _c
        with open(DATA_PATH) as rf:
            for r in list(_c.DictReader(rf)):
                if r["label"] in counts: counts[r["label"]]+=1
    except Exception: pass
    print("Keys 0-5 select gesture, SPACE capture, q quit.")
    with vision.HandLandmarker.create_from_options(opts) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame,1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb), int(time.time()*1000))
            has = bool(res.hand_landmarks)
            cv2.putText(frame, f"Target {cur}={LABELS[cur]} counts {counts}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
            cv2.putText(frame, "0-5 select SPACE capture" + (" HAND OK" if has else " NO HAND"),
                        (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0) if has else (0,0,255), 2)
            cv2.imshow("Collect - space=capture", frame)
            k = cv2.waitKey(1)&0xFF
            if k==ord('q'): break
            elif ord('0')<=k<=ord('5'): cur=k-ord('0')
            elif k==32 and has:  # space
                feat = normalize(res.hand_landmarks[0])
                w.writerow([LABELS[cur]]+feat); f.flush()
                counts[LABELS[cur]]+=1
                print(f"Saved {LABELS[cur]} #{counts[LABELS[cur]]}")
    f.close(); cap.release(); cv2.destroyAllWindows()
    print(f"Done -> {DATA_PATH} {counts}")

if __name__=="__main__":
    main()

"""
RoboGrip Step 7: Train ML classifier, compare vs rule-based.
Reads data/dataset.csv, trains RandomForest, prints accuracy/precision/recall/F1,
saves models/gesture_rf.pkl + reports confusion matrix.

Install: venv\\Scripts\\python.exe -m pip install scikit-learn pandas matplotlib
Run: venv\\Scripts\\python.exe src\\08_train_ml.py
"""
import os, pickle
BASE = os.path.dirname(os.path.dirname(__file__))
CSV = os.path.join(BASE, "data", "dataset.csv")
OUT = os.path.join(BASE, "models", "gesture_rf.pkl")

def main():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    df = pd.read_csv(CSV)
    print(f"Loaded {len(df)} samples: {df['label'].value_counts().to_dict()}")
    if len(df) < 60:
        print("Collect more: need 80+ per gesture ideally, min 60 total to train.")
        return
    X = df.drop(columns=["label"]).values
    y = df["label"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    print(f"\nAccuracy: {accuracy_score(yte,pred):.2%}")
    print(classification_report(yte, pred, digits=3))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(yte, pred, labels=sorted(yte.tolist().__class__ and sorted(set(y)))))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT,"wb") as f: pickle.dump(clf,f)
    print(f"\nSaved {OUT}. Use in 09_ml_control.py. Copy these numbers to report.")
    # Try plot
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
        disp = ConfusionMatrixDisplay.from_predictions(yte, pred, xticks_rotation=45)
        plt.tight_layout(); plt.savefig(os.path.join(BASE,"models","confusion.png"), dpi=150)
        print("Saved models/confusion.png for report.")
    except Exception as e:
        print(f"Plot skip: {e}")

if __name__=="__main__":
    main()

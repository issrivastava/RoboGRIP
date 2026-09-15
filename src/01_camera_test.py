"""
RoboGrip Step 1: Camera Test
Tests if your webcam works with OpenCV.
Run: venv\Scripts\python.exe src\01_camera_test.py
Press 'q' to quit.
"""
import cv2

def main():
    print("Opening webcam (index 0)...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        print("Try changing index to 1 in cv2.VideoCapture(1)")
        return

    print("Webcam OK! Press 'q' in the video window to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to grab frame")
            break

        # Mirror for natural interaction
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, "RoboGrip Camera OK - press 'q' to quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("RoboGrip - Camera Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera test done.")

if __name__ == "__main__":
    main()

import cv2
import time

print("Starting camera probe...")
for i in range(6):
    print(f"Trying index {i} ...", flush=True)
    cap = cv2.VideoCapture(i)
    # small wait to let device initialize
    time.sleep(0.5)
    opened = cap.isOpened()
    print(f" -> opened = {opened}", flush=True)
    if opened:
        print(f"Camera found at index {i}", flush=True)
        cap.release()
    else:
        # defensive release if needed
        try:
            cap.release()
        except:
            pass

print("Probe finished.")

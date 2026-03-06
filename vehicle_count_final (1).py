import cv2, math, os, time
import numpy as np
from collections import deque
from datetime import datetime
from ultralytics import YOLO

# -------- CONFIG --------
CAM_INDEX = 0
MODEL_PATH = "yolov8n.pt"
CONF_THRESH = 0.40
MIN_AREA = 600
DIST_THRESH = 70
LINE_POSITION = 0.60
ENTRY_POS = 0.45
MAX_DISAPPEARED = 18
TRAJ_LEN = 25
MIN_SEEN_FRAMES = 3
VEHICLE_CLASSES = {"car","truck","bus","motorbike","bicycle"}
IOU_MERGE_THRESH = 0.5       # MERGE detections if IoU > 0.5
# ------------------------

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit(f"Cannot open camera {CAM_INDEX}")

next_id = 0
tracks = {}
total_count = 0
frame_idx = 0

def centroid(b):
    x1,y1,x2,y2 = b
    return (int((x1+x2)/2), int((y1+y2)/2))

def iou(a, b):
    xA = max(a[0],b[0]); yA=max(a[1],b[1])
    xB = min(a[2],b[2]); yB=min(a[3],b[3])
    inter_w = xB-xA; inter_h = yB-yA
    if inter_w<=0 or inter_h<=0: return 0
    inter = inter_w*inter_h
    areaA=(a[2]-a[0])*(a[3]-a[1])
    areaB=(b[2]-b[0])*(b[3]-b[1])
    return inter/(areaA+areaB-inter)

def merge_duplicate_boxes(dets):
    """ Remove overlapping duplicate detections (toy car fix) """
    if len(dets) <= 1:
        return dets
    keep = []
    used = set()

    for i in range(len(dets)):
        if i in used:
            continue
        boxA = dets[i]
        merged = boxA
        for j in range(i+1, len(dets)):
            if j in used:
                continue
            boxB = dets[j]
            if iou(boxA, boxB) > IOU_MERGE_THRESH:
                # merge by taking the bounding rectangle around both
                x1 = min(merged[0], boxB[0])
                y1 = min(merged[1], boxB[1])
                x2 = max(merged[2], boxB[2])
                y2 = max(merged[3], boxB[3])
                merged = (x1,y1,x2,y2)
                used.add(j)
        keep.append(merged)
        used.add(i)

    return keep

print("Running... press Q to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_idx += 1
    h,w = frame.shape[:2]
    line_y = int(h*LINE_POSITION)
    entry_y = int(h*ENTRY_POS)

    # --- YOLO detection ---
    r = model.predict(frame, conf=CONF_THRESH, verbose=False)[0]

    detections = []
    if r.boxes is not None:
        boxes = r.boxes.xyxy.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)

        for (x1,y1,x2,y2), ci in zip(boxes, clss):
            cls_name = model.names[int(ci)]
            if cls_name not in VEHICLE_CLASSES:
                continue
            area = (x2-x1) * (y2-y1)
            if area < MIN_AREA:
                continue
            detections.append((int(x1),int(y1),int(x2),int(y2)))

    # --- REMOVE DUPLICATES ---
    detections = merge_duplicate_boxes(detections)

    # --- TRACKING ---
    if len(tracks) == 0:
        for box in detections:
            tracks[next_id] = {
                "bbox": box,
                "centroid": centroid(box),
                "trajectory": deque([centroid(box)], maxlen=TRAJ_LEN),
                "seen": 1,
                "dis": 0,
                "counted": False
            }
            next_id += 1
    else:
        # match detections to existing tracks
        track_ids = list(tracks.keys())
        det_centroids = [centroid(b) for b in detections]

        used_det = set()
        for tid in track_ids:
            tx,ty = tracks[tid]["centroid"]
            best = None
            best_dist = 99999

            for i,(cx,cy) in enumerate(det_centroids):
                if i in used_det:
                    continue
                d = math.hypot(tx-cx, ty-cy)
                if d < best_dist:
                    best_dist = d
                    best = i

            if best is not None and best_dist < DIST_THRESH:
                box = detections[best]
                tracks[tid]["bbox"] = box
                tracks[tid]["centroid"] = det_centroids[best]
                tracks[tid]["trajectory"].append(det_centroids[best])
                tracks[tid]["seen"] += 1
                tracks[tid]["dis"] = 0
                used_det.add(best)
            else:
                tracks[tid]["dis"] += 1

        # add new tracks
        for i, box in enumerate(detections):
            if i not in used_det:
                tracks[next_id] = {
                    "bbox": box,
                    "centroid": centroid(box),
                    "trajectory": deque([centroid(box)], maxlen=TRAJ_LEN),
                    "seen": 1,
                    "dis": 0,
                    "counted": False
                }
                next_id += 1

        # remove lost
        for tid in list(tracks.keys()):
            if tracks[tid]["dis"] > MAX_DISAPPEARED:
                del tracks[tid]

    # --- COUNTING ---
    for tid, t in tracks.items():
        if t["counted"]:
            continue
        if t["seen"] < MIN_SEEN_FRAMES:
            continue

        # early count
        if t["centroid"][1] >= entry_y:
            total_count += 1
            t["counted"] = True
            continue

        # main crossing
        traj = t["trajectory"]
        if len(traj) >= 2:
            if traj[-2][1] < line_y <= traj[-1][1]:
                total_count += 1
                t["counted"] = True

    # --- VISUALS ---
    for tid, t in tracks.items():
        x1,y1,x2,y2 = t["bbox"]
        col = (0,255,0) if t["counted"] else (0,255,255)
        cv2.rectangle(frame,(x1,y1),(x2,y2),col,2)
        cv2.putText(frame, f"ID{tid}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

    cv2.line(frame,(0,entry_y),(w,entry_y),(255,100,100),1)
    cv2.line(frame,(0,line_y),(w,line_y),(0,200,255),2)
    cv2.putText(frame,f"Total Vehicles: {total_count}",(10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

    cv2.imshow("Vehicle Count (final)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Final Vehicle Count:", total_count)
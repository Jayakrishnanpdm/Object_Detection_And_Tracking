# Detected if person picking an object
from ultralytics import YOLO
import cv2
from collections import defaultdict
import numpy as np
from alert import send_email_alert, send_telegram_alert


model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)  # Change to video file if needed

active_objects = {}  # Track objects
person_last_seen = {}  # Track last seen positions of persons

def get_detections(results):
    detections = {}
    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            name = model.names[cls_id]
            track_id = int(box.id[0]) if box.id is not None else None
            bbox = box.xyxy[0].cpu().numpy()

            if track_id is not None:
                detections[track_id] = {"name": name, "bbox": bbox}
    return detections

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    xB = min(boxA[2], boxB[2])
    yA = max(boxA[1], boxB[1])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea

    return interArea / unionArea if unionArea > 0 else 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, conf=0.35)
    detections = get_detections(results)

    # Update tracking
    for track_id, obj in detections.items():
        if obj["name"] == "person":
            person_last_seen[track_id] = obj["bbox"]

        if track_id not in active_objects:
            active_objects[track_id] = {"name": obj["name"], "bbox": obj["bbox"], "missing_frames": 0}
        else:
            active_objects[track_id]["bbox"] = obj["bbox"]
            active_objects[track_id]["missing_frames"] = 0

    # Check missing objects
    for track_id in list(active_objects.keys()):
        if track_id not in detections:
            active_objects[track_id]["missing_frames"] += 1

            if active_objects[track_id]["missing_frames"] == 30:
                # Check if any person was near this object before disappearance
                for pid, pbbox in person_last_seen.items():
                    if iou(active_objects[track_id]["bbox"], pbbox) > 0.1:
                        alert_msg = f"⚠ Suspicious: {active_objects[track_id]['name']} picked by Person {pid}"
                        image_path = "alert.jpg"
                        cv2.imwrite(image_path, frame)
                        send_email_alert(image_path, alert_msg)
                        send_telegram_alert(image_path, alert_msg)
                        print("Alert sent:", alert_msg)
                        break


                del active_objects[track_id]

    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Theft Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

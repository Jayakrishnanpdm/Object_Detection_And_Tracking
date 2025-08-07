from ultralytics import YOLO
import cv2
import numpy as np
from alert import send_email_alert, send_telegram_alert
from deep_sort_realtime.deepsort_tracker import DeepSort

def draw_alert_boxes(frame, person_bbox, object_bbox, person_id):
    p_x1, p_y1, p_x2, p_y2 = map(int, person_bbox)
    o_x1, o_y1, o_x2, o_y2 = map(int, object_bbox)

    cv2.rectangle(frame, (p_x1, p_y1), (p_x2, p_y2), (255, 0, 0), 2)
    cv2.putText(frame, f"Person {person_id}", (p_x1, p_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.rectangle(frame, (o_x1, o_y1), (o_x2, o_y2), (0, 0, 255), 2)
    cv2.putText(frame, "Object", (o_x1, o_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame

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

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

tracker = DeepSort(max_age=30, n_init=3)
active_objects = {}
person_last_seen = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections, detection_names = [], []

    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            name = model.names[cls_id]
            bbox = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy()) 
            if conf > 0.5:
                # Convert [x1, y1, x2, y2] -> [x, y, w, h] for DeepSORT
                detections.append((
                    [float(bbox[0]), 
                    float(bbox[1]), 
                    float(bbox[2] - bbox[0]), 
                    float(bbox[3] - bbox[1])],
                    float(conf)
                ))
                detection_names.append(name)

    # DeepSORT expects [[x, y, w, h, conf], ...]
    tracks = tracker.update_tracks(detections, frame=frame)

    detections_dict = {}
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        name = None
        # Associate class name using IoU with detections
        for i, det in enumerate(detections):
            x, y, w, h = det[0]
            det_box = [x, y, x + w, y + h]

            if iou(ltrb, det_box) > 0.7:
                name = detection_names[i]
                break
        if name is not None:
            detections_dict[track_id] = {"name": name, "bbox": ltrb}

    # Track persons and objects
    for track_id, obj in detections_dict.items():
        if obj["name"] == "person":
            person_last_seen[track_id] = obj["bbox"]
        else:
            if track_id not in active_objects:
                active_objects[track_id] = {"name": obj["name"], "bbox": obj["bbox"], "missing_frames": 0}
            else:
                active_objects[track_id]["bbox"] = obj["bbox"]
                active_objects[track_id]["missing_frames"] = 0

    # Check for missing objects
    for track_id in list(active_objects.keys()):
        if active_objects[track_id]["name"] != "person":
            if track_id not in detections_dict:
                active_objects[track_id]["missing_frames"] += 1
                if active_objects[track_id]["missing_frames"] == 120:
                    for pid, pbbox in person_last_seen.items():
                        if iou(active_objects[track_id]["bbox"], pbbox) > 0.1:
                            alert_msg = f"⚠ Suspicious: {active_objects[track_id]['name']} picked by Person {pid}"
                            image_path = "alert.jpg"
                            alert_frame = draw_alert_boxes(frame.copy(), pbbox, active_objects[track_id]["bbox"], pid)
                            cv2.imwrite(image_path, alert_frame)
                            send_email_alert(image_path, alert_msg)
                            send_telegram_alert(image_path, alert_msg)
                            print("Alert sent:", alert_msg)
                            break
                    del active_objects[track_id]

    # Annotate
    annotated_frame = frame.copy()
    for tid, obj in detections_dict.items():
        box = obj["bbox"]
        x1, y1, x2, y2 = map(int, box)
        label = f"{obj['name']} ID:{tid}"
        color = (0, 255, 0) if obj["name"] == "person" else (0, 0, 255)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("YOLO + DeepSORT Theft Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

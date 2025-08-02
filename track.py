from ultralytics import YOLO
import cv2

# Load YOLO model (Nano version for speed)
model = YOLO("yolov8n.pt")  

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection + tracking
    results = model.track(frame, persist=True, conf=0.5)

    # Draw bounding boxes + IDs
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Tracking", annotated_frame)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

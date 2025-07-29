from ultralytics import YOLO

model = YOLO("yolov8s.pt")

# Use webcam (0) or video file path
model.predict(source=0, show=True, conf=0.5)
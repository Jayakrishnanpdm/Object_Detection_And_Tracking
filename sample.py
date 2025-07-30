import time
import cv2
prev_time = time.time()
frame_count = 0
cap = cv2.VideoCapture(0)  
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection code here...

    frame_count += 1
    if frame_count >= 30:  # Calculate every 30 frames
        curr_time = time.time()
        fps = frame_count / (curr_time - prev_time)
        print(f"FPS: {fps:.2f}")
        prev_time = curr_time
        frame_count = 0

seconds = 10 / fps
print(f"Object missing for {seconds:.2f} seconds if 10 frames are missed.")

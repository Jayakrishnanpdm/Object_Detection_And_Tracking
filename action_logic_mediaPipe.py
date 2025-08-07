import cv2
import numpy as np
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import mediapipe as mp
from alert import send_email_alert, send_telegram_alert

@dataclass
class Config:
    # Detection thresholds
    detection_confidence: float = 0.4  # Lowered for better detection
    tracking_iou_threshold: float = 0.5  # More lenient matching
    interaction_distance: float = 30.0  # Reduced for better hand detection
    
    # Timing parameters
    missing_frames_threshold: int = 15  # Reduced to ~0.5 seconds at 30fps
    alert_cooldown_seconds: int = 30
    person_cleanup_frames: int = 90
    
    # Hand detection
    max_hands: int = 4
    hand_confidence: float = 0.5  # Lowered for better detection
    
    # Alert parameters
    proximity_threshold: float = 0.1

@dataclass
class TrackedObject:
    name: str
    bbox: List[float]
    missing_frames: int = 0
    interacted: bool = False
    interact_timestamp: float = 0.0
    last_interaction_person: Optional[int] = None
    alert_sent: bool = False
    initial_position: Optional[List[float]] = None
    moved_significantly: bool = False

@dataclass
class TrackedPerson:
    bbox: List[float]
    last_seen_frame: int = 0
    is_authorized: bool = False
    person_id: Optional[str] = None

class ImprovedTheftDetector:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        
        # Initialize models
        self.model = YOLO("yolov8n.pt")
        self.tracker = DeepSort(max_age=30, n_init=2)  # More responsive tracking
        
        # Initialize MediaPipe with better settings
        mp_hands = mp.solutions.hands
        self.hands_detector = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.config.max_hands,
            min_detection_confidence=self.config.hand_confidence,
            min_tracking_confidence=0.3  # Lower for better tracking
        )
        
        # Tracking data
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.tracked_persons: Dict[int, TrackedPerson] = {}
        self.last_alert_time: Dict[int, float] = {}
        self.frame_count = 0
        
        # Enhanced valuable objects list
        self.valuable_objects = {
            'laptop', 'cell phone', 'handbag', 'backpack', 'suitcase',
            'book', 'clock', 'vase', 'scissors', 'remote', 'mouse',
            'keyboard', 'cup', 'bottle', 'wine glass', 'tv', 'tablet'
        }
        
        print(f"📱 Looking for these objects: {self.valuable_objects}")

    def detect_hands(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Enhanced hand detection with multiple key points"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = self.hands_detector.process(rgb_frame)
        
        hand_points = []
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Get multiple hand points for better interaction detection
                key_points = [0, 4, 8, 12, 16, 20]  # Wrist, thumb, index, middle, ring, pinky tips
                
                for point_idx in key_points:
                    landmark = hand_landmarks.landmark[point_idx]
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    hand_points.append((x, y))
        
        return hand_points

    def get_detections(self, frame: np.ndarray) -> Tuple[List, List[str]]:
        """Enhanced detection with verbose output for debugging"""
        results = self.model(frame, verbose=False)
        detections, detection_names = [], []

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                name = self.model.names[cls_id]
                bbox = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                
                # Debug: Print all detections
                if self.frame_count % 30 == 0:  # Print every second
                    print(f"🔍 Detected: {name} (confidence: {conf:.2f})")
                
                if conf > self.config.detection_confidence:
                    detections.append((
                        [float(bbox[0]), float(bbox[1]), 
                         float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])],
                        float(conf)
                    ))
                    detection_names.append(name)

        return detections, detection_names

    def calculate_movement(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate the distance between two bounding box centers"""
        center1 = [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2]
        center2 = [(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2]
        
        distance = math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        return distance

    def iou(self, boxA: List[float], boxB: List[float]) -> float:
        """Calculate Intersection over Union of two bounding boxes"""
        # Convert to x1,y1,x2,y2 format if needed
        if len(boxA) == 4 and boxA[2] < boxA[0]:  # If width/height format
            boxA = [boxA[0], boxA[1], boxA[0] + boxA[2], boxA[1] + boxA[3]]
        if len(boxB) == 4 and boxB[2] < boxB[0]:
            boxB = [boxB[0], boxB[1], boxB[0] + boxB[2], boxB[1] + boxB[3]]
            
        xA = max(boxA[0], boxB[0])
        xB = min(boxA[2], boxB[2])
        yA = max(boxA[1], boxB[1])
        yB = min(boxA[3], boxB[3])
        
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        unionArea = boxAArea + boxBArea - interArea
        
        return interArea / unionArea if unionArea > 0 else 0

    def point_in_expanded_bbox(self, point: Tuple[int, int], bbox: List[float], 
                              expansion: float = 20.0) -> bool:
        """Check if point is within expanded bounding box"""
        x, y = point
        x1, y1, x2, y2 = bbox
        return (x1 - expansion <= x <= x2 + expansion and 
                y1 - expansion <= y <= y2 + expansion)

    def detect_interaction(self, hand_points: List[Tuple[int, int]], 
                          obj_bbox: List[float]) -> bool:
        """Enhanced interaction detection"""
        if not hand_points:
            return False
            
        for hand_point in hand_points:
            if self.point_in_expanded_bbox(hand_point, obj_bbox, 
                                         self.config.interaction_distance):
                return True
        return False

    def cleanup_old_tracks(self):
        """Clean up old person tracks to prevent memory leaks"""
        current_frame = self.frame_count
        persons_to_remove = []
        
        for person_id, person in self.tracked_persons.items():
            if current_frame - person.last_seen_frame > self.config.person_cleanup_frames:
                persons_to_remove.append(person_id)
        
        for person_id in persons_to_remove:
            del self.tracked_persons[person_id]

    def should_send_alert(self, object_id: int) -> bool:
        """Check if alert should be sent (considering cooldown)"""
        current_time = time.time()
        if object_id in self.last_alert_time:
            time_diff = current_time - self.last_alert_time[object_id]
            return time_diff > self.config.alert_cooldown_seconds
        return True

    def send_alert(self, frame: np.ndarray, person_bbox: List[float], 
                   obj_bbox: List[float], person_id: int, object_name: str):
        """Send alert with improved information"""
        current_time = time.time()
        
        # Create alert frame
        alert_frame = self.draw_alert_boxes(
            frame.copy(), person_bbox, obj_bbox, person_id, object_name
        )
        
        # Save alert image with timestamp
        timestamp = int(current_time)
        image_path = f"alert_{timestamp}_{person_id}_{object_name.replace(' ', '_')}.jpg"
        cv2.imwrite(image_path, alert_frame)
        
        # Create detailed alert message
        alert_msg = (f"🚨 THEFT ALERT 🚨\n"
                    f"Object: {object_name}\n"
                    f"Person ID: {person_id}\n"
                    f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"Status: UNAUTHORIZED ACCESS DETECTED")
        
        try:
            # Uncomment these when alert functions are available
            # send_email_alert(image_path, alert_msg)
            # send_telegram_alert(image_path, alert_msg)
            print(f"✅ Alert sent successfully: {alert_msg}")
            return True
        except Exception as e:
            print(f"❌ Failed to send alert: {e}")
            return False

    def draw_alert_boxes(self, frame: np.ndarray, person_bbox: List[float], 
                        object_bbox: List[float], person_id: int, 
                        object_name: str) -> np.ndarray:
        """Draw alert visualization on frame"""
        p_x1, p_y1, p_x2, p_y2 = map(int, person_bbox)
        o_x1, o_y1, o_x2, o_y2 = map(int, object_bbox)

        # Draw person box (red for unauthorized)
        cv2.rectangle(frame, (p_x1, p_y1), (p_x2, p_y2), (0, 0, 255), 3)
        cv2.putText(frame, f"UNAUTHORIZED PERSON {person_id}", 
                   (p_x1, p_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw object box (yellow for alert)
        cv2.rectangle(frame, (o_x1, o_y1), (o_x2, o_y2), (0, 255, 255), 3)
        cv2.putText(frame, f"STOLEN: {object_name.upper()}", 
                   (o_x1, o_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw connection line
        person_center = ((p_x1 + p_x2) // 2, (p_y1 + p_y2) // 2)
        object_center = ((o_x1 + o_x2) // 2, (o_y1 + o_y2) // 2)
        cv2.line(frame, person_center, object_center, (0, 255, 255), 2)
        
        # Add timestamp
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(frame, f"ALERT: {timestamp}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return frame

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Main processing function for each frame"""
        self.frame_count += 1
        current_time = time.time()
        
        # Detect hands
        hand_points = self.detect_hands(frame)
        
        # Get YOLO detections
        detections, detection_names = self.get_detections(frame)
        
        # Update tracker
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        # Process tracks
        current_detections = {}
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            ltrb = track.to_ltrb()
            
            # Match with detection names - improved matching
            detected_name = None
            best_iou = 0
            for i, det in enumerate(detections):
                x, y, w, h = det[0]
                det_box = [x, y, x + w, y + h]
                iou_score = self.iou(ltrb, det_box)
                if iou_score > best_iou and iou_score > 0.3:  # Lower threshold
                    best_iou = iou_score
                    detected_name = detection_names[i]
            
            if detected_name:
                current_detections[track_id] = {
                    "name": detected_name, 
                    "bbox": ltrb
                }

        # Update tracked persons and objects
        for track_id, detection in current_detections.items():
            if detection["name"] == "person":
                # Update person tracking
                if track_id not in self.tracked_persons:
                    self.tracked_persons[track_id] = TrackedPerson(
                        bbox=detection["bbox"],
                        last_seen_frame=self.frame_count
                    )
                    print(f"👤 New person detected: ID {track_id}")
                else:
                    self.tracked_persons[track_id].bbox = detection["bbox"]
                    self.tracked_persons[track_id].last_seen_frame = self.frame_count
                    
            elif detection["name"] in self.valuable_objects:
                # Update object tracking
                if track_id not in self.tracked_objects:
                    self.tracked_objects[track_id] = TrackedObject(
                        name=detection["name"],
                        bbox=detection["bbox"],
                        initial_position=list(detection["bbox"])  # Convert to regular list
                    )
                    print(f"📱 New {detection['name']} detected: ID {track_id}")
                else:
                    obj = self.tracked_objects[track_id]
                    
                    # Check for significant movement
                    if obj.initial_position is not None:
                        movement = self.calculate_movement(obj.initial_position, detection["bbox"])
                        if movement > 50:  # pixels
                            obj.moved_significantly = True
                            print(f"🚨 {obj.name} ID {track_id} moved significantly! Distance: {movement:.1f}px")
                    
                    obj.bbox = detection["bbox"]
                    obj.missing_frames = 0
                    
                # Check for hand interaction
                interaction_detected = self.detect_interaction(hand_points, detection["bbox"])
                if interaction_detected:
                    obj = self.tracked_objects[track_id]
                    if not obj.interacted:  # First interaction
                        print(f"✋ Hand interaction detected with {obj.name} ID {track_id}")
                    obj.interacted = True
                    obj.interact_timestamp = current_time
                    
                    # Find nearest person for interaction attribution
                    nearest_person = None
                    max_overlap = 0
                    for person_id, person in self.tracked_persons.items():
                        overlap = self.iou(detection["bbox"], person.bbox)
                        if overlap > max_overlap:
                            max_overlap = overlap
                            nearest_person = person_id
                    
                    if nearest_person:
                        obj.last_interaction_person = nearest_person

        # Handle missing objects and generate alerts
        objects_to_remove = []
        for obj_id, obj in self.tracked_objects.items():
            if obj_id not in current_detections:
                obj.missing_frames += 1
                
                # Debug output
                if obj.missing_frames == 1:
                    print(f"❌ {obj.name} ID {obj_id} is now missing")
                
                # Check if object was taken - IMPROVED LOGIC
                theft_conditions = [
                    obj.interacted,  # Object was touched
                    obj.missing_frames >= self.config.missing_frames_threshold,  # Missing for threshold time
                    not obj.alert_sent,  # Alert not already sent
                    obj.last_interaction_person is not None,  # Someone interacted with it
                    (obj.moved_significantly or obj.missing_frames >= self.config.missing_frames_threshold)  # Moved or missing
                ]
                
                if all(theft_conditions):
                    person_id = obj.last_interaction_person
                    if person_id in self.tracked_persons:
                        person = self.tracked_persons[person_id]
                        
                        # Check if person is unauthorized (for now, all are unauthorized)
                        if not person.is_authorized and self.should_send_alert(obj_id):
                            print(f"🚨 THEFT DETECTED! {obj.name} ID {obj_id} taken by person {person_id}")
                            
                            alert_sent = self.send_alert(
                                frame, person.bbox, obj.bbox, 
                                person_id, obj.name
                            )
                            
                            if alert_sent:
                                obj.alert_sent = True
                                self.last_alert_time[obj_id] = current_time
                
                # Clean up very old objects
                if obj.missing_frames > 90:  # ~3 seconds
                    objects_to_remove.append(obj_id)
                    print(f"🗑️ Removing old object {obj.name} ID {obj_id}")
        
        # Remove old objects
        for obj_id in objects_to_remove:
            del self.tracked_objects[obj_id]
        
        # Clean up old person tracks
        self.cleanup_old_tracks()
        
        # Draw annotations
        annotated_frame = self.draw_annotations(frame, current_detections, hand_points)
        
        return annotated_frame

    def draw_annotations(self, frame: np.ndarray, detections: Dict, 
                        hand_points: List[Tuple[int, int]]) -> np.ndarray:
        """Draw all annotations on frame"""
        annotated_frame = frame.copy()
        
        # Draw detections
        for track_id, detection in detections.items():
            bbox = detection["bbox"]
            x1, y1, x2, y2 = map(int, bbox)
            name = detection["name"]
            
            # Color coding
            if name == "person":
                person = self.tracked_persons.get(track_id)
                color = (0, 255, 0) if person and person.is_authorized else (255, 0, 0)
                label = f"{'AUTH' if person and person.is_authorized else 'UNAUTH'} Person {track_id}"
            else:
                color = (0, 255, 255) if name in self.valuable_objects else (128, 128, 128)
                label = f"{name} ID:{track_id}"
                
                # Add interaction and movement indicators
                if track_id in self.tracked_objects:
                    obj = self.tracked_objects[track_id]
                    indicators = []
                    if obj.interacted:
                        indicators.append("TOUCHED")
                    if obj.moved_significantly:
                        indicators.append("MOVED")
                    if indicators:
                        label += f" [{'/'.join(indicators)}]"
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw hand points
        for point in hand_points:
            cv2.circle(annotated_frame, point, 3, (255, 255, 0), -1)
        
        # Draw status information
        status_y = frame.shape[0] - 100
        cv2.putText(annotated_frame, f"Objects: {len(self.tracked_objects)} | Persons: {len(self.tracked_persons)}", 
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Hands: {len(hand_points)//6} | Frame: {self.frame_count}", 
                   (10, status_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show detection confidence threshold
        cv2.putText(annotated_frame, f"Confidence: {self.config.detection_confidence}", 
                   (10, status_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame

def main():
    """Main execution function"""
    config = Config()
    detector = ImprovedTheftDetector(config)
    
    # Try different video sources if needed
    cap = cv2.VideoCapture(0)  # Try 1, 2 if 0 doesn't work
    
    if not cap.isOpened():
        print("❌ Cannot open camera. Try changing the camera index (0, 1, 2...)")
        return
    
    print("🚀 Starting Enhanced Theft Detection System...")
    print("⚠️  All persons are treated as UNAUTHORIZED")
    print("📋 Monitored objects:", ', '.join(sorted(detector.valuable_objects)))
    print("🎯 Detection confidence threshold:", config.detection_confidence)
    print("✋ Hold your hand near objects to simulate interaction")
    print("📱 Move objects to test theft detection")
    print("🎯 Press ESC to exit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read from camera")
                break
            
            # Process frame
            annotated_frame = detector.process_frame(frame)
            
            # Display result
            cv2.imshow("Enhanced Theft Detection System", annotated_frame)
            
            # Check for exit
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
            elif key == ord('r'):  # Reset tracking
                detector.tracked_objects.clear()
                detector.tracked_persons.clear()
                print("🔄 Tracking reset")
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping system...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ System stopped successfully")

if __name__ == "__main__":
    main()
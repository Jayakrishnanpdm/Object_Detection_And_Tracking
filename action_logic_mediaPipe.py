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
# from alert import send_email_alert, send_telegram_alert  # Comment out if not available

@dataclass
class Config:
    # Detection thresholds
    detection_confidence: float = 0.5
    tracking_iou_threshold: float = 0.5
    interaction_distance: float = 50.0  # Increased for easier interaction detection
    
    # Timing parameters - IMPROVED and more lenient for testing
    missing_frames_threshold: int = 45  # ~1.5 seconds at 30fps (reduced for testing)
    consecutive_missing_threshold: int = 30  # Must be missing for 1 second continuously (reduced)
    reappearance_reset_frames: int = 15  # Reset after object reappears for 0.5 second
    alert_cooldown_seconds: int = 30  # Reduced cooldown for testing
    person_cleanup_frames: int = 150
    
    # Hand detection
    max_hands: int = 4
    hand_confidence: float = 0.3  # Reduced for better detection
    
    # Enhanced interaction parameters
    proximity_threshold: float = 0.15  # Increased
    hand_person_distance_threshold: float = 100.0  # Increased for easier association
    interaction_confidence_threshold: float = 0.2  # Reduced for easier interaction
    min_interaction_duration: float = 0.5  # Reduced minimum duration
    hand_object_distance_threshold: float = 80.0  # Increased for easier interaction

@dataclass
class HandInteraction:
    """Track hand interactions with objects"""
    person_id: int
    timestamp: float
    confidence: float
    hand_position: Tuple[int, int]
    duration: float = 0.0

@dataclass
class TrackedObject:
    name: str
    bbox: List[float]
    missing_frames: int = 0
    consecutive_missing_frames: int = 0
    reappearance_frames: int = 0
    interacted: bool = False
    interact_timestamp: float = 0.0
    last_interaction_person: Optional[int] = None
    alert_sent: bool = False
    initial_position: Optional[List[float]] = None
    moved_significantly: bool = False
    interaction_history: List[HandInteraction] = field(default_factory=list)
    last_seen_timestamp: float = 0.0
    confirmed_missing: bool = False
    was_missing: bool = False
    stability_counter: int = 0
    false_positive_strikes: int = 0
    first_interaction_logged: bool = False  # NEW: Track if first interaction was logged

@dataclass
class TrackedPerson:
    bbox: List[float]
    last_seen_frame: int = 0
    is_authorized: bool = False
    person_id: Optional[str] = None
    center_history: List[Tuple[int, int]] = field(default_factory=list)
    velocity: Tuple[float, float] = (0.0, 0.0)

class ImprovedTheftDetector:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        
        # Initialize models
        print("🔄 Loading YOLO model...")
        self.model = YOLO("yolov8n.pt")
        print("🔄 Initializing DeepSort tracker...")
        self.tracker = DeepSort(max_age=50, n_init=3)
        
        # Initialize MediaPipe with optimized settings
        print("🔄 Initializing MediaPipe hands...")
        mp_hands = mp.solutions.hands
        self.hands_detector = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.config.max_hands,
            min_detection_confidence=self.config.hand_confidence,
            min_tracking_confidence=0.3
        )
        
        # Enhanced tracking data
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.tracked_persons: Dict[int, TrackedPerson] = {}
        self.last_alert_time: Dict[int, float] = {}
        self.frame_count = 0
        self.fps_counter = []
        
        # Enhanced valuable objects list
        self.valuable_objects = {
            'laptop', 'cell phone', 'handbag', 'backpack', 'suitcase',
            'book', 'clock', 'vase', 'scissors', 'remote', 'mouse',
            'keyboard', 'cup', 'bottle', 'wine glass', 'tv', 'tablet',
            'camera', 'watch', 'wallet', 'purse'
        }
        
        print(f"✅ Initialization complete!")
        print(f"📱 Monitoring {len(self.valuable_objects)} object types")

    def detect_hands_with_enhanced_tracking(self, frame: np.ndarray) -> List[Tuple[Tuple[int, int], Optional[int], float]]:
        """Enhanced hand detection with confidence scoring"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = self.hands_detector.process(rgb_frame)
        
        enhanced_hand_data = []
        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            for idx, (hand_landmarks, handedness) in enumerate(zip(
                hand_results.multi_hand_landmarks, hand_results.multi_handedness)):
                
                # Get hand center and key points
                hand_points = []
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    hand_points.append((x, y))
                
                # Calculate hand center
                hand_center = (
                    int(np.mean([p[0] for p in hand_points])),
                    int(np.mean([p[1] for p in hand_points]))
                )
                
                # Get hand confidence
                hand_confidence = handedness.classification[0].score
                
                # Associate with nearest person
                associated_person, association_confidence = self.associate_hand_with_person_enhanced(
                    hand_center, hand_confidence)
                
                enhanced_hand_data.append((hand_center, associated_person, association_confidence))
                
                # Also add key fingertips for precise interaction detection
                key_points = [4, 8, 12, 16, 20]  # Thumb, index, middle, ring, pinky tips
                for tip_idx in key_points:
                    if tip_idx < len(hand_points):
                        tip_point = hand_points[tip_idx]
                        enhanced_hand_data.append((tip_point, associated_person, association_confidence * 0.7))
        
        return enhanced_hand_data

    def associate_hand_with_person_enhanced(self, hand_point: Tuple[int, int], 
                                          hand_confidence: float) -> Tuple[Optional[int], float]:
        """Enhanced hand-person association with confidence scoring"""
        min_distance = float('inf')
        closest_person_id = None
        association_confidence = 0.0
        
        hand_x, hand_y = hand_point
        
        for person_id, person in self.tracked_persons.items():
            bbox = person.bbox
            
            # Check if hand is within expanded person bounding box
            margin = 30  # Add margin for better association
            expanded_bbox = [bbox[0] - margin, bbox[1] - margin, 
                           bbox[2] + margin, bbox[3] + margin]
            
            if (expanded_bbox[0] <= hand_x <= expanded_bbox[2] and 
                expanded_bbox[1] <= hand_y <= expanded_bbox[3]):
                return person_id, 0.9 * hand_confidence
            
            # Calculate distance to person center
            person_center_x = (bbox[0] + bbox[2]) / 2
            person_center_y = (bbox[1] + bbox[3]) / 2
            distance = math.sqrt((hand_x - person_center_x)**2 + (hand_y - person_center_y)**2)
            
            if distance < self.config.hand_person_distance_threshold and distance < min_distance:
                min_distance = distance
                closest_person_id = person_id
                association_confidence = max(0.2, 
                    (1.0 - distance / self.config.hand_person_distance_threshold) * hand_confidence)
        
        return closest_person_id, association_confidence

    def get_detections_with_stability(self, frame: np.ndarray) -> Tuple[List, List[str]]:
        """Enhanced detection with stability filtering"""
        results = self.model(frame, verbose=False)
        detections, detection_names = [], []

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                name = self.model.names[cls_id]
                bbox = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                
                if conf > self.config.detection_confidence:
                    detections.append((
                        [float(bbox[0]), float(bbox[1]), 
                         float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])],
                        float(conf)
                    ))
                    detection_names.append(name)

        return detections, detection_names

    def calculate_movement_with_smoothing(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate movement distance between two bounding boxes"""
        center1 = [(bbox1[0] + bbox1[2]/2), (bbox1[1] + bbox1[3]/2)]
        center2 = [(bbox2[0] + bbox2[2]/2), (bbox2[1] + bbox2[3]/2)]
        
        distance = math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        return distance

    def detect_hand_object_interaction_enhanced(self, hand_data: List[Tuple[Tuple[int, int], Optional[int], float]], 
                                              obj_bbox: List[float], obj_id: int) -> Optional[HandInteraction]:
        """Enhanced interaction detection with duration and confidence"""
        current_time = time.time()
        best_interaction = None
        highest_confidence = 0.0
        
        for (hand_point, person_id, association_confidence) in hand_data:
            if person_id is None or association_confidence < self.config.interaction_confidence_threshold:
                continue
            
            # Check if hand is close to object
            hand_object_distance = self.calculate_point_to_bbox_distance(hand_point, obj_bbox)
            
            if hand_object_distance <= self.config.hand_object_distance_threshold:
                # Calculate interaction confidence
                distance_confidence = max(0.1, 1.0 - (hand_object_distance / self.config.hand_object_distance_threshold))
                total_confidence = association_confidence * distance_confidence
                
                if total_confidence > highest_confidence:
                    highest_confidence = total_confidence
                    best_interaction = HandInteraction(
                        person_id=person_id,
                        timestamp=current_time,
                        confidence=total_confidence,
                        hand_position=hand_point
                    )
        
        # Calculate interaction duration
        if best_interaction and obj_id in self.tracked_objects:
            obj = self.tracked_objects[obj_id]
            recent_interactions = [
                i for i in obj.interaction_history 
                if i.person_id == best_interaction.person_id and 
                current_time - i.timestamp <= 3.0
            ]
            if recent_interactions:
                best_interaction.duration = current_time - recent_interactions[0].timestamp
        
        return best_interaction

    def calculate_point_to_bbox_distance(self, point: Tuple[int, int], bbox: List[float]) -> float:
        """Calculate minimum distance from point to bounding box"""
        px, py = point
        
        # Convert bbox to x1, y1, x2, y2 format if needed
        if len(bbox) == 4:
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
        else:
            x1, y1, x2, y2 = bbox
        
        # If point is inside bbox, distance is 0
        if x1 <= px <= x2 and y1 <= py <= y2:
            return 0.0
        
        # Calculate distance to nearest edge
        dx = max(x1 - px, 0, px - x2)
        dy = max(y1 - py, 0, py - y2)
        return math.sqrt(dx*dx + dy*dy)

    def update_person_velocity(self, person_id: int, new_bbox: List[float]):
        """Update person velocity for better tracking"""
        if person_id in self.tracked_persons:
            person = self.tracked_persons[person_id]
            old_center = ((person.bbox[0] + person.bbox[2]) / 2, (person.bbox[1] + person.bbox[3]) / 2)
            new_center = ((new_bbox[0] + new_bbox[2]) / 2, (new_bbox[1] + new_bbox[3]) / 2)
            
            person.velocity = (new_center[0] - old_center[0], new_center[1] - old_center[1])

    def is_valid_theft_scenario(self, obj: TrackedObject) -> Tuple[bool, str]:
        """FIXED: More lenient theft validation with detailed logging"""
        current_time = time.time()
        
        print(f"🔍 THEFT CHECK for {obj.name}:")
        print(f"   - Interacted: {obj.interacted}")
        print(f"   - Confirmed missing: {obj.confirmed_missing}")
        print(f"   - Consecutive missing frames: {obj.consecutive_missing_frames}/{self.config.consecutive_missing_threshold}")
        print(f"   - Missing frames: {obj.missing_frames}/{self.config.missing_frames_threshold}")
        print(f"   - Interaction history: {len(obj.interaction_history)} interactions")
        print(f"   - Alert already sent: {obj.alert_sent}")
        
        # Criterion 1: Object must have been interacted with
        if not obj.interacted or len(obj.interaction_history) == 0:
            print(f"   ❌ FAIL: No interaction detected")
            return False, "No interaction detected"
        
        # Criterion 2: Object must be missing for minimum consecutive frames
        if obj.consecutive_missing_frames < self.config.consecutive_missing_threshold:
            print(f"   ❌ FAIL: Missing for only {obj.consecutive_missing_frames} consecutive frames (need {self.config.consecutive_missing_threshold})")
            return False, f"Missing for only {obj.consecutive_missing_frames} consecutive frames"
        
        # Criterion 3: Total missing frames threshold
        if obj.missing_frames < self.config.missing_frames_threshold:
            print(f"   ❌ FAIL: Total missing frames {obj.missing_frames} < {self.config.missing_frames_threshold}")
            return False, f"Total missing frames insufficient"
        
        # Criterion 4: Recent meaningful interaction (more lenient)
        recent_interactions = [
            i for i in obj.interaction_history 
            if current_time - i.timestamp <= 15.0  # Increased time window
        ]
        if not recent_interactions:
            print(f"   ❌ FAIL: No recent interactions within 15 seconds")
            return False, "No recent interactions"
        
        # Criterion 5: Alert not already sent
        if obj.alert_sent:
            print(f"   ❌ FAIL: Alert already sent")
            return False, "Alert already sent"
        
        # Criterion 6: Confidence threshold (more lenient)
        max_confidence = max(i.confidence for i in recent_interactions)
        if max_confidence < 0.3:  # Reduced threshold
            print(f"   ❌ FAIL: Interaction confidence too low: {max_confidence:.2f}")
            return False, f"Interaction confidence too low: {max_confidence:.2f}"
        
        print(f"   ✅ PASS: All theft criteria met!")
        return True, "All criteria met"

    def get_theft_suspect_with_evidence(self, obj: TrackedObject) -> Tuple[Optional[int], Dict]:
        """Determine theft suspect with detailed evidence"""
        current_time = time.time()
        
        # Get recent interactions (more lenient time window)
        recent_interactions = [
            i for i in obj.interaction_history
            if current_time - i.timestamp <= 15.0 and  # Increased window
            i.confidence >= 0.2  # Reduced threshold
        ]
        
        if not recent_interactions:
            print(f"   ❌ No qualifying recent interactions")
            return None, {}
        
        # Score persons based on interaction evidence
        person_evidence = defaultdict(lambda: {
            'total_confidence': 0.0, 'interaction_count': 0, 'total_duration': 0.0,
            'last_interaction': 0.0, 'max_confidence': 0.0
        })
        
        for interaction in recent_interactions:
            pid = interaction.person_id
            evidence = person_evidence[pid]
            
            # Weight recent interactions more heavily
            time_weight = max(0.2, 1.0 - ((current_time - interaction.timestamp) / 15.0))
            weighted_confidence = interaction.confidence * time_weight
            
            evidence['total_confidence'] += weighted_confidence
            evidence['interaction_count'] += 1
            evidence['total_duration'] += interaction.duration
            evidence['last_interaction'] = max(evidence['last_interaction'], interaction.timestamp)
            evidence['max_confidence'] = max(evidence['max_confidence'], interaction.confidence)
        
        # Find most suspicious person
        best_suspect = None
        best_score = 0.0
        best_evidence = {}
        
        for person_id, evidence in person_evidence.items():
            # Calculate suspicion score (more lenient)
            score = (evidence['total_confidence'] * 
                    math.log(evidence['interaction_count'] + 1) * 
                    min(evidence['total_duration'] + 1, 5.0))  # +1 for short interactions
            
            if score > best_score:
                best_score = score
                best_suspect = person_id
                best_evidence = dict(evidence)
                best_evidence['suspicion_score'] = score
        
        print(f"   🎯 Best suspect: Person {best_suspect} (score: {best_score:.2f})")
        return best_suspect, best_evidence

    def send_enhanced_alert(self, frame: np.ndarray, person_bbox: List[float], 
                          obj_bbox: List[float], person_id: int, object_name: str,
                          evidence: Dict) -> bool:
        """Send enhanced alert with detailed evidence"""
        current_time = time.time()
        
        print(f"\n" + "="*60)
        print(f"🚨 SENDING THEFT ALERT 🚨")
        print(f"="*60)
        
        # Create enhanced alert frame
        alert_frame = self.draw_theft_alert_frame(
            frame.copy(), person_bbox, obj_bbox, person_id, object_name, evidence
        )
        
        # Save alert image
        timestamp = int(current_time)
        image_path = f"theft_alert_{timestamp}_{person_id}_{object_name.replace(' ', '_')}.jpg"
        cv2.imwrite(image_path, alert_frame)
        print(f"💾 Alert image saved: {image_path}")
        
        # Create detailed alert message
        alert_msg = self.create_detailed_alert_message(object_name, person_id, evidence, timestamp)
        
        try:
            # Uncomment these when alert functions are available
            # send_email_alert(image_path, alert_msg)
            # send_telegram_alert(image_path, alert_msg)
            
            print(alert_msg)
            print(f"="*60)
            return True
        except Exception as e:
            print(f"❌ Failed to send alert: {e}")
            return False

    def create_detailed_alert_message(self, object_name: str, person_id: int, 
                                    evidence: Dict, timestamp: int) -> str:
        """Create detailed alert message with evidence"""
        return f"""
🚨 THEFT ALERT - HIGH CONFIDENCE 🚨
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📱 STOLEN OBJECT: {object_name.upper()}
👤 SUSPECT: Person ID {person_id} (UNAUTHORIZED)
🕒 TIME: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))}

📊 EVIDENCE SUMMARY:
   • Suspicion Score: {evidence.get('suspicion_score', 0):.2f}
   • Interactions: {evidence.get('interaction_count', 0)}
   • Max Confidence: {evidence.get('max_confidence', 0):.2f}
   • Total Duration: {evidence.get('total_duration', 0):.1f}s
   • Last Interaction: {time.strftime('%H:%M:%S', time.localtime(evidence.get('last_interaction', 0)))}

🔍 STATUS: OBJECT CONFIRMED STOLEN
⚠️ ACTION REQUIRED: INVESTIGATE IMMEDIATELY
        """.strip()

    def draw_theft_alert_frame(self, frame: np.ndarray, person_bbox: List[float], 
                             object_bbox: List[float], person_id: int, 
                             object_name: str, evidence: Dict) -> np.ndarray:
        """Draw comprehensive theft alert visualization"""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Draw suspect person (thick red box)
        p_x1, p_y1, p_x2, p_y2 = map(int, person_bbox)
        cv2.rectangle(overlay, (p_x1-3, p_y1-3), (p_x2+3, p_y2+3), (0, 0, 255), 5)
        cv2.putText(overlay, f"THIEF - PERSON {person_id}", 
                   (p_x1, p_y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        
        # Draw stolen object (thick yellow box) - use last known position
        o_x1, o_y1, o_x2, o_y2 = map(int, object_bbox)
        cv2.rectangle(overlay, (o_x1-3, o_y1-3), (o_x2+3, o_y2+3), (0, 255, 255), 5)
        cv2.putText(overlay, f"STOLEN: {object_name.upper()}", 
                   (o_x1, o_y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3)
        
        # Add alert banner
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 255), -1)
        cv2.putText(overlay, "THEFT DETECTED", (w//2 - 150, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(overlay, time.strftime('%Y-%m-%d %H:%M:%S'), (w//2 - 100, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return overlay

    def reset_object_status(self, obj_id: int, obj: TrackedObject):
        """Reset object status when it reappears"""
        print(f"🔄 OBJECT RESET: {obj.name} ID {obj_id} has reappeared - resetting theft status")
        obj.missing_frames = 0
        obj.consecutive_missing_frames = 0
        obj.confirmed_missing = False
        obj.was_missing = True
        obj.alert_sent = False  # Allow new alerts
        obj.reappearance_frames = 1
        obj.stability_counter = 1

    def cleanup_old_tracks(self):
        """Enhanced cleanup with better memory management"""
        current_frame = self.frame_count
        current_time = time.time()
        
        # Clean up old persons
        persons_to_remove = []
        for person_id, person in self.tracked_persons.items():
            if current_frame - person.last_seen_frame > self.config.person_cleanup_frames:
                persons_to_remove.append(person_id)
        
        for person_id in persons_to_remove:
            print(f"🗑️ Removing old person ID {person_id}")
            del self.tracked_persons[person_id]
            if person_id in self.last_alert_time:
                del self.last_alert_time[person_id]
        
        # Clean up old interaction histories
        for obj in self.tracked_objects.values():
            old_count = len(obj.interaction_history)
            obj.interaction_history = [
                i for i in obj.interaction_history 
                if current_time - i.timestamp <= 30.0
            ]
            new_count = len(obj.interaction_history)
            if old_count != new_count:
                print(f"🧹 Cleaned {old_count - new_count} old interactions for {obj.name}")

    def should_send_alert(self, object_id: int) -> bool:
        """Enhanced alert cooldown check"""
        current_time = time.time()
        if object_id in self.last_alert_time:
            time_diff = current_time - self.last_alert_time[object_id]
            remaining = self.config.alert_cooldown_seconds - time_diff
            if remaining > 0:
                print(f"🔄 Alert cooldown: {remaining:.1f}s remaining for object {object_id}")
                return False
        return True

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Main processing function with enhanced logic and better terminal output"""
        start_time = time.time()
        self.frame_count += 1
        current_time = time.time()
        
        # Enhanced hand detection
        enhanced_hand_data = self.detect_hands_with_enhanced_tracking(frame)
        
        # Get stable detections
        detections, detection_names = self.get_detections_with_stability(frame)
        
        # Update tracker
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        # Process tracks with enhanced logic
        current_detections = {}
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            ltrb = track.to_ltrb()
            
            # Enhanced track-detection matching
            detected_name = None
            best_iou = 0
            for i, det in enumerate(detections):
                x, y, w, h = det[0]
                det_box = [x, y, x + w, y + h]
                iou_score = self.iou(ltrb, det_box)
                if iou_score > best_iou and iou_score > 0.3:
                    best_iou = iou_score
                    detected_name = detection_names[i]
            
            if detected_name:
                current_detections[track_id] = {
                    "name": detected_name, 
                    "bbox": ltrb,
                    "confidence": best_iou
                }

        # Update tracking
        for track_id, detection in current_detections.items():
            if detection["name"] == "person":
                self.update_person_tracking(track_id, detection)
            elif detection["name"] in self.valuable_objects:
                self.update_object_tracking_enhanced(track_id, detection, enhanced_hand_data)

        # Enhanced missing object handling
        self.handle_missing_objects_enhanced(current_detections, enhanced_hand_data, frame)
        
        # Cleanup every 5 seconds
        if self.frame_count % 150 == 0:
            self.cleanup_old_tracks()
        
        # Performance tracking
        process_time = time.time() - start_time
        self.fps_counter.append(1.0 / process_time if process_time > 0 else 30.0)
        if len(self.fps_counter) > 30:
            self.fps_counter.pop(0)
        
        # Draw enhanced annotations
        annotated_frame = self.draw_enhanced_annotations(
            frame, current_detections, enhanced_hand_data)
        
        return annotated_frame

    def update_person_tracking(self, track_id: int, detection: Dict):
        """Enhanced person tracking update"""
        if track_id not in self.tracked_persons:
            self.tracked_persons[track_id] = TrackedPerson(
                bbox=detection["bbox"],
                last_seen_frame=self.frame_count
            )
            print(f"👤 NEW PERSON DETECTED: ID {track_id}")
        else:
            person = self.tracked_persons[track_id]
            self.update_person_velocity(track_id, detection["bbox"])
            person.bbox = detection["bbox"]
            person.last_seen_frame = self.frame_count

    def update_object_tracking_enhanced(self, track_id: int, detection: Dict, 
                                      enhanced_hand_data: List):
        """Enhanced object tracking with better terminal output"""
        current_time = time.time()
        
        if track_id not in self.tracked_objects:
            # New object detection
            self.tracked_objects[track_id] = TrackedObject(
                name=detection["name"],
                bbox=detection["bbox"],
                initial_position=list(detection["bbox"]),
                last_seen_timestamp=current_time,
                stability_counter=1
            )
            print(f"📱 NEW OBJECT DETECTED: {detection['name']} ID {track_id}")
        else:
            obj = self.tracked_objects[track_id]
            
            # Object is currently visible - handle reappearance
            if obj.was_missing or obj.consecutive_missing_frames > 0:
                obj.reappearance_frames += 1
                if obj.reappearance_frames >= self.config.reappearance_reset_frames:
                    self.reset_object_status(track_id, obj)
            
            # Reset missing counters
            obj.missing_frames = 0
            obj.consecutive_missing_frames = 0
            obj.confirmed_missing = False
            obj.last_seen_timestamp = current_time
            obj.stability_counter = min(obj.stability_counter + 1, 10)
            
            # Check for significant movement with better terminal output
            if obj.initial_position and not obj.moved_significantly:
                movement = self.calculate_movement_with_smoothing(obj.initial_position, detection["bbox"])
                if movement > 50 and obj.stability_counter > 3:  # More lenient
                    obj.moved_significantly = True
                    print(f"🚨 OBJECT MOVED: {obj.name} ID {track_id} moved {movement:.1f}px from initial position!")
            
            obj.bbox = detection["bbox"]
            
        # Enhanced interaction detection with better logging
        interaction = self.detect_hand_object_interaction_enhanced(
            enhanced_hand_data, detection["bbox"], track_id)
        
        if interaction and track_id in self.tracked_objects:
            obj = self.tracked_objects[track_id]
            
            # Log first interaction
            if not obj.first_interaction_logged:
                print(f"✋ FIRST INTERACTION: {obj.name} ID {track_id} touched by Person {interaction.person_id} "
                      f"(confidence: {interaction.confidence:.2f}, distance: {self.calculate_point_to_bbox_distance(interaction.hand_position, detection['bbox']):.1f}px)")
                obj.first_interaction_logged = True
            
            # Add to interaction history with deduplication
            should_add = True
            if obj.interaction_history:
                last_interaction = obj.interaction_history[-1]
                if (interaction.person_id == last_interaction.person_id and 
                    current_time - last_interaction.timestamp < 0.5):
                    # Update duration of existing interaction
                    last_interaction.duration = current_time - last_interaction.timestamp
                    should_add = False
            
            if should_add:
                obj.interaction_history.append(interaction)
                # Log significant interactions
                if interaction.confidence > 0.5:
                    print(f"✋ STRONG INTERACTION: {obj.name} ID {track_id} with Person {interaction.person_id} "
                          f"(confidence: {interaction.confidence:.2f}, duration: {interaction.duration:.1f}s)")
            
            # Clean old interactions
            obj.interaction_history = [
                i for i in obj.interaction_history 
                if current_time - i.timestamp <= 20.0
            ]
            
            # Mark as interacted
            if not obj.interacted:
                print(f"📝 OBJECT MARKED AS INTERACTED: {obj.name} ID {track_id}")
            
            obj.interacted = True
            obj.interact_timestamp = current_time
            obj.last_interaction_person = interaction.person_id

    def handle_missing_objects_enhanced(self, current_detections: Dict, 
                                      enhanced_hand_data: List, frame: np.ndarray):
        """Enhanced missing object handling with detailed terminal output"""
        current_time = time.time()
        objects_to_remove = []
        
        for obj_id, obj in self.tracked_objects.items():
            if obj_id not in current_detections:
                # Object is missing
                obj.missing_frames += 1
                obj.consecutive_missing_frames += 1
                obj.reappearance_frames = 0
                
                # Progressive missing status with better logging
                if obj.consecutive_missing_frames == 1:
                    print(f"❓ OBJECT MISSING: {obj.name} ID {obj_id} disappeared from view")
                elif obj.consecutive_missing_frames == self.config.consecutive_missing_threshold:
                    obj.confirmed_missing = True
                    print(f"⚠️ CONFIRMED MISSING: {obj.name} ID {obj_id} missing for {obj.consecutive_missing_frames} consecutive frames "
                          f"({obj.consecutive_missing_frames/30:.1f}s)")
                elif obj.consecutive_missing_frames == self.config.missing_frames_threshold:
                    print(f"🔥 THEFT THRESHOLD REACHED: {obj.name} ID {obj_id} missing for {obj.missing_frames} total frames "
                          f"({obj.missing_frames/30:.1f}s)")
                
                # Enhanced theft detection with detailed logging
                if (obj.stability_counter > 2 and 
                    obj.consecutive_missing_frames >= self.config.consecutive_missing_threshold and
                    obj.missing_frames >= self.config.missing_frames_threshold):
                    
                    print(f"\n🔍 CHECKING THEFT SCENARIO for {obj.name} ID {obj_id}...")
                    is_valid_theft, reason = self.is_valid_theft_scenario(obj)
                    
                    if is_valid_theft and self.should_send_alert(obj_id):
                        suspect_id, evidence = self.get_theft_suspect_with_evidence(obj)
                        
                        if (suspect_id and suspect_id in self.tracked_persons and 
                            not self.tracked_persons[suspect_id].is_authorized):
                            
                            print(f"🚨 THEFT CONFIRMED: {obj.name} ID {obj_id} stolen by Person {suspect_id}!")
                            print(f"   Evidence score: {evidence.get('suspicion_score', 0):.2f}")
                            print(f"   Interactions: {evidence.get('interaction_count', 0)}")
                            print(f"   Max confidence: {evidence.get('max_confidence', 0):.2f}")
                            
                            alert_sent = self.send_enhanced_alert(
                                frame, self.tracked_persons[suspect_id].bbox, 
                                obj.bbox, suspect_id, obj.name, evidence
                            )
                            
                            if alert_sent:
                                obj.alert_sent = True
                                self.last_alert_time[obj_id] = current_time
                                print(f"✅ THEFT ALERT SENT SUCCESSFULLY for {obj.name} ID {obj_id}")
                        else:
                            if not suspect_id:
                                print(f"❌ No valid suspect found for {obj.name} ID {obj_id}")
                            elif suspect_id not in self.tracked_persons:
                                print(f"❌ Suspect {suspect_id} no longer tracked for {obj.name} ID {obj_id}")
                            else:
                                print(f"❌ Suspect {suspect_id} is authorized for {obj.name} ID {obj_id}")
                    else:
                        if obj.consecutive_missing_frames % 30 == 0:  # Log every second
                            print(f"❌ THEFT NOT CONFIRMED for {obj.name} ID {obj_id}: {reason}")
                
                # Remove very old objects
                if obj.missing_frames > 450:  # ~15 seconds
                    objects_to_remove.append(obj_id)
                    print(f"🗑️ REMOVING OLD OBJECT: {obj.name} ID {obj_id} "
                          f"(missing {obj.missing_frames} frames / {obj.missing_frames/30:.1f}s)")
        
        # Clean up old objects
        for obj_id in objects_to_remove:
            if obj_id in self.tracked_objects:
                del self.tracked_objects[obj_id]
            if obj_id in self.last_alert_time:
                del self.last_alert_time[obj_id]

    def iou(self, boxA: List[float], boxB: List[float]) -> float:
        """Enhanced IoU calculation"""
        # Convert to x1,y1,x2,y2 format if needed
        if len(boxA) == 4 and boxA[2] < boxA[0]:
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

    def draw_enhanced_annotations(self, frame: np.ndarray, detections: Dict, 
                                enhanced_hand_data: List) -> np.ndarray:
        """Draw comprehensive annotations with enhanced visuals"""
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw object detections with enhanced status
        for track_id, detection in detections.items():
            bbox = detection["bbox"]
            x1, y1, x2, y2 = map(int, bbox)
            name = detection["name"]
            
            if name == "person":
                person = self.tracked_persons.get(track_id)
                color = (0, 255, 0) if person and person.is_authorized else (255, 100, 100)
                label = f"{'AUTH' if person and person.is_authorized else 'UNAUTH'} P{track_id}"
                
                # Draw velocity indicator
                if person and (abs(person.velocity[0]) > 2 or abs(person.velocity[1]) > 2):
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    vel_end = (center[0] + int(person.velocity[0] * 3), 
                              center[1] + int(person.velocity[1] * 3))
                    cv2.arrowedLine(annotated_frame, center, vel_end, (255, 255, 0), 2)
                    
            else:
                # Enhanced object visualization
                if track_id in self.tracked_objects:
                    obj = self.tracked_objects[track_id]
                    
                    # Color coding based on status
                    if obj.alert_sent:
                        color = (0, 0, 255)  # Red - theft alert sent
                    elif obj.confirmed_missing:
                        color = (0, 100, 255)  # Orange - confirmed missing
                    elif obj.missing_frames > 0:
                        color = (0, 200, 255)  # Light orange - temporarily missing
                    elif obj.interacted:
                        color = (0, 255, 255)  # Yellow - interacted
                    else:
                        color = (0, 255, 0)  # Green - normal
                    
                    # Create detailed label
                    status_indicators = []
                    if obj.interacted:
                        status_indicators.append(f"TOUCHED({len(obj.interaction_history)})")
                    if obj.moved_significantly:
                        status_indicators.append("MOVED")
                    if obj.confirmed_missing:
                        status_indicators.append("MISSING")
                    elif obj.missing_frames > 0:
                        status_indicators.append(f"?{obj.consecutive_missing_frames}f")
                    if obj.alert_sent:
                        status_indicators.append("ALERT_SENT")
                    
                    label = f"{name} #{track_id}"
                    if status_indicators:
                        label += f" [{'/'.join(status_indicators)}]"
                        
                    # Stability indicator
                    stability_color = (0, 255, 0) if obj.stability_counter > 5 else (100, 100, 255)
                    cv2.circle(annotated_frame, (x2-10, y1+10), 5, stability_color, -1)
                    
                else:
                    color = (128, 128, 128)
                    label = f"{name} #{track_id}"
            
            # Draw bounding box
            thickness = 3 if name in self.valuable_objects else 2
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Enhanced label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1-25), (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw enhanced hand points with confidence
        for (hand_point, person_id, confidence) in enhanced_hand_data:
            if confidence > 0.2:  # Show more hands for debugging
                radius = max(3, int(confidence * 8))
                color = (0, 255, 0) if person_id else (100, 100, 255)
                cv2.circle(annotated_frame, hand_point, radius, color, -1)
                
                if person_id is not None:
                    cv2.putText(annotated_frame, f"P{person_id}", 
                               (hand_point[0] + 8, hand_point[1] - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Enhanced status panel with more information
        panel_height = 160
        cv2.rectangle(annotated_frame, (0, h - panel_height), (w, h), (0, 0, 0), -1)
        
        # Main stats
        avg_fps = np.mean(self.fps_counter) if self.fps_counter else 0
        cv2.putText(annotated_frame, f"Objects: {len(self.tracked_objects)} | "
                   f"Persons: {len(self.tracked_persons)} | "
                   f"FPS: {avg_fps:.1f}", 
                   (10, h - 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Interaction stats
        total_interactions = sum(len(obj.interaction_history) for obj in self.tracked_objects.values())
        active_hands = len([h for h in enhanced_hand_data if h[2] > 0.2])
        cv2.putText(annotated_frame, f"Hands: {active_hands} | "
                   f"Interactions: {total_interactions} | "
                   f"Frame: {self.frame_count}", 
                   (10, h - 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Object status summary
        interacted_count = sum(1 for obj in self.tracked_objects.values() if obj.interacted)
        moved_count = sum(1 for obj in self.tracked_objects.values() if obj.moved_significantly)
        cv2.putText(annotated_frame, f"Touched: {interacted_count} | "
                   f"Moved: {moved_count} | "
                   f"Missing Threshold: {self.config.missing_frames_threshold}f", 
                   (10, h - 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Alert stats
        alerts_sent = sum(1 for obj in self.tracked_objects.values() if obj.alert_sent)
        missing_objects = sum(1 for obj in self.tracked_objects.values() if obj.confirmed_missing)
        cv2.putText(annotated_frame, f"Alerts Sent: {alerts_sent} | "
                   f"Missing: {missing_objects} | "
                   f"Consecutive Threshold: {self.config.consecutive_missing_threshold}f", 
                   (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Thresholds info
        cv2.putText(annotated_frame, f"Hand Distance: {self.config.hand_object_distance_threshold}px | "
                   f"Interaction Confidence: {self.config.interaction_confidence_threshold}", 
                   (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Controls
        cv2.putText(annotated_frame, "ESC: Exit | R: Reset | D: Debug | S: Stats | I: Info", 
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return annotated_frame

def main():
    """Enhanced main execution function with better terminal feedback"""
    print("\n" + "="*80)
    print("🚀 INITIALIZING ENHANCED THEFT DETECTION SYSTEM")
    print("="*80)
    
    # Create enhanced configuration
    config = Config()
    print(f"⚙️ Configuration loaded:")
    print(f"   - Missing frames threshold: {config.missing_frames_threshold} frames ({config.missing_frames_threshold/30:.1f}s)")
    print(f"   - Consecutive missing threshold: {config.consecutive_missing_threshold} frames ({config.consecutive_missing_threshold/30:.1f}s)")
    print(f"   - Hand-object distance: {config.hand_object_distance_threshold}px")
    print(f"   - Interaction confidence: {config.interaction_confidence_threshold}")
    print(f"   - Alert cooldown: {config.alert_cooldown_seconds}s")
    
    detector = ImprovedTheftDetector(config)
    
    # Try different camera sources
    camera_sources = [0, 1, 2]
    cap = None
    
    for source in camera_sources:
        print(f"🔍 Trying camera source {source}...")
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            print(f"📷 Camera source {source} opened successfully")
            break
        cap.release()
    
    if cap is None or not cap.isOpened():
        print("❌ Cannot open any camera source")
        print("💡 Make sure your camera is connected and not in use by another application")
        return
    
    # Optimize camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"📷 Camera settings: {actual_width}x{actual_height} @ {actual_fps}fps")
    
    # Print system information
    print("\n" + "="*80)
    print("🎯 ENHANCED THEFT DETECTION SYSTEM - ACTIVE")
    print("="*80)
    print("⚠️  All persons are treated as UNAUTHORIZED by default")
    print(f"📱 Monitoring {len(detector.valuable_objects)} object types:")
    print(f"   {', '.join(sorted(list(detector.valuable_objects)[:10]))}...")
    print(f"🎯 Detection Strategy:")
    print(f"   1. Object must be TOUCHED by a person (hand interaction)")
    print(f"   2. Object must disappear for {config.consecutive_missing_threshold}+ consecutive frames")
    print(f"   3. Object must be missing for {config.missing_frames_threshold}+ total frames")
    print(f"   4. Recent interaction within 15 seconds")
    print(f"   5. Interaction confidence > 0.3")
    print("\n📋 HOW TO TEST:")
    print("   1. Place a valuable object (laptop, phone, book, etc.) in view")
    print("   2. Move your hand close to the object (within 80px)")
    print("   3. Wait for 'TOUCHED' status to appear")
    print("   4. Remove the object from camera view")
    print("   5. Wait for theft detection (1-2 seconds)")
    print("\n⌨️  CONTROLS:")
    print("   ESC: Exit system")
    print("   R: Reset all tracking data")
    print("   D: Print detailed debug information")
    print("   S: Toggle statistics display")
    print("   I: Print current interaction info")
    print("="*80)
    print("🟢 SYSTEM READY - Monitoring for objects and interactions...")
    print("="*80)
    
    frame_skip = 0
    stats_enabled = False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read from camera")
                break
            
            # Process every frame for better responsiveness
            try:
                annotated_frame = detector.process_frame(frame)
                cv2.imshow("Enhanced Theft Detection System", annotated_frame)
                
            except Exception as e:
                print(f"❌ Frame processing error: {e}")
                continue
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                print("\n🛑 User requested system stop...")
                break
            elif key == ord('r') or key == ord('R'):
                # Reset all tracking
                detector.tracked_objects.clear()
                detector.tracked_persons.clear()
                detector.last_alert_time.clear()
                detector.frame_count = 0
                print("\n🔄 ALL TRACKING DATA RESET")
            elif key == ord('d') or key == ord('D'):
                # Debug information
                print(f"\n" + "="*60)
                print(f"📊 DEBUG INFORMATION (Frame {detector.frame_count})")
                print(f"="*60)
                print(f"🎥 System Stats:")
                print(f"   - FPS: {np.mean(detector.fps_counter) if detector.fps_counter else 0:.1f}")
                print(f"   - Objects tracked: {len(detector.tracked_objects)}")
                print(f"   - Persons tracked: {len(detector.tracked_persons)}")
                
                print(f"\n📱 Object Details:")
                for obj_id, obj in detector.tracked_objects.items():
                    status = []
                    if obj.interacted: status.append("TOUCHED")
                    if obj.moved_significantly: status.append("MOVED") 
                    if obj.confirmed_missing: status.append("CONFIRMED_MISSING")
                    if obj.missing_frames > 0: status.append(f"MISSING_{obj.missing_frames}f")
                    if obj.alert_sent: status.append("ALERT_SENT")
                    
                    print(f"   📦 {obj.name} #{obj_id}: {'/'.join(status) if status else 'NORMAL'}")
                    print(f"      - Missing: {obj.missing_frames}/{detector.config.missing_frames_threshold} frames")
                    print(f"      - Consecutive: {obj.consecutive_missing_frames}/{detector.config.consecutive_missing_threshold} frames")
                    print(f"      - Interactions: {len(obj.interaction_history)}")
                    print(f"      - Stability: {obj.stability_counter}/10")
                    if obj.interaction_history:
                        latest = obj.interaction_history[-1]
                        print(f"      - Last interaction: Person {latest.person_id} ({latest.confidence:.2f})")
                
                print(f"\n👥 Person Details:")
                for person_id, person in detector.tracked_persons.items():
                    auth_status = "AUTHORIZED" if person.is_authorized else "UNAUTHORIZED"
                    print(f"   👤 Person #{person_id}: {auth_status}")
                    print(f"      - Last seen: frame {person.last_seen_frame}")
                    print(f"      - Velocity: ({person.velocity[0]:.1f}, {person.velocity[1]:.1f})")
                print(f"="*60)
                
            elif key == ord('s') or key == ord('S'):
                stats_enabled = not stats_enabled
                print(f"📈 Detailed statistics {'ENABLED' if stats_enabled else 'DISABLED'}")
                
            elif key == ord('i') or key == ord('I'):
                # Print current interaction info
                print(f"\n📊 CURRENT INTERACTION STATUS:")
                for obj_id, obj in detector.tracked_objects.items():
                    if obj.interacted and obj.interaction_history:
                        latest = obj.interaction_history[-1]
                        time_ago = time.time() - latest.timestamp
                        print(f"   ✋ {obj.name} #{obj_id}: Last touched by Person {latest.person_id} "
                              f"{time_ago:.1f}s ago (conf: {latest.confidence:.2f})")
                    elif obj.interacted:
                        print(f"   📝 {obj.name} #{obj_id}: Previously interacted but no recent history")
                    else:
                        print(f"   ⭕ {obj.name} #{obj_id}: Never been touched")
            
            # Print periodic statistics if enabled
            if stats_enabled and detector.frame_count % 90 == 0:
                total_interactions = sum(len(obj.interaction_history) for obj in detector.tracked_objects.values())
                alerts_sent = sum(1 for obj in detector.tracked_objects.values() if obj.alert_sent)
                missing_count = sum(1 for obj in detector.tracked_objects.values() if obj.confirmed_missing)
                interacted_count = sum(1 for obj in detector.tracked_objects.values() if obj.interacted)
                
                print(f"📊 PERIODIC STATS - Objects: {len(detector.tracked_objects)}, "
                      f"Touched: {interacted_count}, Interactions: {total_interactions}, "
                      f"Missing: {missing_count}, Alerts: {alerts_sent}")
                
    except KeyboardInterrupt:
        print("\n🛑 System interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"❌ Critical system error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🧹 Cleaning up resources...")
        cap.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        total_interactions = sum(len(obj.interaction_history) for obj in detector.tracked_objects.values())
        alerts_sent = sum(1 for obj in detector.tracked_objects.values() if obj.alert_sent)
        interacted_objects = sum(1 for obj in detector.tracked_objects.values() if obj.interacted)
        
        print(f"\n" + "="*60)
        print(f"📊 FINAL SESSION STATISTICS")
        print(f"="*60)
        print(f"⏱️  Total runtime: {detector.frame_count} frames")
        print(f"📱 Objects detected: {len(detector.tracked_objects)}")
        print(f"👥 Persons detected: {len(detector.tracked_persons)}")
        print(f"✋ Objects touched: {interacted_objects}")
        print(f"📞 Total interactions: {total_interactions}")
        print(f"🚨 Theft alerts sent: {alerts_sent}")
        print(f"="*60)
        print("✅ SYSTEM STOPPED SUCCESSFULLY")

if __name__ == "__main__":
    main()
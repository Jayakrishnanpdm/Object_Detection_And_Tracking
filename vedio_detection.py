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

@dataclass
class Config:
    # Detection thresholds
    detection_confidence: float = 0.4  # Lowered for better object detection
    tracking_iou_threshold: float = 0.4
    interaction_distance: float = 50.0
    
    # More conservative timing parameters to reduce false alarms
    missing_frames_threshold: int = 90  # 3 seconds at 30fps
    consecutive_missing_threshold: int = 60  # 2 seconds
    reappearance_reset_frames: int = 15  # 0.5 seconds
    alert_cooldown_seconds: int = 30  # Increased cooldown
    person_cleanup_frames: int = 120  # 4 seconds
    
    # Hand detection - optimized for performance
    max_hands: int = 2
    hand_confidence: float = 0.5
    hand_detection_interval: int = 2  # Process hands every 2nd frame
    
    # More conservative interaction parameters
    proximity_threshold: float = 0.15
    hand_person_distance_threshold: float = 150.0
    interaction_confidence_threshold: float = 0.3  # Raised threshold
    min_interaction_duration: float = 1.0  # Longer minimum duration
    hand_object_distance_threshold: float = 80.0  # More reasonable distance
    
    # Video processing optimization
    frame_skip: int = 1  # Process more frames for better detection
    resize_factor: float = 0.9  # Less aggressive resizing

@dataclass
class HandInteraction:
    person_id: int
    timestamp: float
    confidence: float
    hand_position: Tuple[int, int]
    duration: float = 0.0
    interaction_type: str = "touch"

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
    confidence_score: float = 0.0
    detection_count: int = 0  # Track how many times detected

@dataclass
class TrackedPerson:
    bbox: List[float]
    last_seen_frame: int = 0
    is_authorized: bool = False
    person_id: Optional[str] = None
    center_history: List[Tuple[int, int]] = field(default_factory=list)
    velocity: Tuple[float, float] = (0.0, 0.0)
    stability_score: float = 0.0

class OptimizedTheftDetector:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        
        # Initialize models with optimized settings
        print("Loading YOLO model...")
        self.model = YOLO("yolov8s.pt")  # Using small version for better accuracy
        print("Initializing DeepSort tracker...")
        self.tracker = DeepSort(max_age=50, n_init=3)  # More conservative tracking
        
        # Initialize MediaPipe with performance optimizations
        print("Initializing MediaPipe hands...")
        mp_hands = mp.solutions.hands
        self.hands_detector = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.config.max_hands,
            min_detection_confidence=self.config.hand_confidence,
            min_tracking_confidence=0.5,
            model_complexity=1  # Better accuracy
        )
        
        # Tracking data
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.tracked_persons: Dict[int, TrackedPerson] = {}
        self.last_alert_time: Dict[int, float] = {}
        self.frame_count = 0
        self.processed_frame_count = 0
        self.fps_counter = []
        
        # Performance optimization
        self.last_hand_detection_frame = 0
        self.cached_hand_data = []
        
        # Enhanced valuable objects list
        self.valuable_objects = {
            'laptop', 'cell phone', 'handbag', 'backpack', 'suitcase',
            'book', 'clock', 'vase', 'scissors', 'remote', 'mouse',
            'keyboard', 'cup', 'bottle', 'wine glass', 'tv', 'tablet',
            'camera', 'watch', 'wallet', 'purse', 'briefcase', 'bag'
        }
        
        # Video processing state
        self.video_fps = 30
        self.original_size = None
        self.resize_size = None
        
        print(f"System initialized. Monitoring {len(self.valuable_objects)} object types")

    def optimize_frame_for_processing(self, frame: np.ndarray) -> np.ndarray:
        """Optimize frame size for faster processing"""
        if self.original_size is None:
            self.original_size = (frame.shape[1], frame.shape[0])
            new_width = int(frame.shape[1] * self.config.resize_factor)
            new_height = int(frame.shape[0] * self.config.resize_factor)
            self.resize_size = (new_width, new_height)
            print(f"Optimizing video processing: {self.original_size} -> {self.resize_size}")
        
        return cv2.resize(frame, self.resize_size)

    def scale_coordinates_back(self, coords: List[float]) -> List[float]:
        """Scale coordinates back to original frame size"""
        if self.original_size is None or self.resize_size is None:
            return coords
        
        scale_x = self.original_size[0] / self.resize_size[0]
        scale_y = self.original_size[1] / self.resize_size[1]
        
        if len(coords) == 4:
            return [coords[0] * scale_x, coords[1] * scale_y, 
                   coords[2] * scale_x, coords[3] * scale_y]
        return coords

    def detect_hands_optimized(self, frame: np.ndarray) -> List[Tuple[Tuple[int, int], Optional[int], float]]:
        """Optimized hand detection with caching"""
        # Use cached results if within detection interval
        if (self.frame_count - self.last_hand_detection_frame < self.config.hand_detection_interval and 
            self.cached_hand_data):
            return self.cached_hand_data
        
        hand_data = []
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            hand_results = self.hands_detector.process(rgb_frame)
            
            if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
                for hand_landmarks, handedness in zip(
                    hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                    
                    # Get hand center with confidence
                    hand_points = []
                    for landmark in hand_landmarks.landmark:
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        hand_points.append((x, y))
                    
                    if hand_points:
                        hand_center = (
                            int(np.mean([p[0] for p in hand_points])),
                            int(np.mean([p[1] for p in hand_points]))
                        )
                        
                        hand_confidence = handedness.classification[0].score
                        
                        # Associate with person
                        associated_person, association_confidence = self.associate_hand_with_person(
                            hand_center, hand_confidence)
                        
                        if association_confidence > 0.2:
                            hand_data.append((hand_center, associated_person, association_confidence))
        
        except Exception as e:
            print(f"Hand detection error: {e}")
            hand_data = self.cached_hand_data
        
        # Update cache
        self.cached_hand_data = hand_data
        self.last_hand_detection_frame = self.frame_count
        
        return hand_data

    def associate_hand_with_person(self, hand_point: Tuple[int, int], 
                                 hand_confidence: float) -> Tuple[Optional[int], float]:
        """Improved hand-person association"""
        min_distance = float('inf')
        closest_person_id = None
        association_confidence = 0.0
        
        hand_x, hand_y = hand_point
        
        for person_id, person in self.tracked_persons.items():
            bbox = person.bbox
            
            # Expanded bounding box for better association
            margin = 60
            expanded_bbox = [bbox[0] - margin, bbox[1] - margin, 
                           bbox[2] + margin, bbox[3] + margin]
            
            # Check if hand is within expanded person area
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
                distance_factor = 1.0 - (distance / self.config.hand_person_distance_threshold)
                association_confidence = max(0.4, distance_factor * hand_confidence * person.stability_score)
        
        return closest_person_id, association_confidence

    def get_detections_optimized(self, frame: np.ndarray) -> Tuple[List, List[str]]:
        """Optimized detection with better filtering"""
        # Use slightly optimized frame size for better performance
        process_frame = self.optimize_frame_for_processing(frame)
        
        results = self.model(process_frame, verbose=False, conf=self.config.detection_confidence)
        detections, detection_names = [], []

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                name = self.model.names[cls_id]
                bbox = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                
                # Scale bbox back to original size
                scaled_bbox = [
                    bbox[0] * (frame.shape[1] / process_frame.shape[1]),
                    bbox[1] * (frame.shape[0] / process_frame.shape[0]),
                    bbox[2] * (frame.shape[1] / process_frame.shape[1]),
                    bbox[3] * (frame.shape[0] / process_frame.shape[0])
                ]
                
                # Convert to x,y,w,h format for DeepSort
                detection_bbox = [
                    float(scaled_bbox[0]), 
                    float(scaled_bbox[1]),
                    float(scaled_bbox[2] - scaled_bbox[0]), 
                    float(scaled_bbox[3] - scaled_bbox[1])
                ]
                
                detections.append((detection_bbox, conf))
                detection_names.append(name)

        return detections, detection_names

    def detect_hand_object_interaction(self, hand_data: List[Tuple[Tuple[int, int], Optional[int], float]], 
                                     obj_bbox: List[float], obj_id: int) -> Optional[HandInteraction]:
        """Enhanced interaction detection"""
        current_time = time.time()
        best_interaction = None
        highest_confidence = 0.0
        
        for (hand_point, person_id, association_confidence) in hand_data:
            if person_id is None or association_confidence < self.config.interaction_confidence_threshold:
                continue
            
            # Calculate hand-object distance
            hand_object_distance = self.calculate_point_to_bbox_distance(hand_point, obj_bbox)
            
            if hand_object_distance <= self.config.hand_object_distance_threshold:
                # Enhanced confidence calculation
                distance_confidence = max(0.3, 1.0 - (hand_object_distance / self.config.hand_object_distance_threshold))
                total_confidence = association_confidence * distance_confidence
                
                # Boost confidence for very close interactions
                if hand_object_distance < 25:
                    total_confidence *= 1.3
                
                if total_confidence > highest_confidence:
                    highest_confidence = total_confidence
                    best_interaction = HandInteraction(
                        person_id=person_id,
                        timestamp=current_time,
                        confidence=total_confidence,
                        hand_position=hand_point,
                        interaction_type="direct_touch" if hand_object_distance < 25 else "proximity"
                    )
        
        # Calculate interaction duration
        if best_interaction and obj_id in self.tracked_objects:
            obj = self.tracked_objects[obj_id]
            recent_interactions = [
                i for i in obj.interaction_history 
                if (i.person_id == best_interaction.person_id and 
                    current_time - i.timestamp <= 3.0)
            ]
            if recent_interactions:
                best_interaction.duration = current_time - recent_interactions[0].timestamp
        
        return best_interaction

    def calculate_point_to_bbox_distance(self, point: Tuple[int, int], bbox: List[float]) -> float:
        """Calculate minimum distance from point to bounding box"""
        px, py = point
        
        # Handle different bbox formats
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
        else:
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
        
        # Point inside bbox
        if x1 <= px <= x2 and y1 <= py <= y2:
            return 0.0
        
        # Calculate distance to nearest edge
        dx = max(x1 - px, 0, px - x2)
        dy = max(y1 - py, 0, py - y2)
        return math.sqrt(dx*dx + dy*dy)

    def is_valid_theft_scenario(self, obj: TrackedObject) -> Tuple[bool, str]:
        """More conservative theft validation"""
        current_time = time.time()
        
        print(f"THEFT CHECK for {obj.name} (#{hash(obj) % 1000}):")
        print(f"  - Detection count: {obj.detection_count}")
        print(f"  - Interacted: {obj.interacted} ({len(obj.interaction_history)} interactions)")
        print(f"  - Missing frames: {obj.missing_frames}/{self.config.missing_frames_threshold}")
        print(f"  - Consecutive missing: {obj.consecutive_missing_frames}/{self.config.consecutive_missing_threshold}")
        print(f"  - Stability: {obj.stability_counter}")
        
        # Must be well-established object (detected multiple times)
        if obj.detection_count < 10:
            return False, f"Object not well established (detected {obj.detection_count} times)"
        
        # Must have meaningful interaction
        if not obj.interacted or len(obj.interaction_history) == 0:
            return False, "No interaction detected"
        
        # Must be missing for longer consecutive frames
        if obj.consecutive_missing_frames < self.config.consecutive_missing_threshold:
            return False, f"Only missing {obj.consecutive_missing_frames} consecutive frames"
        
        # Must be missing for total frames
        if obj.missing_frames < self.config.missing_frames_threshold:
            return False, f"Total missing frames insufficient: {obj.missing_frames}"
        
        # Check for substantial interaction (longer duration or multiple touches)
        meaningful_interactions = [
            i for i in obj.interaction_history 
            if (current_time - i.timestamp <= 15.0 and 
                (i.confidence >= 0.4 or i.duration >= self.config.min_interaction_duration))
        ]
        
        if len(meaningful_interactions) < 2:
            return False, "Insufficient meaningful interactions"
        
        # Alert not already sent
        if obj.alert_sent:
            return False, "Alert already sent"
        
        print(f"  ✓ THEFT CONFIRMED for {obj.name}")
        return True, "All criteria met"

    def get_theft_suspect(self, obj: TrackedObject) -> Tuple[Optional[int], Dict]:
        """Get most likely theft suspect with evidence"""
        current_time = time.time()
        
        # Get recent high-quality interactions
        recent_interactions = [
            i for i in obj.interaction_history
            if (current_time - i.timestamp <= 12.0 and i.confidence >= 0.3)
        ]
        
        if not recent_interactions:
            return None, {}
        
        # Score persons by interaction evidence
        person_scores = defaultdict(lambda: {
            'total_confidence': 0.0, 'interaction_count': 0, 
            'total_duration': 0.0, 'max_confidence': 0.0,
            'last_interaction': 0.0, 'direct_touches': 0
        })
        
        for interaction in recent_interactions:
            pid = interaction.person_id
            evidence = person_scores[pid]
            
            # Time-weighted confidence
            time_weight = max(0.4, 1.0 - ((current_time - interaction.timestamp) / 12.0))
            weighted_confidence = interaction.confidence * time_weight
            
            evidence['total_confidence'] += weighted_confidence
            evidence['interaction_count'] += 1
            evidence['max_confidence'] = max(evidence['max_confidence'], interaction.confidence)
            evidence['last_interaction'] = max(evidence['last_interaction'], interaction.timestamp)
            
            if interaction.interaction_type == "direct_touch":
                evidence['direct_touches'] += 1
            
            evidence['total_duration'] += interaction.duration
        
        # Find best suspect
        best_suspect = None
        best_score = 0.0
        best_evidence = {}
        
        for person_id, evidence in person_scores.items():
            # Enhanced scoring
            base_score = evidence['total_confidence'] * math.log(evidence['interaction_count'] + 1)
            touch_bonus = evidence['direct_touches'] * 0.8
            duration_factor = min(evidence['total_duration'] + 1, 4.0)
            
            final_score = base_score * duration_factor + touch_bonus
            
            if final_score > best_score:
                best_score = final_score
                best_suspect = person_id
                best_evidence = dict(evidence)
                best_evidence['suspicion_score'] = final_score
        
        return best_suspect, best_evidence

    def send_theft_alert(self, frame: np.ndarray, person_bbox: List[float], 
                        obj_bbox: List[float], person_id: int, object_name: str,
                        evidence: Dict) -> bool:
        """Send theft alert"""
        current_time = time.time()
        
        print(f"\n{'='*50}")
        print(f"THEFT ALERT - {object_name.upper()} STOLEN!")
        print(f"{'='*50}")
        print(f"Suspect: Person {person_id}")
        print(f"Time: {time.strftime('%H:%M:%S', time.localtime(current_time))}")
        print(f"Evidence Score: {evidence.get('suspicion_score', 0):.2f}")
        print(f"Interactions: {evidence.get('interaction_count', 0)}")
        print(f"Direct Touches: {evidence.get('direct_touches', 0)}")
        print(f"Max Confidence: {evidence.get('max_confidence', 0):.2f}")
        
        # Save alert frame
        alert_frame = self.draw_alert_visualization(
            frame.copy(), person_bbox, obj_bbox, person_id, object_name, evidence
        )
        
        timestamp = int(current_time)
        image_path = f"THEFT_ALERT_{timestamp}_{object_name.replace(' ', '_')}_person_{person_id}.jpg"
        cv2.imwrite(image_path, alert_frame)
        print(f"Alert image saved: {image_path}")
        print(f"{'='*50}")
        return True

    def create_alert_message(self, object_name: str, person_id: int, evidence: Dict) -> str:
        """Create detailed alert message"""
        return f"""
THEFT ALERT - {object_name.upper()}

Suspect: Person ID {person_id}
Time: {time.strftime('%Y-%m-%d %H:%M:%S')}
Suspicion Score: {evidence.get('suspicion_score', 0):.2f}
Total Interactions: {evidence.get('interaction_count', 0)}
Direct Touches: {evidence.get('direct_touches', 0)}
Max Confidence: {evidence.get('max_confidence', 0):.2f}

IMMEDIATE INVESTIGATION REQUIRED
        """.strip()

    def draw_alert_visualization(self, frame: np.ndarray, person_bbox: List[float], 
                               object_bbox: List[float], person_id: int, 
                               object_name: str, evidence: Dict) -> np.ndarray:
        """Draw theft alert visualization with proper coordinate handling"""
        h, w = frame.shape[:2]
        
        try:
            # Draw suspect with thick red box
            p_x1, p_y1, p_x2, p_y2 = map(int, person_bbox)
            # Ensure coordinates are within bounds
            p_x1 = max(0, min(p_x1, w-1))
            p_y1 = max(0, min(p_y1, h-1))
            p_x2 = max(p_x1+1, min(p_x2, w))
            p_y2 = max(p_y1+1, min(p_y2, h))
            
            cv2.rectangle(frame, (p_x1-3, p_y1-3), (p_x2+3, p_y2+3), (0, 0, 255), 4)
            cv2.putText(frame, f"THIEF - P{person_id}", (p_x1, max(p_y1-10, 20)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        except Exception as e:
            print(f"Error drawing person bbox in alert: {e}")
        
        try:
            # Draw stolen object location (last known position)
            o_x1, o_y1, o_x2, o_y2 = map(int, object_bbox)
            # Ensure coordinates are within bounds
            o_x1 = max(0, min(o_x1, w-1))
            o_y1 = max(0, min(o_y1, h-1))
            o_x2 = max(o_x1+1, min(o_x2, w))
            o_y2 = max(o_y1+1, min(o_y2, h))
            
            cv2.rectangle(frame, (o_x1-2, o_y1-2), (o_x2+2, o_y2+2), (0, 255, 255), 3)
            cv2.putText(frame, f"STOLEN: {object_name.upper()}", (o_x1, max(o_y1-10, 20)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        except Exception as e:
            print(f"Error drawing object bbox in alert: {e}")
        
        try:
            # Alert banner
            cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 255), -1)
            cv2.putText(frame, "THEFT DETECTED", (max(w//2-120, 10), 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, time.strftime('%H:%M:%S'), (max(w//2-50, 10), 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        except Exception as e:
            print(f"Error drawing alert banner: {e}")
        
        return frame

    def update_object_tracking(self, track_id: int, detection: Dict, hand_data: List):
        """Enhanced object tracking"""
        current_time = time.time()
        
        if track_id not in self.tracked_objects:
            # New object
            self.tracked_objects[track_id] = TrackedObject(
                name=detection["name"],
                bbox=detection["bbox"],
                initial_position=list(detection["bbox"]),
                last_seen_timestamp=current_time,
                stability_counter=1,
                confidence_score=detection.get("confidence", 0.5),
                detection_count=1
            )
            print(f"NEW OBJECT: {detection['name']} #{track_id}")
        else:
            obj = self.tracked_objects[track_id]
            obj.detection_count += 1  # Increment detection count
            
            # Handle reappearance
            if obj.consecutive_missing_frames > 0:
                print(f"OBJECT REAPPEARED: {obj.name} #{track_id}")
                obj.reappearance_frames += 1
                if obj.reappearance_frames >= self.config.reappearance_reset_frames:
                    obj.missing_frames = 0
                    obj.consecutive_missing_frames = 0
                    obj.confirmed_missing = False
                    obj.was_missing = True
                    # Don't reset alert_sent immediately
            
            # Update object state
            obj.missing_frames = 0
            obj.consecutive_missing_frames = 0
            obj.confirmed_missing = False
            obj.last_seen_timestamp = current_time
            obj.stability_counter = min(obj.stability_counter + 1, 15)
            obj.confidence_score = max(obj.confidence_score * 0.9, detection.get("confidence", 0.5))
            
            # Check for movement
            if obj.initial_position and not obj.moved_significantly:
                movement = self.calculate_movement(obj.initial_position, detection["bbox"])
                if movement > 50 and obj.stability_counter > 5:
                    obj.moved_significantly = True
                    print(f"OBJECT MOVED: {obj.name} #{track_id} moved {movement:.1f}px")
            
            obj.bbox = detection["bbox"]
        
        # Detect interactions
        interaction = self.detect_hand_object_interaction(hand_data, detection["bbox"], track_id)
        
        if interaction and track_id in self.tracked_objects:
            obj = self.tracked_objects[track_id]
            
            # Add interaction with deduplication
            should_add = True
            if obj.interaction_history:
                last = obj.interaction_history[-1]
                if (interaction.person_id == last.person_id and 
                    current_time - last.timestamp < 0.5):
                    last.duration = current_time - last.timestamp
                    should_add = False
            
            if should_add:
                obj.interaction_history.append(interaction)
                if not obj.interacted:
                    print(f"FIRST TOUCH: {obj.name} #{track_id} by Person {interaction.person_id} "
                          f"(conf: {interaction.confidence:.2f})")
                    obj.interacted = True
                
            # Clean old interactions (keep longer history)
            obj.interaction_history = [
                i for i in obj.interaction_history 
                if current_time - i.timestamp <= 20.0
            ]
            
            obj.interact_timestamp = current_time
            obj.last_interaction_person = interaction.person_id

    def calculate_movement(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate movement between two bounding boxes"""
        center1 = [(bbox1[0] + bbox1[2]/2), (bbox1[1] + bbox1[3]/2)]
        center2 = [(bbox2[0] + bbox2[2]/2), (bbox2[1] + bbox2[3]/2)]
        return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

    def handle_missing_objects(self, current_detections: Dict, hand_data: List, frame: np.ndarray):
        """Handle missing objects with conservative theft detection"""
        current_time = time.time()
        objects_to_remove = []
        
        for obj_id, obj in self.tracked_objects.items():
            if obj_id not in current_detections:
                obj.missing_frames += 1
                obj.consecutive_missing_frames += 1
                obj.reappearance_frames = 0
                
                # Status updates
                if obj.consecutive_missing_frames == 1:
                    print(f"OBJECT MISSING: {obj.name} #{obj_id}")
                elif obj.consecutive_missing_frames == self.config.consecutive_missing_threshold:
                    obj.confirmed_missing = True
                    print(f"CONFIRMED MISSING: {obj.name} #{obj_id} "
                          f"({obj.consecutive_missing_frames} frames)")
                
                # More conservative theft detection
                if (obj.stability_counter > 5 and obj.detection_count > 15 and
                    obj.consecutive_missing_frames >= self.config.consecutive_missing_threshold):
                    
                    is_theft, reason = self.is_valid_theft_scenario(obj)
                    
                    if is_theft and self.should_send_alert(obj_id):
                        suspect_id, evidence = self.get_theft_suspect(obj)
                        
                        if (suspect_id and suspect_id in self.tracked_persons and
                            evidence.get('suspicion_score', 0) > 2.0):  # Minimum suspicion threshold
                            print(f"THEFT DETECTED: {obj.name} #{obj_id} by Person {suspect_id}")
                            
                            alert_sent = self.send_theft_alert(
                                frame, self.tracked_persons[suspect_id].bbox,
                                obj.bbox, suspect_id, obj.name, evidence
                            )
                            
                            if alert_sent:
                                obj.alert_sent = True
                                self.last_alert_time[obj_id] = current_time
                    else:
                        if obj.consecutive_missing_frames % 60 == 0:  # Log every 2 seconds
                            print(f"NOT THEFT: {obj.name} #{obj_id} - {reason}")
                
                # Remove very old objects
                if obj.missing_frames > 300:  # 10 seconds
                    objects_to_remove.append(obj_id)
                    print(f"REMOVING OLD OBJECT: {obj.name} #{obj_id}")
        
        # Clean up old objects
        for obj_id in objects_to_remove:
            if obj_id in self.tracked_objects:
                del self.tracked_objects[obj_id]
            if obj_id in self.last_alert_time:
                del self.last_alert_time[obj_id]

    def should_send_alert(self, object_id: int) -> bool:
        """Check if alert should be sent (cooldown)"""
        current_time = time.time()
        if object_id in self.last_alert_time:
            time_diff = current_time - self.last_alert_time[object_id]
            if time_diff < self.config.alert_cooldown_seconds:
                return False
        return True

    def update_person_tracking(self, track_id: int, detection: Dict):
        """Update person tracking"""
        if track_id not in self.tracked_persons:
            self.tracked_persons[track_id] = TrackedPerson(
                bbox=detection["bbox"],
                last_seen_frame=self.frame_count,
                stability_score=0.5
            )
            print(f"NEW PERSON: #{track_id}")
        else:
            person = self.tracked_persons[track_id]
            person.bbox = detection["bbox"]
            person.last_seen_frame = self.frame_count
            person.stability_score = min(person.stability_score + 0.1, 1.0)

    def cleanup_old_tracks(self):
        """Clean up old tracking data"""
        current_frame = self.frame_count
        
        # Remove old persons
        persons_to_remove = [
            pid for pid, person in self.tracked_persons.items()
            if current_frame - person.last_seen_frame > self.config.person_cleanup_frames
        ]
        
        for pid in persons_to_remove:
            del self.tracked_persons[pid]
            if pid in self.last_alert_time:
                del self.last_alert_time[pid]

    def iou(self, boxA: List[float], boxB: List[float]) -> float:
        """Calculate Intersection over Union"""
        # Convert to x1,y1,x2,y2 format
        if len(boxA) == 4 and len(boxA) == 4:
            # Handle x,y,w,h format
            if boxA[2] < boxA[0] + 50:  # Likely w,h format
                boxA = [boxA[0], boxA[1], boxA[0] + boxA[2], boxA[1] + boxA[3]]
            if boxB[2] < boxB[0] + 50:  # Likely w,h format
                boxB = [boxB[0], boxB[1], boxB[0] + boxB[2], boxB[1] + boxB[3]]
            
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        unionArea = boxAArea + boxBArea - interArea
        
        return interArea / unionArea if unionArea > 0 else 0

    def draw_annotations(self, frame: np.ndarray, detections: Dict, hand_data: List) -> np.ndarray:
        """Fixed draw comprehensive annotations with proper bbox handling"""
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw detections
        for track_id, detection in detections.items():
            try:
                bbox = detection["bbox"]
                name = detection["name"]
                
                # Convert bbox to proper format
                if len(bbox) != 4:
                    continue
                
                # Handle different bbox formats more robustly
                x1, y1, x2, y2 = map(float, bbox)
                
                # Check if bbox is in x,y,w,h format and convert to x1,y1,x2,y2
                if x2 <= x1 or y2 <= y1:  # Likely x,y,w,h format
                    w, h_box = x2, y2
                    x2, y2 = x1 + w, y1 + h_box
                
                # Convert to integers
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Ensure bbox is within frame bounds
                x1 = max(0, min(x1, w-1))
                y1 = max(0, min(y1, h-1))
                x2 = max(x1+1, min(x2, w))
                y2 = max(y1+1, min(y2, h))
                
                # Skip if bbox is too small
                if x2 - x1 < 5 or y2 - y1 < 5:
                    continue
                    
            except (ValueError, TypeError, IndexError) as e:
                print(f"Bbox conversion error for track {track_id}: {e}")
                continue
                
                if name == "person":
                    color = (0, 255, 0)
                    label = f"Person {track_id}"
                else:
                    # Color coding for objects
                    if track_id in self.tracked_objects:
                        obj = self.tracked_objects[track_id]
                        if obj.alert_sent:
                            color = (0, 0, 255)  # Red - theft detected
                        elif obj.confirmed_missing:
                            color = (0, 100, 255)  # Orange - missing
                        elif obj.interacted:
                            color = (0, 255, 255)  # Yellow - touched
                        else:
                            color = (0, 255, 0)  # Green - normal
                        
                        # Status indicators
                        status = []
                        if obj.interacted:
                            status.append(f"TOUCHED({len(obj.interaction_history)})")
                        if obj.moved_significantly:
                            status.append("MOVED")
                        if obj.confirmed_missing:
                            status.append("MISSING")
                        if obj.alert_sent:
                            status.append("THEFT_ALERT")
                        
                        label = f"{name} #{track_id}"
                        if status:
                            label += f" [{'/'.join(status[:2])}]"  # Limit status length
                    else:
                        color = (128, 128, 128)
                        label = f"{name} #{track_id}"
                
                # Draw bounding box - use tuple format for OpenCV
                thickness = 3 if name in self.valuable_objects else 2
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                
                # Draw label with background
                try:
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    label_bg_y1 = max(0, y1 - 25)
                    label_bg_y2 = max(label_bg_y1 + 20, y1)
                    label_bg_x2 = min(x1 + label_size[0] + 5, w)
                    
                    cv2.rectangle(annotated_frame, (x1, label_bg_y1), (label_bg_x2, label_bg_y2), color, -1)
                    cv2.putText(annotated_frame, label, (x1 + 2, y1 - 8), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                except Exception as label_error:
                    # Fallback - just draw simple label
                    cv2.putText(annotated_frame, f"{name}_{track_id}", (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw hand points
        for (hand_point, person_id, confidence) in hand_data:
            try:
                if confidence > 0.2:
                    radius = max(3, int(confidence * 10))
                    color = (0, 255, 0) if person_id else (100, 100, 255)
                    
                    # Ensure hand_point coordinates are valid
                    hx, hy = int(hand_point[0]), int(hand_point[1])
                    hx = max(0, min(hx, w-1))
                    hy = max(0, min(hy, h-1))
                    
                    cv2.circle(annotated_frame, (hx, hy), radius, color, -1)
                    
                    if person_id is not None:
                        cv2.putText(annotated_frame, f"P{person_id}", 
                                   (hx + 8, hy - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            except Exception as hand_error:
                continue  # Skip problematic hand points
        
        # Status panel
        panel_height = 120
        try:
            cv2.rectangle(annotated_frame, (0, h - panel_height), (w, h), (0, 0, 0), -1)
            
            # Statistics
            avg_fps = np.mean(self.fps_counter) if self.fps_counter else 0
            total_interactions = sum(len(obj.interaction_history) for obj in self.tracked_objects.values())
            alerts_sent = sum(1 for obj in self.tracked_objects.values() if obj.alert_sent)
            
            stats_text = [
                f"Objects: {len(self.tracked_objects)} | Persons: {len(self.tracked_persons)} | FPS: {avg_fps:.1f}",
                f"Hands: {len([h for h in hand_data if h[2] > 0.2])} | Interactions: {total_interactions} | Alerts: {alerts_sent}",
                f"Frame: {self.frame_count} | Processed: {self.processed_frame_count}",
                "ESC: Exit | R: Reset | D: Debug | SPACE: Pause"
            ]
            
            for i, text in enumerate(stats_text):
                y_pos = h - panel_height + 20 + (i * 25)
                cv2.putText(annotated_frame, text, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        except Exception as panel_error:
            pass  # Skip panel if there's an error
        
        return annotated_frame

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Optimized main frame processing function"""
        start_time = time.time()
        self.frame_count += 1
        
        # Skip frames for performance (but less aggressive)
        if self.frame_count % (self.config.frame_skip + 1) != 0:
            return self.draw_annotations(frame, {}, self.cached_hand_data)
        
        self.processed_frame_count += 1
        
        try:
            # Get detections and hand data
            hand_data = self.detect_hands_optimized(frame)
            detections, detection_names = self.get_detections_optimized(frame)
            
            # Update tracker
            tracks = self.tracker.update_tracks(detections, frame=frame)
            
            # Match tracks to detections
            current_detections = {}
            for track in tracks:
                if not track.is_confirmed():
                    continue
                    
                track_id = track.track_id
                ltrb = track.to_ltrb()
                
                # Find best matching detection
                best_match = None
                best_iou = 0
                for i, (det_bbox, conf) in enumerate(detections):
                    x, y, w, h = det_bbox
                    det_ltrb = [x, y, x + w, y + h]
                    iou_score = self.iou(ltrb, det_ltrb)
                    if iou_score > best_iou and iou_score > 0.2:  # Lower threshold
                        best_iou = iou_score
                        best_match = i
                
                if best_match is not None:
                    current_detections[track_id] = {
                        "name": detection_names[best_match],
                        "bbox": ltrb,
                        "confidence": detections[best_match][1]
                    }
            
            # Update tracking
            for track_id, detection in current_detections.items():
                if detection["name"] == "person":
                    self.update_person_tracking(track_id, detection)
                elif detection["name"] in self.valuable_objects:
                    self.update_object_tracking(track_id, detection, hand_data)
            
            # Handle missing objects
            self.handle_missing_objects(current_detections, hand_data, frame)
            
            # Cleanup every 5 seconds
            if self.frame_count % 150 == 0:
                self.cleanup_old_tracks()
            
            # Performance tracking
            process_time = time.time() - start_time
            if process_time > 0:
                fps = 1.0 / process_time
                self.fps_counter.append(fps)
                if len(self.fps_counter) > 30:
                    self.fps_counter.pop(0)
            
            return self.draw_annotations(frame, current_detections, hand_data)
            
        except Exception as e:
            print(f"Frame processing error: {e}")
            return self.draw_annotations(frame, {}, self.cached_hand_data)


def main():
    """Main function for video-based theft detection"""
    print("="*80)
    print("FIXED VIDEO-BASED THEFT DETECTION SYSTEM")
    print("="*80)
    
    # Configuration
    config = Config()
    print(f"Configuration:")
    print(f"  - Detection confidence: {config.detection_confidence}")
    print(f"  - Missing threshold: {config.missing_frames_threshold} frames")
    print(f"  - Hand detection interval: {config.hand_detection_interval} frames")
    print(f"  - Frame skip: {config.frame_skip}")
    print(f"  - Resize factor: {config.resize_factor}")
    
    # Initialize detector
    detector = OptimizedTheftDetector(config)
    
    # Open video file
    video_path = "input_vedio1.mp4"  # Make sure this file exists
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{video_path}'")
        print("Please ensure the file 'input_video.mp4' exists in the current directory")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video loaded: {width}x{height}, {fps:.1f} FPS, {duration:.1f}s ({total_frames} frames)")
    
    detector.video_fps = fps
    
    print("\nSystem ready. Controls:")
    print("  ESC: Exit")
    print("  SPACE: Pause/Resume")
    print("  R: Reset tracking")
    print("  D: Debug info")
    print("  S: Skip 10 seconds forward")
    print("  B: Go back 10 seconds")
    print("="*80)
    
    paused = False
    frame_number = 0
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("End of video reached")
                    break
                
                frame_number += 1
            
                # Process frame
                processed_frame = detector.process_frame(frame)
                
                # Add progress bar
                progress = frame_number / total_frames
                bar_width = 400
                bar_height = 10
                bar_x = (processed_frame.shape[1] - bar_width) // 2
                bar_y = processed_frame.shape[0] - 30
                
                # Background
                cv2.rectangle(processed_frame, (bar_x, bar_y), 
                            (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
                # Progress
                cv2.rectangle(processed_frame, (bar_x, bar_y), 
                            (bar_x + int(bar_width * progress), bar_y + bar_height), (0, 255, 0), -1)
                
                # Time info
                current_time = frame_number / fps if fps > 0 else 0
                time_text = f"{current_time/60:.0f}:{current_time%60:04.1f} / {duration/60:.0f}:{duration%60:04.1f}"
                cv2.putText(processed_frame, time_text, (bar_x, bar_y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow("Fixed Theft Detection - Video Analysis", processed_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord(' '):  # SPACE - pause/resume
                paused = not paused
                print("PAUSED" if paused else "RESUMED")
            elif key == ord('r') or key == ord('R'):  # Reset
                detector.tracked_objects.clear()
                detector.tracked_persons.clear()
                detector.last_alert_time.clear()
                detector.frame_count = 0
                detector.processed_frame_count = 0
                print("TRACKING RESET")
            elif key == ord('d') or key == ord('D'):  # Debug
                print(f"\nDEBUG INFO (Frame {frame_number}):")
                print(f"  Objects: {len(detector.tracked_objects)}")
                print(f"  Persons: {len(detector.tracked_persons)}")
                print(f"  Processed frames: {detector.processed_frame_count}")
                for obj_id, obj in detector.tracked_objects.items():
                    status = []
                    if obj.interacted: status.append("TOUCHED")
                    if obj.confirmed_missing: status.append("MISSING")
                    if obj.alert_sent: status.append("THEFT_ALERT")
                    print(f"    {obj.name} #{obj_id}: {'/'.join(status) or 'NORMAL'} (detected {obj.detection_count} times)")
            elif key == ord('s') or key == ord('S'):  # Skip forward
                skip_frames = int(10 * fps)  # 10 seconds
                for _ in range(skip_frames):
                    ret, _ = cap.read()
                    if not ret:
                        break
                    frame_number += 1
                print(f"SKIPPED 10 seconds (frame {frame_number})")
            elif key == ord('b') or key == ord('B'):  # Go back
                back_frames = int(10 * fps)  # 10 seconds
                new_frame = max(0, frame_number - back_frames)
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
                frame_number = new_frame
                print(f"WENT BACK 10 seconds (frame {frame_number})")
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Final stats
        total_interactions = sum(len(obj.interaction_history) for obj in detector.tracked_objects.values())
        alerts_sent = sum(1 for obj in detector.tracked_objects.values() if obj.alert_sent)
        
        print(f"\n{'='*60}")
        print("FINAL STATISTICS")
        print(f"{'='*60}")
        print(f"Total frames processed: {detector.processed_frame_count}/{frame_number}")
        print(f"Objects detected: {len(detector.tracked_objects)}")
        print(f"Total interactions: {total_interactions}")
        print(f"Theft alerts: {alerts_sent}")
        print("System stopped successfully")


if __name__ == "__main__":
    main()
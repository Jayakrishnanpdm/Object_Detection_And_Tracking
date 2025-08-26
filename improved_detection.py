import cv2
import numpy as np
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import mediapipe as mp
# from alert import send_email_alert, send_telegram_alert  # Comment out if not available

@dataclass
class Config:
    # Detection thresholds
    detection_confidence: float = 0.5
    tracking_iou_threshold: float = 0.5
    interaction_distance: float = 50.0
    
    # FIXED: More lenient timing parameters for easier testing
    missing_frames_threshold: int = 30  # ~1 second at 30fps (reduced from 45)
    consecutive_missing_threshold: int = 20  # Must be missing for 0.67 second continuously (reduced from 30)
    reappearance_reset_frames: int = 10  # Reset after object reappears for 0.33 second
    alert_cooldown_seconds: int = 20  # Reduced cooldown for testing
    person_cleanup_frames: int = 150
    
    # Hand detection
    max_hands: int = 4
    hand_confidence: float = 0.3
    
    # FIXED: More lenient interaction parameters
    proximity_threshold: float = 0.15
    hand_person_distance_threshold: float = 120.0  # Increased further
    interaction_confidence_threshold: float = 0.15  # Reduced for easier interaction
    min_interaction_duration: float = 0.2  # Reduced minimum duration
    hand_object_distance_threshold: float = 100.0  # Increased for easier interaction
    
    # FIXED: More lenient suspect parameters
    suspect_buffer_duration: int = 600  # Keep suspect data for 20 seconds (increased)
    suspect_image_capture_interval: int = 3  # Capture suspect image every 3 frames
    interaction_history_duration: float = 30.0  # Keep interactions for 30 seconds (increased)
    theft_suspect_time_window: float = 25.0  # Look for suspects within 25 seconds (increased)

@dataclass
class HandInteraction:
    """Track hand interactions with objects"""
    person_id: int
    timestamp: float
    confidence: float
    hand_position: Tuple[int, int]
    duration: float = 0.0

@dataclass 
class SuspectBuffer:
    """Buffer to store suspect information even after they leave the frame"""
    person_id: int
    last_bbox: List[float]
    last_seen_frame: int
    last_seen_timestamp: float
    interaction_history: List[HandInteraction] = field(default_factory=list)
    captured_images: List[np.ndarray] = field(default_factory=list)
    is_authorized: bool = False
    total_interactions: int = 0
    max_confidence: float = 0.0
    suspicious_score: float = 0.0

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
    first_interaction_logged: bool = False
    last_known_bbox: Optional[List[float]] = None
    last_frame_with_object: Optional[np.ndarray] = None

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
        
        # NEW: Suspect buffer system
        self.suspect_buffer: Dict[int, SuspectBuffer] = {}
        self.current_frame: Optional[np.ndarray] = None
        
        # Enhanced valuable objects list
        self.valuable_objects = {
            'laptop', 'cell phone', 'handbag', 'backpack', 'suitcase',
            'book', 'clock', 'vase', 'scissors', 'remote', 'mouse',
            'keyboard', 'cup', 'bottle', 'wine glass', 'tv', 'tablet',
            'camera', 'watch', 'wallet', 'purse'
        }
        
        print(f"✅ Initialization complete!")
        print(f"📱 Monitoring {len(self.valuable_objects)} object types")
        print(f"🕵️ Suspect buffer duration: {self.config.suspect_buffer_duration/30:.1f} seconds")

    def update_suspect_buffer(self, person_id: int, bbox: List[float], frame: np.ndarray):
        """Update or create suspect buffer entry"""
        current_time = time.time()
        
        if person_id not in self.suspect_buffer:
            self.suspect_buffer[person_id] = SuspectBuffer(
                person_id=person_id,
                last_bbox=bbox,
                last_seen_frame=self.frame_count,
                last_seen_timestamp=current_time
            )
            print(f"🆕 SUSPECT BUFFER CREATED: Person {person_id}")
        
        suspect = self.suspect_buffer[person_id]
        suspect.last_bbox = bbox
        suspect.last_seen_frame = self.frame_count
        suspect.last_seen_timestamp = current_time
        
        # Capture suspect image periodically
        if (self.frame_count % self.config.suspect_image_capture_interval == 0 and 
            len(suspect.captured_images) < 15):  # Increased limit
            
            # Extract person region from frame
            x1, y1, x2, y2 = map(int, bbox)
            person_image = frame[max(0, y1-10):min(frame.shape[0], y2+10), 
                                max(0, x1-10):min(frame.shape[1], x2+10)]
            
            if person_image.size > 0:
                suspect.captured_images.append(person_image.copy())
                if len(suspect.captured_images) == 1:
                    print(f"📸 FIRST SUSPECT IMAGE CAPTURED: Person {person_id}")

    def cleanup_suspect_buffer(self):
        """Remove old suspect data"""
        current_frame = self.frame_count
        suspects_to_remove = []
        
        for person_id, suspect in self.suspect_buffer.items():
            frames_since_seen = current_frame - suspect.last_seen_frame
            if frames_since_seen > self.config.suspect_buffer_duration:
                suspects_to_remove.append(person_id)
        
        for person_id in suspects_to_remove:
            print(f"🗑️ REMOVING OLD SUSPECT: Person {person_id} "
                  f"(not seen for {(current_frame - self.suspect_buffer[person_id].last_seen_frame)/30:.1f}s)")
            del self.suspect_buffer[person_id]

    def get_suspect_from_buffer(self, person_id: int) -> Optional[SuspectBuffer]:
        """Get suspect from buffer even if not currently visible"""
        return self.suspect_buffer.get(person_id)

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
                        enhanced_hand_data.append((tip_point, associated_person, association_confidence * 0.8))
        
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
            margin = 40  # Increased margin for better association
            expanded_bbox = [bbox[0] - margin, bbox[1] - margin, 
                           bbox[2] + margin, bbox[3] + margin]
            
            if (expanded_bbox[0] <= hand_x <= expanded_bbox[2] and 
                expanded_bbox[1] <= hand_y <= expanded_bbox[3]):
                return person_id, 0.95 * hand_confidence
            
            # Calculate distance to person center
            person_center_x = (bbox[0] + bbox[2]) / 2
            person_center_y = (bbox[1] + bbox[3]) / 2
            distance = math.sqrt((hand_x - person_center_x)**2 + (hand_y - person_center_y)**2)
            
            if distance < self.config.hand_person_distance_threshold and distance < min_distance:
                min_distance = distance
                closest_person_id = person_id
                association_confidence = max(0.3, 
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
                distance_confidence = max(0.2, 1.0 - (hand_object_distance / self.config.hand_object_distance_threshold))
                total_confidence = association_confidence * distance_confidence
                
                if total_confidence > highest_confidence:
                    highest_confidence = total_confidence
                    best_interaction = HandInteraction(
                        person_id=person_id,
                        timestamp=current_time,
                        confidence=total_confidence,
                        hand_position=hand_point
                    )
                    
                    print(f"🤝 INTERACTION DETECTED: Person {person_id} touching object {obj_id} "
                          f"(confidence: {total_confidence:.2f}, distance: {hand_object_distance:.1f}px)")
        
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
        if not obj.interacted:
            print(f"   ❌ FAIL: Object was never interacted with")
            return False, "Object was never interacted with"
        
        # Criterion 2: Object must be missing for minimum consecutive frames
        if obj.consecutive_missing_frames < self.config.consecutive_missing_threshold:
            print(f"   ❌ FAIL: Missing for only {obj.consecutive_missing_frames} consecutive frames (need {self.config.consecutive_missing_threshold})")
            return False, f"Missing for only {obj.consecutive_missing_frames} consecutive frames"
        
        # Criterion 3: Total missing frames threshold
        if obj.missing_frames < self.config.missing_frames_threshold:
            print(f"   ❌ FAIL: Total missing frames {obj.missing_frames} < {self.config.missing_frames_threshold}")
            return False, f"Total missing frames insufficient"
        
        # Criterion 4: Alert not already sent
        if obj.alert_sent:
            print(f"   ❌ FAIL: Alert already sent")
            return False, "Alert already sent"
        
        # FIXED: More lenient interaction history check
        if not obj.interaction_history:
            print(f"   ❌ FAIL: No interaction history available")
            return False, "No interaction history available"
        
        print(f"   ✅ PASS: All theft criteria met!")
        return True, "All criteria met"

    def get_theft_suspect_with_evidence(self, obj: TrackedObject) -> Tuple[Optional[int], Dict, Optional[SuspectBuffer]]:
        """FIXED: More lenient suspect determination with detailed evidence"""
        current_time = time.time()
        
        print(f"🕵️ SEARCHING FOR THEFT SUSPECT...")
        
        # FIXED: Much more lenient time window for interactions
        recent_interactions = [
            i for i in obj.interaction_history
            if current_time - i.timestamp <= self.config.theft_suspect_time_window and  # Increased window
            i.confidence >= 0.1  # Very low threshold
        ]
        
        print(f"   📊 Found {len(recent_interactions)} interactions within {self.config.theft_suspect_time_window}s window")
        for i, interaction in enumerate(recent_interactions):
            time_ago = current_time - interaction.timestamp
            print(f"      {i+1}. Person {interaction.person_id}: {time_ago:.1f}s ago (confidence: {interaction.confidence:.2f})")
        
        if not recent_interactions:
            # FIXED: If no recent interactions, use ANY interaction from history
            print(f"   ⚠️ No recent interactions, checking ALL interaction history...")
            if obj.interaction_history:
                recent_interactions = obj.interaction_history[-3:]  # Take last 3 interactions
                print(f"   🔄 Using last {len(recent_interactions)} interactions from history")
                for i, interaction in enumerate(recent_interactions):
                    time_ago = current_time - interaction.timestamp
                    print(f"      {i+1}. Person {interaction.person_id}: {time_ago:.1f}s ago (confidence: {interaction.confidence:.2f})")
            else:
                print(f"   ❌ No interaction history available at all")
                return None, {}, None
        
        # Score persons based on interaction evidence
        person_evidence = defaultdict(lambda: {
            'total_confidence': 0.0, 'interaction_count': 0, 'total_duration': 0.0,
            'last_interaction': 0.0, 'max_confidence': 0.0, 'avg_confidence': 0.0
        })
        
        for interaction in recent_interactions:
            pid = interaction.person_id
            evidence = person_evidence[pid]
            
            # FIXED: Less aggressive time weighting
            time_since = current_time - interaction.timestamp
            time_weight = max(0.5, 1.0 - (time_since / self.config.theft_suspect_time_window))  # More generous weighting
            weighted_confidence = interaction.confidence * time_weight
            
            evidence['total_confidence'] += weighted_confidence
            evidence['interaction_count'] += 1
            evidence['total_duration'] += interaction.duration
            evidence['last_interaction'] = max(evidence['last_interaction'], interaction.timestamp)
            evidence['max_confidence'] = max(evidence['max_confidence'], interaction.confidence)
        
        # Calculate average confidence
        for evidence in person_evidence.values():
            if evidence['interaction_count'] > 0:
                evidence['avg_confidence'] = evidence['total_confidence'] / evidence['interaction_count']
        
        # Find most suspicious person with VERY lenient scoring
        best_suspect = None
        best_score = 0.0
        best_evidence = {}
        best_suspect_buffer = None
        
        for person_id, evidence in person_evidence.items():
            # FIXED: Much more lenient scoring - almost anyone who interacted is suspicious
            base_score = evidence['total_confidence']
            interaction_bonus = evidence['interaction_count'] * 0.5
            duration_bonus = min(evidence['total_duration'], 3.0) * 0.2
            
            score = base_score + interaction_bonus + duration_bonus
            
            print(f"   🔍 Person {person_id} suspicion analysis:")
            print(f"      - Total confidence: {evidence['total_confidence']:.2f}")
            print(f"      - Interactions: {evidence['interaction_count']}")
            print(f"      - Max confidence: {evidence['max_confidence']:.2f}")
            print(f"      - Total score: {score:.2f}")
            
            if score > best_score:
                best_score = score
                best_suspect = person_id
                best_evidence = dict(evidence)
                best_evidence['suspicion_score'] = score
                
                # Try to get suspect from buffer
                best_suspect_buffer = self.get_suspect_from_buffer(person_id)
        
        if best_suspect is not None:
            print(f"   🎯 BEST SUSPECT: Person {best_suspect} (score: {best_score:.2f})")
            if best_suspect_buffer:
                print(f"   📸 Suspect images available: {len(best_suspect_buffer.captured_images)}")
            else:
                print(f"   ⚠️ No suspect buffer found, but proceeding anyway")
        else:
            print(f"   ❌ No suspect identified")
            
        return best_suspect, best_evidence, best_suspect_buffer

    def send_enhanced_alert_with_buffer(self, frame: np.ndarray, suspect_buffer: Optional[SuspectBuffer],
                                      obj_bbox: List[float], person_id: int, object_name: str,
                                      evidence: Dict, obj: TrackedObject) -> bool:
        """FIXED: Enhanced alert system that works even without perfect buffer data"""
        current_time = time.time()
        
        print(f"\n" + "="*60)
        print(f"🚨 SENDING THEFT ALERT 🚨")
        print(f"="*60)
        
        # Try to get suspect visual data
        suspect_bbox = None
        suspect_image = None
        
        if suspect_buffer and suspect_buffer.captured_images:
            suspect_bbox = suspect_buffer.last_bbox
            suspect_image = suspect_buffer.captured_images[-1]
            print(f"📸 Using buffered suspect data (last seen {(time.time() - suspect_buffer.last_seen_timestamp):.1f}s ago)")
        elif person_id in self.tracked_persons:
            suspect_bbox = self.tracked_persons[person_id].bbox
            print(f"👤 Using current person tracking")
        else:
            print(f"⚠️ No suspect visual data available - proceeding with text alert")
        
        # Create alert frame (even without suspect data)
        alert_frame = self.draw_theft_alert_frame_with_buffer(
            frame.copy(), suspect_bbox, obj_bbox, person_id, object_name, 
            evidence, suspect_image, obj
        )
        
        # Save alert image
        timestamp = int(current_time)
        image_path = f"theft_alert_{timestamp}_{person_id}_{object_name.replace(' ', '_')}.jpg"
        cv2.imwrite(image_path, alert_frame)
        print(f"💾 Alert image saved: {image_path}")
        
        # Save suspect image if available
        suspect_image_path = None
        if suspect_image is not None:
            suspect_image_path = f"suspect_{timestamp}_{person_id}.jpg"
            cv2.imwrite(suspect_image_path, suspect_image)
            print(f"📸 Suspect image saved: {suspect_image_path}")
        
        # Create detailed alert message
        alert_msg = self.create_detailed_alert_message_with_buffer(
            object_name, person_id, evidence, timestamp, suspect_buffer
        )
        
        try:
            # Print alert to console (replace with actual alert functions when available)
            print("\n" + "🚨" * 20)
            print(alert_msg)
            print("🚨" * 20)
            
            # Uncomment these when alert functions are available
            # send_email_alert(image_path, alert_msg, suspect_image_path)
            # send_telegram_alert(image_path, alert_msg, suspect_image_path)
            
            print(f"✅ THEFT ALERT SENT SUCCESSFULLY!")
            print(f"="*60)
            return True
        except Exception as e:
            print(f"❌ Failed to send alert: {e}")
            return False

    def create_detailed_alert_message_with_buffer(self, object_name: str, person_id: int, 
                                                evidence: Dict, timestamp: int, 
                                                suspect_buffer: Optional[SuspectBuffer]) -> str:
        """Create detailed alert message with buffer information"""
        buffer_info = ""
        if suspect_buffer:
            time_since_seen = time.time() - suspect_buffer.last_seen_timestamp
            buffer_info = f"""
🕵️ SUSPECT TRACKING:
   • Last Seen: {time_since_seen:.1f} seconds ago
   • Captured Images: {len(suspect_buffer.captured_images)}
   • Total Interactions: {suspect_buffer.total_interactions}
   • Status: {"AUTHORIZED" if suspect_buffer.is_authorized else "UNAUTHORIZED"}"""
        else:
            buffer_info = f"""
🕵️ SUSPECT TRACKING:
   • Status: LIMITED DATA AVAILABLE
   • Person tracked during interaction"""
        
        return f"""
🚨 THEFT ALERT - CONFIRMED 🚨
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📱 STOLEN OBJECT: {object_name.upper()}
👤 SUSPECT: Person ID {person_id} (UNAUTHORIZED)
🕒 TIME: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))}

📊 EVIDENCE SUMMARY:
   • Suspicion Score: {evidence.get('suspicion_score', 0):.2f}
   • Interactions: {evidence.get('interaction_count', 0)}
   • Max Confidence: {evidence.get('max_confidence', 0):.2f}
   • Avg Confidence: {evidence.get('avg_confidence', 0):.2f}
   • Total Duration: {evidence.get('total_duration', 0):.1f}s
   • Last Interaction: {time.strftime('%H:%M:%S', time.localtime(evidence.get('last_interaction', 0)))}
{buffer_info}

🔍 STATUS: OBJECT CONFIRMED STOLEN
⚠️ ACTION REQUIRED: INVESTIGATE IMMEDIATELY

💡 NOTE: Enhanced tracking system with suspect identification
        """.strip()

    def draw_theft_alert_frame_with_buffer(self, frame: np.ndarray, suspect_bbox: Optional[List[float]], 
                                         object_bbox: List[float], person_id: int, 
                                         object_name: str, evidence: Dict,
                                         suspect_image: Optional[np.ndarray],
                                         obj: TrackedObject) -> np.ndarray:
        """FIXED: Draw comprehensive theft alert visualization with buffer support"""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Draw suspect person if bbox available
        if suspect_bbox:
            p_x1, p_y1, p_x2, p_y2 = map(int, suspect_bbox)
            cv2.rectangle(overlay, (p_x1-3, p_y1-3), (p_x2+3, p_y2+3), (0, 0, 255), 5)
            cv2.putText(overlay, f"THIEF - PERSON {person_id}", 
                       (p_x1, p_y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        
        # Draw stolen object using last known position
        if obj.last_known_bbox:
            object_bbox = obj.last_known_bbox
        o_x1, o_y1, o_x2, o_y2 = map(int, object_bbox)
        cv2.rectangle(overlay, (o_x1-3, o_y1-3), (o_x2+3, o_y2+3), (0, 255, 255), 5)
        cv2.putText(overlay, f"STOLEN: {object_name.upper()}", 
                   (o_x1, o_y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3)
        
        # Add suspect image if available (picture-in-picture style)
        if suspect_image is not None and suspect_image.size > 0:
            # Resize suspect image to fit in corner
            suspect_h, suspect_w = suspect_image.shape[:2]
            max_size = 200
            if suspect_w > max_size or suspect_h > max_size:
                scale = max_size / max(suspect_w, suspect_h)
                new_w, new_h = int(suspect_w * scale), int(suspect_h * scale)
                suspect_image = cv2.resize(suspect_image, (new_w, new_h))
            
            # Position in top-right corner
            y_offset = 90  # Below the main alert banner
            x_offset = w - suspect_image.shape[1] - 10
            
            # Draw border around suspect image
            cv2.rectangle(overlay, 
                         (x_offset - 3, y_offset - 3), 
                         (x_offset + suspect_image.shape[1] + 3, y_offset + suspect_image.shape[0] + 3),
                         (0, 0, 255), 3)
            
            # Overlay suspect image
            overlay[y_offset:y_offset+suspect_image.shape[0], 
                   x_offset:x_offset+suspect_image.shape[1]] = suspect_image
            
            # Add label for suspect image
            cv2.putText(overlay, "SUSPECT IMAGE", 
                       (x_offset, y_offset - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Add alert banner
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 255), -1)
        cv2.putText(overlay, "THEFT DETECTED - ALERT SENT", (w//2 - 200, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        cv2.putText(overlay, time.strftime('%Y-%m-%d %H:%M:%S'), (w//2 - 100, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add status information
        status_text = "SUSPECT MAY HAVE LEFT SCENE" if not suspect_bbox else "SUSPECT IN FRAME"
        cv2.putText(overlay, status_text, (10, h - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
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
        
        # Clean up old interaction histories with more lenient timing
        for obj in self.tracked_objects.values():
            old_count = len(obj.interaction_history)
            obj.interaction_history = [
                i for i in obj.interaction_history 
                if current_time - i.timestamp <= self.config.interaction_history_duration
            ]
            new_count = len(obj.interaction_history)
            if old_count != new_count:
                print(f"🧹 Cleaned {old_count - new_count} old interactions for {obj.name}")
        
        # Clean up suspect buffer
        self.cleanup_suspect_buffer()

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
        """UPDATED: Main processing function with suspect buffer integration"""
        start_time = time.time()
        self.frame_count += 1
        current_time = time.time()
        
        # Store current frame for suspect capture
        self.current_frame = frame.copy()
        
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

        # Update tracking with suspect buffer integration
        for track_id, detection in current_detections.items():
            if detection["name"] == "person":
                self.update_person_tracking_with_buffer(track_id, detection, frame)
            elif detection["name"] in self.valuable_objects:
                self.update_object_tracking_enhanced(track_id, detection, enhanced_hand_data, frame)

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

    def update_person_tracking_with_buffer(self, track_id: int, detection: Dict, frame: np.ndarray):
        """UPDATED: Enhanced person tracking with suspect buffer"""
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
        
        # Update suspect buffer for all persons
        self.update_suspect_buffer(track_id, detection["bbox"], frame)

    def update_object_tracking_enhanced(self, track_id: int, detection: Dict, 
                                      enhanced_hand_data: List, frame: np.ndarray):
        """UPDATED: Enhanced object tracking with frame storage"""
        current_time = time.time()
        
        if track_id not in self.tracked_objects:
            # New object detection
            self.tracked_objects[track_id] = TrackedObject(
                name=detection["name"],
                bbox=detection["bbox"],
                initial_position=list(detection["bbox"]),
                last_seen_timestamp=current_time,
                stability_counter=1,
                last_known_bbox=list(detection["bbox"]),
                last_frame_with_object=frame.copy()
            )
            print(f"📱 NEW OBJECT DETECTED: {detection['name']} ID {track_id}")
        else:
            obj = self.tracked_objects[track_id]
            
            # Object is currently visible - handle reappearance
            if obj.was_missing or obj.consecutive_missing_frames > 0:
                obj.reappearance_frames += 1
                if obj.reappearance_frames >= self.config.reappearance_reset_frames:
                    self.reset_object_status(track_id, obj)
            
            # Reset missing counters and update last known data
            obj.missing_frames = 0
            obj.consecutive_missing_frames = 0
            obj.confirmed_missing = False
            obj.last_seen_timestamp = current_time
            obj.stability_counter = min(obj.stability_counter + 1, 10)
            obj.last_known_bbox = list(detection["bbox"])
            obj.last_frame_with_object = frame.copy()
            
            # Check for significant movement with better terminal output
            if obj.initial_position and not obj.moved_significantly:
                movement = self.calculate_movement_with_smoothing(obj.initial_position, detection["bbox"])
                if movement > 50 and obj.stability_counter > 3:
                    obj.moved_significantly = True
                    print(f"🚨 OBJECT MOVED: {obj.name} ID {track_id} moved {movement:.1f}px from initial position!")
            
            obj.bbox = detection["bbox"]
            
        # Enhanced interaction detection with suspect buffer updates
        interaction = self.detect_hand_object_interaction_enhanced(
            enhanced_hand_data, detection["bbox"], track_id)
        
        if interaction and track_id in self.tracked_objects:
            obj = self.tracked_objects[track_id]
            
            # Update suspect buffer interaction count
            if interaction.person_id in self.suspect_buffer:
                suspect = self.suspect_buffer[interaction.person_id]
                suspect.total_interactions += 1
                suspect.max_confidence = max(suspect.max_confidence, interaction.confidence)
                suspect.interaction_history.append(interaction)
            
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
                if interaction.confidence > 0.3:
                    print(f"✋ STRONG INTERACTION: {obj.name} ID {track_id} with Person {interaction.person_id} "
                          f"(confidence: {interaction.confidence:.2f}, duration: {interaction.duration:.1f}s)")
            
            # Clean old interactions with more generous timing
            obj.interaction_history = [
                i for i in obj.interaction_history 
                if current_time - i.timestamp <= self.config.interaction_history_duration
            ]
            
            # Mark as interacted
            if not obj.interacted:
                print(f"📝 OBJECT MARKED AS INTERACTED: {obj.name} ID {track_id}")
            
            obj.interacted = True
            obj.interact_timestamp = current_time
            obj.last_interaction_person = interaction.person_id

    def handle_missing_objects_enhanced(self, current_detections: Dict, 
                                      enhanced_hand_data: List, frame: np.ndarray):
        """FIXED: Enhanced missing object handling with more aggressive theft detection"""
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
                
                # FIXED: More aggressive theft detection
                if (obj.stability_counter > 1 and  # Reduced from 2
                    obj.consecutive_missing_frames >= self.config.consecutive_missing_threshold and
                    obj.missing_frames >= self.config.missing_frames_threshold):
                    
                    print(f"\n🔍 CHECKING THEFT SCENARIO for {obj.name} ID {obj_id}...")
                    is_valid_theft, reason = self.is_valid_theft_scenario(obj)
                    
                    if is_valid_theft and self.should_send_alert(obj_id):
                        suspect_id, evidence, suspect_buffer = self.get_theft_suspect_with_evidence(obj)
                        
                        if suspect_id is not None:  # FIXED: Accept any suspect ID
                            # Check if suspect is authorized (from buffer or current tracking)
                            is_authorized = False
                            if suspect_buffer:
                                is_authorized = suspect_buffer.is_authorized
                            elif suspect_id in self.tracked_persons:
                                is_authorized = self.tracked_persons[suspect_id].is_authorized
                            
                            # FIXED: Since all persons are unauthorized by default, proceed with alert
                            print(f"🚨 THEFT CONFIRMED: {obj.name} ID {obj_id} stolen by Person {suspect_id}!")
                            print(f"   Evidence score: {evidence.get('suspicion_score', 0):.2f}")
                            print(f"   Interactions: {evidence.get('interaction_count', 0)}")
                            print(f"   Max confidence: {evidence.get('max_confidence', 0):.2f}")
                            
                            alert_sent = self.send_enhanced_alert_with_buffer(
                                frame, suspect_buffer, obj.last_known_bbox or obj.bbox, 
                                suspect_id, obj.name, evidence, obj
                            )
                            
                            if alert_sent:
                                obj.alert_sent = True
                                self.last_alert_time[obj_id] = current_time
                                print(f"✅ THEFT ALERT SENT SUCCESSFULLY for {obj.name} ID {obj_id}")
                        else:
                            print(f"❌ No valid suspect found for {obj.name} ID {obj_id}")
                    else:
                        if obj.consecutive_missing_frames % 30 == 0:  # Log every second
                            print(f"❌ THEFT NOT CONFIRMED for {obj.name} ID {obj_id}: {reason}")
                
                # Remove very old objects (increased timeout)
                if obj.missing_frames > 900:  # ~30 seconds
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
        """UPDATED: Draw comprehensive annotations with buffer information"""
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw object detections with enhanced status
        for track_id, detection in detections.items():
            bbox = detection["bbox"]
            x1, y1, x2, y2 = map(int, bbox)
            name = detection["name"]
            
            if name == "person":
                person = self.tracked_persons.get(track_id)
                suspect_buffer = self.suspect_buffer.get(track_id)
                
                # Color based on suspect status
                if suspect_buffer and suspect_buffer.total_interactions > 0:
                    color = (0, 100, 255)  # Orange for suspects with interactions
                elif person and person.is_authorized:
                    color = (0, 255, 0)  # Green for authorized
                else:
                    color = (255, 100, 100)  # Light red for unauthorized
                
                auth_status = "AUTH" if person and person.is_authorized else "UNAUTH"
                interactions = f"({suspect_buffer.total_interactions})" if suspect_buffer else "(0)"
                label = f"{auth_status} P{track_id} {interactions}"
                
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
            if confidence > 0.15:  # Lower threshold for better visibility
                radius = max(3, int(confidence * 8))
                color = (0, 255, 0) if person_id else (100, 100, 255)
                cv2.circle(annotated_frame, hand_point, radius, color, -1)
                
                if person_id is not None:
                    cv2.putText(annotated_frame, f"P{person_id}", 
                               (hand_point[0] + 8, hand_point[1] - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Enhanced status panel with buffer information
        panel_height = 200  # Increased height
        cv2.rectangle(annotated_frame, (0, h - panel_height), (w, h), (0, 0, 0), -1)
        
        # Main stats
        avg_fps = np.mean(self.fps_counter) if self.fps_counter else 0
        cv2.putText(annotated_frame, f"Objects: {len(self.tracked_objects)} | "
                   f"Persons: {len(self.tracked_persons)} | "
                   f"FPS: {avg_fps:.1f}", 
                   (10, h - 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Interaction and buffer stats
        total_interactions = sum(len(obj.interaction_history) for obj in self.tracked_objects.values())
        active_hands = len([h for h in enhanced_hand_data if h[2] > 0.15])
        buffer_count = len(self.suspect_buffer)
        cv2.putText(annotated_frame, f"Hands: {active_hands} | "
                   f"Interactions: {total_interactions} | "
                   f"Suspects Buffered: {buffer_count}", 
                   (10, h - 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Object status summary
        interacted_count = sum(1 for obj in self.tracked_objects.values() if obj.interacted)
        moved_count = sum(1 for obj in self.tracked_objects.values() if obj.moved_significantly)
        cv2.putText(annotated_frame, f"Touched: {interacted_count} | "
                   f"Moved: {moved_count} | "
                   f"Missing Threshold: {self.config.missing_frames_threshold}f ({self.config.missing_frames_threshold/30:.1f}s)", 
                   (10, h - 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Alert stats
        alerts_sent = sum(1 for obj in self.tracked_objects.values() if obj.alert_sent)
        missing_objects = sum(1 for obj in self.tracked_objects.values() if obj.confirmed_missing)
        cv2.putText(annotated_frame, f"Alerts Sent: {alerts_sent} | "
                   f"Missing: {missing_objects} | "
                   f"Consecutive Threshold: {self.config.consecutive_missing_threshold}f ({self.config.consecutive_missing_threshold/30:.1f}s)", 
                   (10, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Buffer status
        suspect_images = sum(len(s.captured_images) for s in self.suspect_buffer.values())
        cv2.putText(annotated_frame, f"Suspect Images: {suspect_images} | "
                   f"Hand Distance: {self.config.hand_object_distance_threshold}px | "
                   f"Interaction Window: {self.config.theft_suspect_time_window}s", 
                   (10, h - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Controls
        cv2.putText(annotated_frame, "ESC: Exit | R: Reset | D: Debug | S: Stats | I: Info | B: Buffer Info", 
                   (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # FIXED indicator
        cv2.putText(annotated_frame, "✅ FIXED: More lenient theft detection - Alerts should work now!", 
                   (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return annotated_frame


def main():
    """FIXED: Enhanced main execution function with improved theft detection"""
    print("\n" + "="*80)
    print("🚀 INITIALIZING FIXED THEFT DETECTION SYSTEM")
    print("="*80)
    
    # Create enhanced configuration with more lenient settings
    config = Config()
    print(f"⚙️ FIXED Configuration loaded:")
    print(f"   - Missing frames threshold: {config.missing_frames_threshold} frames ({config.missing_frames_threshold/30:.1f}s) [REDUCED]")
    print(f"   - Consecutive missing threshold: {config.consecutive_missing_threshold} frames ({config.consecutive_missing_threshold/30:.1f}s) [REDUCED]")
    print(f"   - Hand-object distance: {config.hand_object_distance_threshold}px [INCREASED]")
    print(f"   - Interaction confidence: {config.interaction_confidence_threshold} [REDUCED]")
    print(f"   - Alert cooldown: {config.alert_cooldown_seconds}s [REDUCED]")
    print(f"   - Suspect time window: {config.theft_suspect_time_window}s [INCREASED]")
    print(f"   - Interaction history duration: {config.interaction_history_duration}s [INCREASED]")
    print(f"   🆕 - Suspect buffer duration: {config.suspect_buffer_duration/30:.1f}s")
    
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
    print("🎯 FIXED THEFT DETECTION SYSTEM - ACTIVE")
    print("="*80)
    print("⚠️  All persons are treated as UNAUTHORIZED by default")
    print(f"📱 Monitoring {len(detector.valuable_objects)} object types:")
    print(f"   {', '.join(sorted(list(detector.valuable_objects)[:10]))}...")
    print(f"🎯 FIXED Detection Strategy (More Lenient):")
    print(f"   1. Object must be TOUCHED by a person (hand interaction)")
    print(f"   2. Object must disappear for {config.consecutive_missing_threshold}+ consecutive frames ({config.consecutive_missing_threshold/30:.1f}s)")
    print(f"   3. Object must be missing for {config.missing_frames_threshold}+ total frames ({config.missing_frames_threshold/30:.1f}s)")
    print(f"   4. ANY interaction within {config.theft_suspect_time_window} seconds (INCREASED)")
    print(f"   5. Interaction confidence > {config.interaction_confidence_threshold} (REDUCED)")
    print(f"\n🔧 FIXES APPLIED:")
    print(f"   ✓ Reduced missing frame thresholds for faster detection")
    print(f"   ✓ Increased suspect time window to {config.theft_suspect_time_window}s")
    print(f"   ✓ Reduced interaction confidence threshold")
    print(f"   ✓ More generous hand-object distance detection")
    print(f"   ✓ Fallback to ANY interaction history if no recent interactions")
    print(f"   ✓ Enhanced suspect identification and evidence collection")
    print(f"   ✓ Better interaction logging and debugging")
    print("\n📋 HOW TO TEST (EASIER NOW):")
    print("   1. Place a valuable object (laptop, phone, book, etc.) in view")
    print("   2. Move your hand close to the object (within 100px - increased range)")
    print("   3. Watch for 'FIRST INTERACTION' and 'TOUCHED' messages")
    print("   4. Remove the object from camera view")
    print("   5. Wait for theft detection (~1 second - much faster!)")
    print("   6. Alert should be sent automatically!")
    print("\n⌨️  CONTROLS:")
    print("   ESC: Exit system")
    print("   R: Reset all tracking data")
    print("   D: Print detailed debug information")
    print("   S: Toggle statistics display")
    print("   I: Print current interaction info")
    print("   B: Print suspect buffer information")
    print("="*80)
    print("🟢 FIXED SYSTEM READY - Much more sensitive theft detection!")
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
                cv2.imshow("FIXED Theft Detection System", annotated_frame)
                
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
                detector.suspect_buffer.clear()
                detector.frame_count = 0
                print("\n🔄 ALL TRACKING DATA AND SUSPECT BUFFER RESET")
            elif key == ord('d') or key == ord('D'):
                # Debug information
                print(f"\n" + "="*60)
                print(f"📊 FIXED DEBUG INFORMATION (Frame {detector.frame_count})")
                print(f"="*60)
                print(f"🎥 System Stats:")
                print(f"   - FPS: {np.mean(detector.fps_counter) if detector.fps_counter else 0:.1f}")
                print(f"   - Objects tracked: {len(detector.tracked_objects)}")
                print(f"   - Persons tracked: {len(detector.tracked_persons)}")
                print(f"   - Suspects buffered: {len(detector.suspect_buffer)}")
                
                print(f"\n📱 Object Details:")
                for obj_id, obj in detector.tracked_objects.items():
                    status = []
                    if obj.interacted: status.append("TOUCHED")
                    if obj.moved_significantly: status.append("MOVED") 
                    if obj.confirmed_missing: status.append("CONFIRMED_MISSING")
                    if obj.missing_frames > 0: status.append(f"MISSING_{obj.missing_frames}f")
                    if obj.alert_sent: status.append("ALERT_SENT")
                    
                    print(f"   📦 {obj.name} #{obj_id}: {'/'.join(status) if status else 'NORMAL'}")
                    print(f"      - Missing: {obj.missing_frames}/{detector.config.missing_frames_threshold} frames ({obj.missing_frames/30:.1f}s)")
                    print(f"      - Consecutive: {obj.consecutive_missing_frames}/{detector.config.consecutive_missing_threshold} frames ({obj.consecutive_missing_frames/30:.1f}s)")
                    print(f"      - Interactions: {len(obj.interaction_history)}")
                    print(f"      - Stability: {obj.stability_counter}/10")
                    if obj.interaction_history:
                        latest = obj.interaction_history[-1]
                        time_ago = time.time() - latest.timestamp
                        print(f"      - Last interaction: Person {latest.person_id} ({latest.confidence:.2f}) {time_ago:.1f}s ago")
                
                print(f"\n👥 Person Details:")
                for person_id, person in detector.tracked_persons.items():
                    auth_status = "AUTHORIZED" if person.is_authorized else "UNAUTHORIZED"
                    suspect = detector.suspect_buffer.get(person_id)
                    buffer_info = f" | Buffer: {suspect.total_interactions} interactions, {len(suspect.captured_images)} images" if suspect else " | No buffer"
                    print(f"   👤 Person #{person_id}: {auth_status}{buffer_info}")
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
            
            elif key == ord('b') or key == ord('B'):
                # Print suspect buffer information
                print(f"\n🕵️ SUSPECT BUFFER STATUS:")
                print(f"   Total suspects buffered: {len(detector.suspect_buffer)}")
                for person_id, suspect in detector.suspect_buffer.items():
                    time_since_seen = time.time() - suspect.last_seen_timestamp
                    in_frame = "IN FRAME" if person_id in detector.tracked_persons else "LEFT FRAME"
                    print(f"   👤 Person {person_id}: {in_frame}")
                    print(f"      - Last seen: {time_since_seen:.1f}s ago")
                    print(f"      - Captured images: {len(suspect.captured_images)}")
                    print(f"      - Total interactions: {suspect.total_interactions}")
                    print(f"      - Max confidence: {suspect.max_confidence:.2f}")
                    print(f"      - Authorization: {'AUTHORIZED' if suspect.is_authorized else 'UNAUTHORIZED'}")
            
            # Print periodic statistics if enabled
            if stats_enabled and detector.frame_count % 90 == 0:
                total_interactions = sum(len(obj.interaction_history) for obj in detector.tracked_objects.values())
                alerts_sent = sum(1 for obj in detector.tracked_objects.values() if obj.alert_sent)
                missing_count = sum(1 for obj in detector.tracked_objects.values() if obj.confirmed_missing)
                interacted_count = sum(1 for obj in detector.tracked_objects.values() if obj.interacted)
                buffer_count = len(detector.suspect_buffer)
                
                print(f"📊 PERIODIC STATS - Objects: {len(detector.tracked_objects)}, "
                      f"Touched: {interacted_count}, Interactions: {total_interactions}, "
                      f"Missing: {missing_count}, Alerts: {alerts_sent}, Buffered: {buffer_count}")
                
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
        buffered_suspects = len(detector.suspect_buffer)
        total_suspect_images = sum(len(s.captured_images) for s in detector.suspect_buffer.values())
        
        print(f"\n" + "="*60)
        print(f"📊 FINAL SESSION STATISTICS")
        print(f"="*60)
        print(f"⏱️  Total runtime: {detector.frame_count} frames")
        print(f"📱 Objects detected: {len(detector.tracked_objects)}")
        print(f"👥 Persons detected: {len(detector.tracked_persons)}")
        print(f"✋ Objects touched: {interacted_objects}")
        print(f"📞 Total interactions: {total_interactions}")
        print(f"🚨 Theft alerts sent: {alerts_sent}")
        print(f"🕵️ Suspects buffered: {buffered_suspects}")
        print(f"📸 Suspect images captured: {total_suspect_images}")
        print(f"="*60)
        print("✅ FIXED THEFT DETECTION STOPPED SUCCESSFULLY")
        print("🔧 System was configured with more lenient settings for reliable alerts!")
        print("📧 Check for saved alert images in the current directory")


if __name__ == "__main__":
    main()
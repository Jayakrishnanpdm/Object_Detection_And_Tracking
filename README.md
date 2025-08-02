# 🛡️ YOLO-Based Theft Detection System  

A deep learning-based security system that uses **YOLOv8** for real-time object detection to identify suspicious activities, such as unauthorized file pickups, in restricted areas (e.g., HR cabin, offices, malls).  

The system detects **people and objects (files)** using YOLO, tracks interactions, and sends **alerts** if any unauthorized person picks up a file.

---

## 🚀 Features
✅ Real-time **object detection** using YOLOv8  
✅ Detects **people & sensitive files** in camera footage  
✅ **Tracks people and objects** using DeepSORT (coming soon)  
✅ **Triggers alerts** if a non-authorized person picks up a file  
✅ Optional **email / Telegram notifications** to HR  
✅ Future scope: Face recognition for authorized person detection, action recognition, and full dashboard.

---

## 📂 Folder Structure
yolo-theft-detector/
│── data/ # Store test videos/images
│── models/ # YOLO weights (optional)
│── detect_demo.py # Basic YOLO detection script
│── requirements.txt # Project dependencies
│── README.md # Project details


---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
bash
git clone https://github.com/your-username/yolo-theft-detector.git
cd yolo-theft-detector

### 2️⃣ Create Virtual Environment
python -m venv yolo_env
source yolo_env/bin/activate    # Mac/Linux
yolo_env\Scripts\activate       # Windows

### 3️⃣ Install All Dependencies
pip install -r requirements.txt

### 4️⃣ Test YOLO Detection
python detect_demo.py

# Base Logic
[Frame N]
 ├─ Detect person & object (YOLOv8)
 ├─ Identify HR using face recognition
 ├─ If person != HR AND near object
 │     └─ Mark person as suspicious
 │     └─ Save frame as last_seen_frame
 ├─ If object is missing
 │     └─ Increment object_missing_counter
 │     └─ If object_missing_counter >= 90 AND suspicious_detected:
 │           └─ Send alert with last_seen_frame
 │           └─ Reset counters
 ├─ If object reappears → reset object_missing_counter

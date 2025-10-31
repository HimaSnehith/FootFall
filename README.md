# Footfall Counter using Computer Vision

## 📌 Objective
A computer vision–based system that counts the number of people entering or exiting through a specific area in a video (e.g., doorway, corridor, gate) using object detection and tracking.

---

## 🧠 Approach

### 1. Detection
- Uses **YOLOv8n (Ultralytics)** (`yolov8n.pt`) for human detection (class `0`).
- Detects bounding boxes in each frame.

### 2. Tracking
- Custom **Centroid Tracker** maintains consistent object IDs across frames.
- Computes centroid distance to match detections between frames.
- Handles disappearance & reappearance using time thresholds.

### 3. ROI / Counting Logic
- A virtual **horizontal or vertical line** defines entry/exit.
- When a tracked centroid crosses the line:
  - Moving **towards camera** = Entry  
  - Moving **away from camera** = Exit  
- Maintains total entry and exit counts.

### 4. Visualization
- Overlays bounding boxes, IDs, trajectories, FPS, and counts.
- Live display with OpenCV.
- Saves processed video and screenshots.

---

### 5. Screenshots
<img width="1920" height="1080" alt="Screenshot (20)" src="https://github.com/user-attachments/assets/f77e01eb-76c4-4c1b-9e4e-0af8f7a83d63" />

<img width="1920" height="1080" alt="Screenshot (19)" src="https://github.com/user-attachments/assets/4f4c634d-0a6b-4c63-b5a0-1f7a8aa95ca2" />

<img width="1920" height="1080" alt="Screenshot (18)" src="https://github.com/user-attachments/assets/8ab56fed-3622-4d6e-8326-3d73834f0fac" />

<img width="1920" height="1080" alt="Screenshot (16)" src="https://github.com/user-attachments/assets/e64ddb01-c10b-472b-8746-5ef6688158cd" />


## ⚙️ Dependencies

```bash
pip install opencv-python opencv-contrib-python ultralytics numpy pillow tqdm filterpy scikit-learn scipy

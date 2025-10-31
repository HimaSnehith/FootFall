# Footfall Counter using Computer Vision

## 📌 Objective
A computer vision–based system that counts the number of people entering or exiting through a specific area in a video (e.g., doorway, corridor, gate) using object detection and tracking.

---

## 🧠 Approach

### 1. Detection
- Uses **YOLOv8n (Ultralytics)** pretrained model (`yolov8n.pt`) for human detection (class `0`).
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

## ⚙️ Dependencies

```bash
pip install ultralytics opencv-python numpy scipy

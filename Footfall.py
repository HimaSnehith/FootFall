import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
from scipy.spatial import distance
import time
import os
from datetime import datetime
import traceback

# ==================== CONFIGURATION ====================

class Config:
    """Configuration"""
    # --- Model & Detection ---
    MODEL_NAME = 'yolov8n.pt'
    CONFIDENCE_THRESHOLD = 0.45
    IOU_THRESHOLD = 0.45
    DEVICE = 'cpu' # 'cuda' for NVIDIA, 'mps' for Apple, 'cpu' otherwise

    # --- Fixed UI Canvas Configuration ---
    UI_WIDTH = 1920
    UI_HEIGHT = 1080
    PANEL_HEIGHT = 150

    # --- Tracking ---
    MAX_DISAPPEARED = 30
    MAX_DISTANCE = 100
    TRACK_HISTORY_LENGTH = 30

    # --- Group Handling ---
    ENABLE_GROUP_DETECTION = True
    GROUP_PROXIMITY_THRESHOLD = 80
    GROUP_TRACKING_BOOST = 1.5

    # --- Counting Logic ---
    LINE_POSITION = 0.5
    LINE_ORIENTATION = 'horizontal'
    COUNTING_BUFFER = 20
    ENTRY_SIDE = 'below'

    # --- Performance ---
    FRAME_SKIP = 1

    # --- Visualization ---
    SHOW_BOUNDING_BOXES = True
    SHOW_TRACKING_IDS = True
    SHOW_TRAJECTORIES = True
    SHOW_FPS = True
    COLOR_ENTRY = (0, 255, 0)
    COLOR_EXIT = (0, 0, 255)
    COLOR_LINE = (255, 255, 0)
    COLOR_BOX = (255, 0, 255)
    COLOR_TRAJECTORY = (0, 255, 255)
    COLOR_GROUP = (147, 20, 255)

    # --- Output ---
    SAVE_OUTPUT = True
    OUTPUT_DIR = 'output'


# ==================== CENTROID TRACKER (Your logic) ====================
class CentroidTracker:
    # ... [Your CentroidTracker class code is unchanged and goes here] ...
    def __init__(self, max_disappeared=50, max_distance=100):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.positions = defaultdict(lambda: deque(maxlen=Config.TRACK_HISTORY_LENGTH))
        self.counted = set()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.in_group = {}

    def register(self, centroid):
        self.objects[self.next_object_id] = tuple(int(x) for x in centroid)
        self.disappeared[self.next_object_id] = 0
        self.positions[self.next_object_id].append(tuple(int(x) for x in centroid))
        self.in_group[self.next_object_id] = False
        self.next_object_id += 1

    def deregister(self, object_id):
        keys_to_del = ['objects', 'disappeared', 'positions', 'in_group']
        for key in keys_to_del:
            if object_id in getattr(self, key):
                del getattr(self, key)[object_id]
        self.counted.discard(object_id)

    def detect_groups(self, detections):
        groups = []
        used = set()
        for i, (x1, y1, w1, h1) in enumerate(detections):
            if i in used: continue
            group = [i]
            c1 = (x1 + w1/2, y1 + h1/2)
            for j, (x2, y2, w2, h2) in enumerate(detections):
                if i == j or j in used: continue
                c2 = (x2 + w2/2, y2 + h2/2)
                dist = np.hypot(c1[0] - c2[0], c1[1] - c2[1])
                if dist < Config.GROUP_PROXIMITY_THRESHOLD:
                    group.append(j)
                    used.add(j)
            if len(group) > 1:
                groups.append(group)
                used.update(group)
        return groups

    def update(self, detections):
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return dict(self.objects)

        input_centroids = np.array([(x + w/2, y + h/2) for x, y, w, h in detections], dtype="int")
        
        in_group_indices = set()
        if Config.ENABLE_GROUP_DETECTION:
            groups = self.detect_groups(detections)
            for g in groups: in_group_indices.update(g)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)): self.register(input_centroids[i])
            return dict(self.objects)

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())
        D = distance.cdist(np.array(object_centroids), input_centroids)

        for obj_idx, obj_id in enumerate(object_ids):
            if self.in_group.get(obj_id, False): D[obj_idx] *= 0.7

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        used_rows, used_cols = set(), set()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols: continue
            object_id = object_ids[row]
            effective_max_dist = self.max_distance * Config.GROUP_TRACKING_BOOST if (col in in_group_indices or self.in_group.get(object_id, False)) else self.max_distance
            if D[row, col] > effective_max_dist: continue
            
            self.objects[object_id] = tuple(map(int, input_centroids[col]))
            self.positions[object_id].append(tuple(map(int, input_centroids[col])))
            self.disappeared[object_id] = 0
            self.in_group[object_id] = (col in in_group_indices)
            used_rows.add(row); used_cols.add(col)

        unused_rows = set(range(D.shape[0])) - used_rows
        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared: self.deregister(object_id)
        
        unused_cols = set(range(D.shape[1])) - used_cols
        for col in unused_cols: self.register(input_centroids[col])
        return dict(self.objects)


# ==================== FOOTFALL COUNTER (Refactored Visualization) ====================

class FootfallCounter:
    def __init__(self, video_source=0, config=None):
        self.video_source = video_source
        self.config = config or Config()
        print("Loading YOLO model...")
        self.model = YOLO(self.config.MODEL_NAME)
        # self.model.to(self.config.DEVICE)
        print(f"✓ Model loaded: {self.config.MODEL_NAME}")

        self.tracker = CentroidTracker(self.config.MAX_DISAPPEARED, self.config.MAX_DISTANCE)
        self.entry_count, self.exit_count = 0, 0
        self.crossed_objects = {}
        self.frame_count, self.start_time = 0, time.time()
        self.cap, self.output_writer = None, None
        self.frame_width, self.frame_height, self.fps, self.counting_line_pos = 0, 0, 0, 0
        self.last_detections = []

    def initialize_video(self):
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened(): raise ValueError(f"Could not open: {self.video_source}")
        
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30

        if self.config.LINE_ORIENTATION == 'horizontal':
            self.counting_line_pos = int(self.frame_height * self.config.LINE_POSITION)
        else:
            self.counting_line_pos = int(self.frame_width * self.config.LINE_POSITION)

        print(f"✓ Video initialized: {self.frame_width}x{self.frame_height} @ {self.fps} FPS")

        if self.config.SAVE_OUTPUT:
            os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.config.OUTPUT_DIR, f'footfall_{ts}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.output_writer = cv2.VideoWriter(path, fourcc, self.fps, (self.config.UI_WIDTH, self.config.UI_HEIGHT))
            print(f"✓ Output video: {path}")

    def detect_people(self, frame):
        results = self.model(frame, conf=self.config.CONFIDENCE_THRESHOLD, iou=self.config.IOU_THRESHOLD, classes=[0], verbose=False) # device=self.config.DEVICE
        return [(int(x1), int(y1), int(x2-x1), int(y2-y1)) for r in results for x1, y1, x2, y2 in r.boxes.xyxy.cpu().numpy()]

    def check_line_crossing(self, object_id, centroid):
        cx, cy = centroid
        current_side = ('below' if cy > self.counting_line_pos else 'above') if self.config.LINE_ORIENTATION == 'horizontal' else ('right' if cx > self.counting_line_pos else 'left')

        if object_id in self.crossed_objects and self.crossed_objects[object_id] != current_side:
            if object_id not in self.tracker.counted:
                action = 'entry' if current_side == self.config.ENTRY_SIDE else 'exit'
                if action == 'entry': self.entry_count += 1
                else: self.exit_count += 1
                self.tracker.counted.add(object_id)
                # REFACTORED: Removed the per-crossing print statement
                # print(f"[{datetime.now().strftime('%H:%M:%S')}] ID {object_id} -> {action.upper()}")
        self.crossed_objects[object_id] = current_side

    def draw_visualizations_on_frame(self, frame, detections, tracked_objects):
        # ... [This function is unchanged] ...
        if self.config.LINE_ORIENTATION == 'horizontal':
            cv2.line(frame, (0, self.counting_line_pos), (self.frame_width, self.counting_line_pos), self.config.COLOR_LINE, 3)
        else:
            cv2.line(frame, (self.counting_line_pos, 0), (self.counting_line_pos, self.frame_height), self.config.COLOR_LINE, 3)
        if self.config.SHOW_BOUNDING_BOXES:
            for (x, y, w, h) in detections:
                cv2.rectangle(frame, (x, y), (x + w, y + h), self.config.COLOR_BOX, 2)
        for object_id, centroid in tracked_objects.items():
            cx, cy = int(centroid[0]), int(centroid[1])
            in_group = self.tracker.in_group.get(object_id, False)
            color = self.config.COLOR_GROUP if in_group else self.config.COLOR_TRAJECTORY
            cv2.circle(frame, (cx, cy), 5, color, -1)
            if self.config.SHOW_TRACKING_IDS:
                text = f"ID:{object_id}{' [G]' if in_group else ''}"
                cv2.putText(frame, text, (cx - 20, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            if self.config.SHOW_TRAJECTORIES and object_id in self.tracker.positions:
                positions = list(self.tracker.positions[object_id])
                for i in range(1, len(positions)):
                    cv2.line(frame, tuple(positions[i - 1]), tuple(positions[i]), color, 2)
        return frame


    def create_final_layout(self, annotated_frame):
        canvas = np.zeros((self.config.UI_HEIGHT, self.config.UI_WIDTH, 3), dtype=np.uint8)
        
        # --- REFACTORED: Stats panel now uses your desired text ---
        panel_area = canvas[0:self.config.PANEL_HEIGHT, :]
        panel_area[:] = (40, 40, 40)
        cv2.putText(panel_area, "FOOTFALL COUNTER", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(panel_area, f"Entry / Moving Towards Camera: {self.entry_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.config.COLOR_ENTRY, 2)
        cv2.putText(panel_area, f"Exit / Moving Away from Camera: {self.exit_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.config.COLOR_EXIT, 2)
        
        total = self.entry_count + self.exit_count
        cv2.putText(panel_area, f"TOTAL: {total}", (self.config.UI_WIDTH - 250, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if self.config.SHOW_FPS:
            elapsed_time = time.time() - self.start_time
            current_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            cv2.putText(panel_area, f"FPS: {current_fps:.1f}", (self.config.UI_WIDTH - 250, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # --- Video scaling and placement logic (unchanged) ---
        video_h, video_w = annotated_frame.shape[:2]
        video_area_h = self.config.UI_HEIGHT - self.config.PANEL_HEIGHT
        video_area_w = self.config.UI_WIDTH
        scale = min(video_area_w / video_w, video_area_h / video_h)
        new_w, new_h = int(video_w * scale), int(video_h * scale)
        resized_video = cv2.resize(annotated_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        x_offset = (video_area_w - new_w) // 2
        y_offset = self.config.PANEL_HEIGHT + (video_area_h - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_video
        
        return canvas

    def run(self):
        try:
            self.initialize_video()
            print("\n" + "="*60 + "\nFOOTFALL COUNTER STARTED\n" + "="*60)
            print("Press 'q' to quit | 's' to save screenshot | 'r' to reset counts\n" + "="*60 + "\n")

            while True:
                ret, frame = self.cap.read()
                if not ret: break

                self.frame_count += 1
                detections = []
                if (self.frame_count % self.config.FRAME_SKIP == 0):
                    detections = self.detect_people(frame)
                    self.last_detections = detections
                else:
                    detections = self.last_detections

                tracked_objects = self.tracker.update(detections)
                for object_id, centroid in tracked_objects.items():
                    self.check_line_crossing(object_id, centroid)
                
                annotated_frame = self.draw_visualizations_on_frame(frame.copy(), detections, tracked_objects)
                final_layout = self.create_final_layout(annotated_frame)
                
                cv2.imshow('Footfall Counter', final_layout)
                
                if self.output_writer is not None:
                    self.output_writer.write(final_layout)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): break
                elif key == ord('s'):
                    os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
                    path = os.path.join(self.config.OUTPUT_DIR, f'screenshot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg')
                    cv2.imwrite(path, final_layout)
                    print(f"Screenshot saved: {path}")
                elif key == ord('r'):
                    self.entry_count, self.exit_count = 0, 0
                    self.tracker.counted.clear()
                    self.crossed_objects.clear()
                    print("Counts reset!")
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            traceback.print_exc()
        finally:
            self.cleanup()
            self.print_final_stats()

    # REFACTORED: This function now prints the detailed summary you wanted
    def print_final_stats(self):
        elapsed_time = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        print("\n" + "=" * 60)
        print("FINAL STATISTICS")
        print("=" * 60)
        print(f"Total Entering / Moving Towards Camera:   {self.entry_count}")
        print(f"Total Exit / Moving Away from Camera:     {self.exit_count}")
        print(f"Net Count:                                {self.entry_count - self.exit_count}")
        print(f"Total Crossings:                          {self.entry_count + self.exit_count}")
        print(f"Frames Processed:                         {self.frame_count}")
        print(f"Average FPS:                              {avg_fps:.2f}")
        print(f"Processing Time:                          {elapsed_time:.2f} seconds")
        print("=" * 60 + "\n")

    def cleanup(self):
        if self.cap is not None: self.cap.release()
        if self.output_writer is not None: self.output_writer.release()
        cv2.destroyAllWindows()
        print("✓ Resources released")


# ==================== MAIN EXECUTION (Your logic) ====================
def main():
    # ... [Your main() function is unchanged and goes here] ...
    print("""
    ╔══════════════════════════════════════════════════════╗
    ║           FOOTFALL COUNTER - Computer Vision         ║
    ╚══════════════════════════════════════════════════════╝
    """)
    print("Video Source:")
    print("1. Webcam")
    print("2. Video file")
    choice = input("\nEnter your choice (1 or 2): ").strip()
    if choice == '1':
        video_source = 0
        print("Using webcam...")
    elif choice == '2':
        video_path = input("Enter video file path: ").strip()
        if not os.path.exists(video_path):
            print(f"Error: File not found: {video_path}")
            return
        video_source = video_path
        print(f"Using video file: {video_path}")
    else:
        print("Invalid choice. Using webcam as default...")
        video_source = 0
    print("\nCounting Line:")
    print("1. Horizontal line (count up/down movement) - default")
    print("2. Vertical line (count left/right movement)")
    line_choice = input("Enter choice (1 or 2, press Enter for default): ").strip()
    if line_choice == '2':
        Config.LINE_ORIENTATION = 'vertical'
        Config.ENTRY_SIDE = 'right'
        print("Using vertical counting line (ENTRY_SIDE set to 'right')")
    else:
        Config.LINE_ORIENTATION = 'horizontal'
        Config.ENTRY_SIDE = 'below'
        print("Using horizontal counting line (ENTRY_SIDE set to 'below')")
    try:
        fs = input("\nFrameskip for inference (1 = every frame, 2 = every 2nd frame, default 1): ").strip()
        if fs:
            Config.FRAME_SKIP = max(1, int(fs))
            print(f"FRAME_SKIP set to {Config.FRAME_SKIP}")
    except Exception:
        print("Invalid frameskip input, using default 1.")
    try:
        counter = FootfallCounter(video_source=video_source)
        counter.run()
    except Exception as e:
        print(f"\n❌ Error initializing counter: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
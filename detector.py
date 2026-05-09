"""
detector.py
- YOLO for players
- Motion detection for ball
- Rule-based racket approximation (no MediaPipe dependency)
"""

import cv2
import numpy as np
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        self.prev_frame = None

    def detect_players(self, frame):
        """Return list of player bounding boxes [(x1,y1,x2,y2)]."""
        results = self.model(frame, verbose=False)
        boxes = []
        for box in results[0].boxes:
            if int(box.cls[0]) == 0:  # person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2, y2))
        return boxes

    def detect_ball(self, frame):
        """Detect ball using motion difference."""
        if self.prev_frame is None:
            self.prev_frame = frame.copy()
            return None

        diff = cv2.absdiff(self.prev_frame, frame)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.GaussianBlur(thresh, (5,5), 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ball_center = None
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 50:
                x, y, w, h = cv2.boundingRect(largest)
                ball_center = (x + w//2, y + h//2)

        self.prev_frame = frame.copy()
        return ball_center

    def detect_racket(self, frame, player_boxes):
        """
        Approximate racket position as lower-right area of the first player.
        Meets requirement without external libraries.
        """
        if not player_boxes:
            return None
        x1, y1, x2, y2 = player_boxes[0]
        # Racket often near bottom right of player's bounding box
        racket_x = x2 - 20
        racket_y = y2 - 30
        h, w, _ = frame.shape
        racket_x = max(0, min(w-1, racket_x))
        racket_y = max(0, min(h-1, racket_y))
        return (racket_x, racket_y)

    def draw_detections(self, frame, player_boxes, ball_center, racket_pos):
        for (x1,y1,x2,y2) in player_boxes:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, "Player", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        if ball_center:
            cv2.circle(frame, ball_center, 6, (0,0,255), -1)
            cv2.putText(frame, "Ball", (ball_center[0]-10, ball_center[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        if racket_pos:
            cv2.circle(frame, racket_pos, 5, (255,0,0), -1)
            cv2.putText(frame, "Racket", (racket_pos[0]-10, racket_pos[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        return frame

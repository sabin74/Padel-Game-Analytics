"""
detector.py

Purpose:
Handles object detection:
- Players using YOLO
- Ball using YOLO + Motion Detection fallback

Design Philosophy:
- Keep detection lightweight
- Use hybrid approach for robustness
"""

import cv2
import numpy as np
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        # Initialize YOLO model
        self.model = YOLO(model_path)
        self.prev_frame = None  # For motion detection

    # PLAYER DETECTION (YOLO)
    def detect_players(self, frame):

        # Detect players using YOLO (class: person)
        # Returns: List of bounding boxes [(x1, y1, x2, y2)]

        results = self.model(frame, verbose=False)

        player_boxes = []

        for box in results[0].boxes:
            cls_id = int(box.cls[0])

            # Class 0 = person (COCO dataset)
            if cls_id == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                player_boxes.append((x1, y1, x2, y2))

        return player_boxes


    # BALL DETECTION (HYBRID)
    def detect_ball(self, frame):
        """
        Detect ball using:
        1. Motion detection (primary)
        2. YOLO (optional attempt)

        Returns:(x, y) center of ball OR None
        """
        ball_center = None

        if self.prev_frame is not None:
            frame_diff = cv2.absdiff(self.prev_frame, frame)
            gray = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)

            # Threshold to highlight motion
            _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)

            # Remove noise
            thresh = cv2.GaussianBlur(thresh, (5, 5), 0)

            # Find contours
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                # Pick largest moving object (likely ball)
                largest_contour = max(contours, key=cv2.contourArea)

                if cv2.contourArea(largest_contour) > 100:  # filter noise
                    x, y, w, h = cv2.boundingRect(largest_contour)

                    # Ball center
                    ball_center = (x + w // 2, y + h // 2)

        # Update previous frame
        self.prev_frame = frame.copy()

        return ball_center

    # VISUALIZATION
    def draw_detections(self, frame, player_boxes, ball_center):
        #Draw players and ball on frame
        # Draw players (Green)

        for (x1, y1, x2, y2) in player_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, "Player", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

        # Draw ball (Red)
        if ball_center:
            cv2.circle(frame, ball_center, 5, (0, 0, 255), -1)
            cv2.putText(
                frame, "Ball",
                (ball_center[0], ball_center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
            )

        return frame




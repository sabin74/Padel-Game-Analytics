"""
tracker.py
Simple ball trajectory storage
"""
import cv2

class BallTracker:
    def __init__(self, max_history=30):
        self.positions = []
        self.max_history = max_history

    def update(self, ball_center):
        if ball_center:
            self.positions.append(ball_center)
            if len(self.positions) > self.max_history:
                self.positions.pop(0)
        return self.positions

def draw_trajectory(frame, positions):
    for i in range(1, len(positions)):
        if positions[i-1] and positions[i]:
            cv2.line(frame, positions[i-1], positions[i], (255,255,0), 2)
    return frame

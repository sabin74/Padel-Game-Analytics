"""
tracker.py

Purpose:
- Track ball movement across frames

Design Philosophy:
- Keep simple (no heavy tracking algorithms)
- Store trajectory over time
- Prepare data for feature extraction
"""

import numpy as np
import cv2


class BallTracker:
    def __init__(self, max_history=50):
        """
        Initialize tracker

        Args:
            max_history (int): How many past positions to keep
        """
        self.positions = []  # Stores ball trajectory
        self.max_history = max_history

    def update(self, ball_center):
        """
        Update tracker with new ball position

        Args:
            ball_center (tuple or None): (x, y)

        Returns:
            List of tracked positions
        """

        if ball_center is not None:
            self.positions.append(ball_center)

            # Keep only recent positions (avoid memory overflow)
            if len(self.positions) > self.max_history:
                self.positions.pop(0)

        return self.positions

    def get_trajectory(self):
        """
        Return trajectory as numpy array
        """
        return np.array(self.positions)

def draw_trajectory(frame, positions):

    # Draw ball trajectory as a line


    for i in range(1, len(positions)):
        if positions[i - 1] is None or positions[i] is None:
            continue

        cv2.line(
            frame,
            positions[i - 1],
            positions[i],
            (255, 0, 0),  # Blue line
            2
        )

    return frame

    def reset(self):
        """
        Reset tracking (useful for new rally detection later)
        """
        self.positions = []


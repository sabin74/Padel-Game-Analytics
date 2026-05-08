"""
features.py

Purpose:
- Extract motion-based features from ball trajectory

Features:
- Direction (dx, dy)
- Speed
- Angle
- Movement pattern (trend)

Why this matters:
- These features directly drive shot classification
"""

import numpy as np
import math

class FeatureExtractor:
    def __init__(self):
        pass

    # BASIC MOTION FEATURES

    def compute_motion(self, positions):
        """
        Compute motion features from trajectory

        Args:
            positions (list): [(x1, y1), (x2, y2), ...]

        Returns:
            dict with direction, speed, angle
        """

        if len(positions) < 2:
            return None

        x1, y1 = positions[-2]
        x2, y2 = positions[-1]

        dx = x2 - x1
        dy = y2 - y1

        # Speed (Euclidean distance)
        speed = math.sqrt(dx**2 + dy**2)

        # Angle (in degrees)
        angle = math.degrees(math.atan2(dy, dx))

        return {
            "dx": dx,
            "dy": dy,
            "speed": speed,
            "angle": angle
        }

    # ADVANCED PATTERN FEATURES
    def compute_trend(self, positions, window=5):
        """
        Analyze recent movement trend

        Args:
            positions (list)
            window (int): number of recent points

        Returns:
            trend info (direction + motion type)
        """

        if len(positions) < window:
            return None

        recent = positions[-window:]

        dx_total = recent[-1][0] - recent[0][0]
        dy_total = recent[-1][1] - recent[0][1]

        # Determine movement direction
        if abs(dx_total) > abs(dy_total):
            direction = "horizontal"
        else:
            direction = "vertical"

        # Determine motion type
        if dy_total > 0:
            motion = "downward"
        else:
            motion = "upward"

        return {
            "direction_type": direction,
            "motion_type": motion,
            "dx_total": dx_total,
            "dy_total": dy_total
        }

    # COMBINED FEATURE OUTPUT
    def extract_features(self, positions):
        """
        Main function to extract all features

        Returns:
            dict with all computed features
        """

        motion = self.compute_motion(positions)
        trend = self.compute_trend(positions)

        if motion is None:
            return None

        features = {}
        features.update(motion)

        if trend:
            features.update(trend)

        return features

def detect_bounce(positions):
    """
    Detect bounce using direction change
    """

    if len(positions) < 3:
        return False

    y1 = positions[-3][1]
    y2 = positions[-2][1]
    y3 = positions[-1][1]

    dy1 = y2 - y1
    dy2 = y3 - y2

    # Direction change
    if dy1 > 0 and dy2 < 0:
        return True

    return False




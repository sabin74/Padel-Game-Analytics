"""
features.py
Extract motion features from ball trajectory
"""
import math

class FeatureExtractor:
    @staticmethod
    def compute_motion(positions):
        if len(positions) < 2:
            return None
        (x1,y1), (x2,y2) = positions[-2], positions[-1]
        dx = x2 - x1
        dy = y2 - y1
        speed = math.hypot(dx, dy)
        angle = math.degrees(math.atan2(dy, dx))
        return {"dx": dx, "dy": dy, "speed": speed, "angle": angle}

    @staticmethod
    def extract_features(positions):
        return FeatureExtractor.compute_motion(positions)

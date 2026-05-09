"""
classifier.py
Rule-based shot classification using ball motion
"""
class ShotClassifier:
    def __init__(self):
        self.speed_threshold_smash = 15   # pixels per frame
        self.speed_threshold_min = 3

    def classify(self, features, racket_pos=None, player_box=None, frame_id=0):
        if features is None:
            return None
        dx = features.get("dx", 0)
        dy = features.get("dy", 0)
        speed = features.get("speed", 0)

        # Smash: high speed + downward
        if speed > self.speed_threshold_smash and dy > 0:
            return "smash"
        # Serve: early frames + downward motion
        if frame_id < 100 and dy > 0:
            return "serve"
        if speed < self.speed_threshold_min:
            return None
        # Forehand/Backhand by horizontal direction
        if abs(dx) > abs(dy):
            return "forehand" if dx > 0 else "backhand"
        return None

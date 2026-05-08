"""
classifier.py

Purpose:
- Classify shots based on motion features

Shot Types:
- Forehand
- Backhand
- Smash
- Serve (basic detection)

Design Philosophy:
- Rule-based (fast + explainable)
- Simple but meaningful
"""

class ShotClassifier:
    def __init__(self):
        # Tunable thresholds (IMPORTANT)
        self.speed_threshold_smash = 20
        self.speed_threshold_min = 3

    def classify(self, features, positions, frame_id):
        """
        Classify shot based on extracted features

        Args:
            features (dict)
            positions (list)
            frame_id (int)

        Returns:
            shot_label (str or None)
        """

        if features is None:
            return None

        dx = features["dx"]
        dy = features["dy"]
        speed = features["speed"]
        angle = features["angle"]


        # SMASH DETECTION

        if speed > self.speed_threshold_smash and dy > 0:
            return "smash"


        # SERVE DETECTION (simple)

        # First few frames OR initial downward motion
        if frame_id < 50 and dy > 0:
            return "serve"


        # FOREHAND / BACKHAND


        # Ignore noise
        if speed < self.speed_threshold_min:
            return None

        # Horizontal movement
        if abs(dx) > abs(dy):

            # Direction-based classification
            if dx > 0:
                return "forehand"
            else:
                return "backhand"

        return None


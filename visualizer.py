"""
visualizer.py
Overlay text and save output video
"""
import cv2

class Visualizer:
    def __init__(self, output_path, fps, frame_size):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    def draw_shot_label(self, frame, shot):
        if shot:
            cv2.putText(frame, f"Shot: {shot.upper()}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
        return frame

    def draw_analytics(self, frame, frame_id, features):
        cv2.putText(frame, f"Frame: {frame_id}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        if features and 'speed' in features:
            cv2.putText(frame, f"Speed: {features['speed']:.1f}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        return frame

    def write_frame(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()

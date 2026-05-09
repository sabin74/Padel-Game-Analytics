"""
video_processor.py
Handles video loading and frame iteration
"""
import cv2

class VideoProcessor:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video loaded: {self.fps:.2f} fps, {self.total_frames} frames")

    def process_frames(self, resize_dim=(640,360), skip_frames=1):
        frame_id = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_id += 1
            if frame_id % skip_frames != 0:
                continue
            frame = cv2.resize(frame, resize_dim)
            yield frame_id, frame
        self.cap.release()
        print("Video processing finished.")

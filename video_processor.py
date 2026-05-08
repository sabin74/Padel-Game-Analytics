"""
video_processor.py

Purpose:
- Handles video loading and frame-by-frame processing.

Why this matters:
- This is the ENTRY POINT of your pipeline.
- Everything else (detection, tracking, classification) depends on this.

Design Goals:
- Clean structure
- Easy to extend later
- Optimized for low-end systems
"""

import cv2


class VideoProcessor:
    def __init__(self, video_path):
        """
        Initialize video capture

        Args:
            video_path (str): Path to input video
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        # Validate video
        if not self.cap.isOpened():
            raise ValueError(f"Error opening video: {video_path}")

        # Extract basic properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video Loaded Successfully")
        print(f"FPS: {self.fps}")
        print(f"Total Frames: {self.total_frames}")

    def process_frames(self, resize_dim=(640, 360), skip_frames=1):
        """
        Generator function to process video frames

        Args:
            resize_dim (tuple): Resize resolution (width, height)
            skip_frames (int): Process every Nth frame

        Yields:
            frame_id (int), frame (ndarray)
        """

        frame_id = 0

        while self.cap.isOpened():
            ret, frame = self.cap.read()

            if not ret:
                break  # End of video

            frame_id += 1

            # Skip frames for performance optimization
            if frame_id % skip_frames != 0:
                continue

            # Resize frame 
            frame = cv2.resize(frame, resize_dim)

            yield frame_id, frame

        self.cap.release()
        print("Video processing completed.")


def display_video(video_path):
    # Simple utility to test video playback

    processor = VideoProcessor(video_path)

    for frame_id, frame in processor.process_frames(skip_frames=1):

        # Display frame
        cv2.imshow("Video Playback", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    VIDEO_PATH = "input_sample_video.mp4"

    display_video(VIDEO_PATH)



"""
visualizer.py

Purpose:
---------
Handle:
- Drawing overlays
- Writing output video
- Displaying analytics
"""

import cv2

class Visualizer:
    def __init__(self, output_path, fps, frame_size):
        """
        Initialize video writer

        Args:
            output_path (str)
            fps (float)
            frame_size (tuple) -> (width, height)
        """

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        self.writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            frame_size
        )

    # DRAW SHOT LABEL
    def draw_shot_label(self, frame, shot):

        if shot:
            cv2.putText(
                frame,
                f"Shot: {shot.upper()}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2
            )

        return frame

    # DRAW ANALYTICS
    def draw_analytics(self, frame, frame_id, features):

        # Frame number
        cv2.putText(
            frame,
            f"Frame: {frame_id}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # Speed info
        if features:
            speed = features.get("speed", 0)

            cv2.putText(
                frame,
                f"Speed: {speed:.2f}",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        return frame

    # WRITE FRAME
    def write_frame(self, frame):
        """
        Save frame into output video
        """
        self.writer.write(frame)

    # RELEASE WRITER
    def release(self):
        """
        Release video writer
        """
        self.writer.release()


"""
test_tracking.py
Test ball tracking and trajectory visualization
"""
import cv2
from video_processor import VideoProcessor
from detector import ObjectDetector
from tracker import BallTracker, draw_trajectory

VIDEO_PATH = "input_sample_video.mp4"
SKIP_FRAMES = 2
FRAME_WIDTH, FRAME_HEIGHT = 640, 360

def main():
    processor = VideoProcessor(VIDEO_PATH)
    detector = ObjectDetector()
    tracker = BallTracker()

    for frame_id, frame in processor.process_frames((FRAME_WIDTH, FRAME_HEIGHT), skip_frames=SKIP_FRAMES):
        ball = detector.detect_ball(frame)
        positions = tracker.update(ball)

        display = frame.copy()
        display = draw_trajectory(display, positions)
        if ball:
            cv2.circle(display, ball, 6, (0,0,255), -1)
        cv2.putText(display, f"Frame: {frame_id}", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow("Tracking Test", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

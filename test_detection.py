"""
test_detection.py
Independent test for detection module
"""
import cv2
from video_processor import VideoProcessor
from detector import ObjectDetector

VIDEO_PATH = "input_sample_video.mp4"
SKIP_FRAMES = 3
FRAME_WIDTH, FRAME_HEIGHT = 640, 360

def main():
    processor = VideoProcessor(VIDEO_PATH)
    detector = ObjectDetector()

    for frame_id, frame in processor.process_frames((FRAME_WIDTH, FRAME_HEIGHT), skip_frames=SKIP_FRAMES):
        players = detector.detect_players(frame)
        ball = detector.detect_ball(frame)
        racket = detector.detect_racket(frame, players)

        display = detector.draw_detections(frame, players, ball, racket)
        cv2.putText(display, f"Frame: {frame_id}", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow("Detection Test", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

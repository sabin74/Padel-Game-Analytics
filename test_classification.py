"""
test_classification.py
End-to-end classification test
"""
import cv2
from video_processor import VideoProcessor
from detector import ObjectDetector
from tracker import BallTracker
from features import FeatureExtractor
from classifier import ShotClassifier

VIDEO_PATH = "input_sample_video.mp4"
SKIP_FRAMES = 5
FRAME_WIDTH, FRAME_HEIGHT = 640, 360

def main():
    processor = VideoProcessor(VIDEO_PATH)
    detector = ObjectDetector()
    tracker = BallTracker()
    fe = FeatureExtractor()
    classifier = ShotClassifier()

    for frame_id, frame in processor.process_frames((FRAME_WIDTH, FRAME_HEIGHT), skip_frames=SKIP_FRAMES):
        players = detector.detect_players(frame)
        ball = detector.detect_ball(frame)
        racket = detector.detect_racket(frame, players)
        positions = tracker.update(ball)
        features = fe.extract_features(positions)
        shot = classifier.classify(features, racket, players[0] if players else None, frame_id)

        display = frame.copy()
        for (x1,y1,x2,y2) in players:
            cv2.rectangle(display, (x1,y1), (x2,y2), (0,255,0), 2)
        if ball:
            cv2.circle(display, ball, 6, (0,0,255), -1)
        if racket:
            cv2.circle(display, racket, 5, (255,0,0), -1)
        if shot:
            cv2.putText(display, f"{shot.upper()}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.putText(display, f"Frame: {frame_id}", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.imshow("Classification Test", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

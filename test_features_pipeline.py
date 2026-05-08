import cv2
from video_processor import VideoProcessor
from detector import ObjectDetector
from tracker import BallTracker
from tracker import draw_trajectory
from features import FeatureExtractor

VIDEO_PATH = "input_sample_video.mp4"

processor = VideoProcessor(VIDEO_PATH)
detector = ObjectDetector()
tracker = BallTracker(max_history=30)
feature_extractor = FeatureExtractor()

for frame_id, frame in processor.process_frames(skip_frames=2):

    # Step 1: Detection
    players = detector.detect_players(frame)
    ball = detector.detect_ball(frame)

    # Step 2: Tracking
    positions = tracker.update(ball)

    # Step 3: Feature Extraction
    features = feature_extractor.extract_features(positions)

    # Step 4: Draw detections
    frame = detector.draw_detections(frame, players, ball)
    frame = draw_trajectory(frame, positions)

    # Step 5: Display features
    if features:
        text = f"Speed: {features['speed']:.2f}"
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Step 6: Show
    cv2.imshow("Feature Extraction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()


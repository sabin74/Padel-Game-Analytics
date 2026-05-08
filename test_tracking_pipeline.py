import cv2
from video_processor import VideoProcessor
from detector import ObjectDetector
from tracker import BallTracker
from tracker import draw_trajectory

VIDEO_PATH = "input_sample_video.mp4"

processor = VideoProcessor(VIDEO_PATH)
detector = ObjectDetector()
tracker = BallTracker(max_history=30)

for frame_id, frame in processor.process_frames(skip_frames=2):

    # Step 1: Detect
    players = detector.detect_players(frame)
    ball = detector.detect_ball(frame)

    # Step 2: Track
    positions = tracker.update(ball)

    # Step 3: Draw detections
    frame = detector.draw_detections(frame, players, ball)

    # Step 4: Draw trajectory
    frame = draw_trajectory(frame, positions)

    # Step 5: Display
    cv2.imshow("Tracking Pipeline", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()


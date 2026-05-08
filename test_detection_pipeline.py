import cv2
from video_processor import VideoProcessor
from detector import ObjectDetector

VIDEO_PATH = "input_sample_video.mp4"

processor = VideoProcessor(VIDEO_PATH)
detector = ObjectDetector()

for frame_id, frame in processor.process_frames(skip_frames=2):

    # Detect players
    players = detector.detect_players(frame)

    # Detect ball
    ball = detector.detect_ball(frame)

    # Draw results
    output_frame = detector.draw_detections(frame, players, ball)

    # Display
    cv2.imshow("Detection Pipeline", output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()


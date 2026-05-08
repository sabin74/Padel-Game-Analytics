import cv2

from video_processor import VideoProcessor
from detector import ObjectDetector
from tracker import BallTracker, draw_trajectory
from features import FeatureExtractor
from classifier import ShotClassifier
from output_writer import OutputWriter
from visualizer import Visualizer
from analytics import ShotAnalytics

VIDEO_PATH = "input_sample_video.mp4"

# INITIALIZE SYSTEM COMPONENTS
processor = VideoProcessor(VIDEO_PATH)

detector = ObjectDetector()
tracker = BallTracker(max_history=30)

feature_extractor = FeatureExtractor()
classifier = ShotClassifier()

writer = OutputWriter()


# VIDEO OUTPUT SETTINGS
FRAME_WIDTH = 640
FRAME_HEIGHT = 360

visualizer = Visualizer(
    output_path="output/output_video.mp4",
    fps=processor.fps,
    frame_size=(FRAME_WIDTH, FRAME_HEIGHT)
)
last_shot = None


# MAIN LOOP
for frame_id, frame in processor.process_frames(
        resize_dim=(FRAME_WIDTH, FRAME_HEIGHT),
        skip_frames=2):

    # DETECTION
    players = detector.detect_players(frame)
    ball = detector.detect_ball(frame)

    # TRACKING
    positions = tracker.update(ball)

    # FEATURE EXTRACTION
    features = feature_extractor.extract_features(positions)

    # SHOT CLASSIFICATION
    shot = classifier.classify(features, positions, frame_id)

    # SAVE OUTPUT DATA
    if shot and shot != last_shot:

        writer.add_result(
            frame_id=frame_id,
            fps=processor.fps,
            shot=shot,
            player="player_1"
        )

        last_shot = shot

    # DRAW VISUALIZATIONS

    frame = detector.draw_detections(frame, players, ball)
    frame = draw_trajectory(frame, positions)
    frame = visualizer.draw_shot_label(frame, shot)
    frame = visualizer.draw_analytics(
        frame,
        frame_id,
        features
    )

    # DISPLAY

    cv2.imshow("Padel Analytics System", frame)

    # SAVE VIDEO FRAME

    visualizer.write_frame(frame)

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Shot Analysis
analytics = ShotAnalytics()
if shot and shot != last_shot:
    analytics.add_shot(shot)

# print Summary
analytics.print_summary()
counts = analytics.get_counts()
plot_shot_distribution(counts)

# SAVE FILES
writer.save_all()
visualizer.release()
cv2.destroyAllWindows()
print("Output video saved successfully.")



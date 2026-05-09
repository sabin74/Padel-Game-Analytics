"""
main.py
Full pipeline: detection, tracking, classification, output, video
"""
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

def main():
    processor = VideoProcessor(VIDEO_PATH)
    detector = ObjectDetector()
    tracker = BallTracker()
    fe = FeatureExtractor()
    classifier = ShotClassifier()
    writer = OutputWriter()
    analytics = ShotAnalytics()

    frame_width, frame_height = 640, 360
    visualizer = Visualizer("output/output_video.mp4", processor.fps, (frame_width, frame_height))

    last_shot = None

    for frame_id, frame in processor.process_frames((frame_width, frame_height), skip_frames=2):
        # Detection
        players = detector.detect_players(frame)
        ball = detector.detect_ball(frame)
        racket = detector.detect_racket(frame, players)   # pass players for fallback

        # Tracking
        positions = tracker.update(ball)

        # Features
        features = fe.extract_features(positions)

        # Classification
        shot = classifier.classify(features, racket, players[0] if players else None, frame_id)

        # Save new shot
        if shot and shot != last_shot:
            writer.add_result(frame_id, processor.fps, shot)
            analytics.add_shot(shot)
            last_shot = shot

        # Drawing
        frame = detector.draw_detections(frame, players, ball, racket)
        frame = draw_trajectory(frame, positions)
        frame = visualizer.draw_shot_label(frame, shot)
        frame = visualizer.draw_analytics(frame, frame_id, features)

        cv2.imshow("Padel Analytics", frame)
        visualizer.write_frame(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    writer.save_all()
    visualizer.release()
    cv2.destroyAllWindows()
    analytics.print_summary()
    analytics.plot_distribution()
    print("Processing complete. Check output/ folder.")

if __name__ == "__main__":
    main()

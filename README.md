# Padel Game Analytics — Shot Classification System

## Project Overview

This project is a computer vision prototype that analyzes a padel match video, detects players, the ball, and the racket, tracks the ball, classifies shots (forehand, backhand, smash, serve), and outputs structured results (JSON/CSV) along with an annotated video and shot analytics.

It was built as part of an AI/ML internship assignment for Layman AI. The goal is to demonstrate a practical end‑to‑end pipeline using Python, OpenCV, YOLOv8, and simple rule‑based logic.

## Key Features

-   **Detection** – YOLOv8 for players, motion differencing for the ball, heuristic (player’s lower‑right area) for the racket.
-   **Tracking** – Ball trajectory stored in a FIFO buffer, visualised as a line.
-   **Shot Classification** – Rule‑based using speed and direction:
    - High speed + downward motion → **smash**
    - Early frames + downward motion → **serve**
    - Horizontal motion → **forehand** (dx>0) or **backhand** (dx<0)
-   **Output Formats** – JSON and CSV with frame number, timestamp, shot type, and player (fixed as player_1).
-   **Analytics** – Shot count summary and a bar chart (matplotlib).
-   **Video Output** – Overlaid bounding boxes, ball circle, racket dot, trajectory, and shot label, saved as MP4.

## Repository Structure

```
.
├── README.md                 # This file
├── requirements.txt          # Required Python packages
├── main.py                   # Full pipeline entry point
├── video_processor.py        # Video loading and frame iteration
├── detector.py               # Player, ball, racket detection
├── tracker.py                # Ball trajectory storage and drawing
├── features.py               # Motion feature extraction
├── classifier.py             # Rule‑based shot classification
├── output_writer.py          # JSON/CSV export
├── visualizer.py             # Overlay text and video writer
├── analytics.py              # Shot counts and bar chart
├── test_detection.py         # Standalone test for detection
├── test_tracking.py          # Standalone test for tracking
├── test_classification.py    # End‑to‑end classification test
├── Padel_Game_Analytics.ipynb # Google Colab notebook version
└── output/                   # Generated after running main.py
    ├── results.json
    ├── results.csv
    ├── output_video.mp4
    └── shot_distribution.png
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/sabin74/Padel-Game-Analytics.git
cd Padel-Game-Analytics
```

### 2. Install Dependencies

Create a virtual environment (recommended) and install the required packages:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The `requirements.txt` file contains:
```
ultralytics
opencv-python
numpy
pandas
matplotlib
```

### 3. Prepare the Input Video

Place your padel match video in the project root and name it `input_sample_video.mp4`.  
If you have a different filename, update the `VIDEO_PATH` variable in `main.py`.

### 4. Run the Main Pipeline

```bash
python main.py
```

The script will:
- Process the video frame by frame (resized to 640x360, processing every 2nd frame).
- Display a live preview (press `q` to stop early).
- Write the annotated video, JSON, CSV, and shot distribution plot into the `output/` folder.

### 5. Test Individual Components (Optional)

```bash
python test_detection.py      # Check player, ball, racket detection
python test_tracking.py       # Verify ball trajectory tracking
python test_classification.py # See real‑time shot labels
```

## Output Files

After running `main.py`, the `output/` folder will contain:

| File | Description |
|------|-------------|
| `results.json` | List of detected shots with frame, timestamp, shot type, player. |
| `results.csv`  | Same data in tabular format. |
| `output_video.mp4` | Original video with all overlays (bounding boxes, trajectory, shot labels). |
| `shot_distribution.png` | Bar chart of forehand / backhand / smash / serve counts. |

## Approach Explanation

### Methodology

1.  **Detection**  
    - **Players** – YOLOv8n (pretrained on COCO, class 0 = person). Fast and reliable.  
    - **Ball** – Frame differencing + contour detection. The moving ball appears as a small blob.  
    - **Racket** – Since no pretrained model exists, we approximate the racket as the lower‑right area of the first detected player. This satisfies the mandatory requirement without external pose libraries.

2.  **Tracking** – The ball’s centre is stored in a list (max 30 positions). The trajectory is drawn as a polyline.

3.  **Feature Extraction** – From consecutive ball positions we compute `dx`, `dy`, speed (Euclidean distance), and angle.

4.  **Classification (Rule‑Based)** – Simple thresholds that work well for a prototype:  
    - `speed > 15` and `dy > 0` → **smash**  
    - `frame_id < 100` and `dy > 0` → **serve**  
    - `abs(dx) > abs(dy)` and `dx > 0` → **forehand**  
    - `abs(dx) > abs(dy)` and `dx < 0` → **backhand**  
    - otherwise `None` (no shot).

5.  **Output** – Each time a new shot is detected, we write a record to JSON/CSV. The visualizer overlays the shot name on the video.

6.  **Analytics** – `collections.Counter` counts the shots, and `matplotlib` generates a bar chart.

### Challenges Faced

-   **Ball detection** – The ball is small and fast, causing motion blur. Solved by lowering the threshold on contour area and using Gaussian blur.
-   **Racket detection** – No standard detector available. Used a heuristic (lower right of player) – acceptable for a prototype but not perfect.
-   **Tuning thresholds** – Different videos may need different speed values. Made them configurable in `classifier.py`.
-   **Performance** – Processing every frame is slow on CPU. Added frame skipping (every 2nd frame) and resizing.

### Improvements for a Production System

-   **Better racket detection** – Train/fine‑tune YOLO on a dataset of padel rackets, or use MediaPipe Pose for wrist landmarks.
-   **Robust tracking** – Use a Kalman filter or SORT to handle occlusions.
-   **Machine learning classifier** – Replace rules with a lightweight CNN or LSTM on cropped player sequences.
-   **Player differentiation** – Assign left/right player IDs based on court zones.
-   **Bounce detection** – Add vertical direction change analysis.
-   **Real‑time dashboard** – Stream analytics with Streamlit or Gradio.

## Google Drive Links (Models & Output)

The following files are large and hosted on Google Drive (as required by the assignment):

-   **YOLO model weights** – `yolov8n.pt` (automatically downloaded by Ultralytics; backup link)  
    [Download from Google Drive](https://drive.google.com/your-yolo-link-here)

-   **Sample output video** – `output/output_video.mp4`  
    [Watch on Google Drive](https://drive.google.com/your-output-video-link-here)

-   **All results (JSON + CSV + chart)** – Zipped folder  
    [Download from Google Drive](https://drive.google.com/your-zip-link-here)


## Demo Video

A short screen recording demonstrating the system in action is available here:  
[📺 Watch Demo Video](https://drive.google.com/your-demo-video-link-here)

The demo shows:
- Running `python main.py`
- The live preview (bounding boxes, trajectory, shot label)
- The final output video playback
- Inspecting `results.json` and the shot distribution plot

## Requirements & Compatibility

-   Python 3.8+
-   Packages listed in `requirements.txt`
-   Works on Windows, macOS, Linux, and Google Colab (see `Padel_Game_Analytics.ipynb`)

## Author

Sabin – AI/ML Intern Candidate  
Layman AI Assignment – May 2026

## License

This code is provided for review purposes as part of a job application. All rights reserved.

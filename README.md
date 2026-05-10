# Padel Game Analytics — Shot Classification System

## Project Overview

This project is a computer vision project that implements the analysis of padel game video, detection of players, ball, and racket, tracking of ball, classification of forehand, backhand, smash, and serves, and, finally, providing the result in JSON/CSV format along with video annotations and shots analytics.

It was developed as part of an internship assignment from Layman AI. The primary goal is to showcase how such a project can be implemented using Python, OpenCV, YOLOv8, and rule-based reasoning.

## Key Features

-   **Detection**: YOLOv8 for players, motion difference for the ball, and heuristic for the racket (at the lower-right corner of player).
-   **Tracking**: Tracking of the ball through its path via a FIFO queue.
-   **Shot Detection**: Detection of shots through rules involving speed and angles:
    - High speed and downwards angle of the shot → **Smash**
    - Shot detected at an early frame and downwards angle → **Serve**
    - Horizontal angle of the shot → **Forehand** (if the direction is right to left) and **Backhand** (vice versa).
-   **Output Format**: Output is done in JSON/CSV file format and includes Frame Number, Timestamp, Shot Type, and Player (player_1).
-   **Analytics**: Counting and plotting shots via a bar graph using matplotlib.
-   **Video Output**: Drawing of bounding boxes, ball position, racket position, shot path, and shot names for each frame.

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

Create a virtual environment and install the required packages:

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

Prepare the file `input_sample_video.mp4` containing the padel match video in the root folder.  
Use another filename but adjust the `VIDEO_PATH` parameter inside `main.py`.

### 4. Run the Main Pipeline

```bash
python main.py
```

It will perform the following tasks:
- Process the input video on a frame basis (with size 640x360, considering each second frame);
- Show a live stream;
- Exit the script by clicking the key 'q';
- Generate the result output video file and JSON/CSV along with the shots graph in `output/` folder.

### 5. Test Individual Components

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
    - **Players** – Using YOLOv8n (COCO dataset, class 0 = person). Efficient and accurate.
    - **Ball** – Difference between consecutive frames and contour detection. Ball movement is denoted by small contours.
    - **Racket** – Since there are no pre-trained models for racket detection, we use the bottom right corner of the first player to comply with the mandatory task without using any external pose estimation tools.

2.  **Tracking** – Coordinates of the center of the ball are recorded in an array up to 30 frames. Path is drawn as a polyline.

3.  **Feature Extraction** – We compute `dx`, `dy`, velocity (Eucledian distance), and angle from successive locations of the ball.

4.  **Classification (Rule‑Based)** – Simple thresholds that work well for a prototype:  
    - `speed > 15` and `dy > 0` → **smash**  
    - `frame_id < 100` and `dy > 0` → **serve**  
    - `abs(dx) > abs(dy)` and `dx > 0` → **forehand**  
    - `abs(dx) > abs(dy)` and `dx < 0` → **backhand**  
    - otherwise `None` (no shot).

5.  **Output** – Each time a new shot is detected, we write a record to JSON/CSV. The visualizer overlays the shot name on the video.

6.  **Analytics** – `collections.Counter` counts the shots, and `matplotlib` generates a bar chart.

### Challenges Faced

-   **Ball detection**: The ball is small and fast moving. This leads to motion blur. This problem was overcome by reducing the threshold of contour area and performing Gaussian blur.
-   **Racket detection**: In this case, we do not have any detector to work with. We used heuristics (based on bottom-right part w.r.t player). This is acceptable for our prototype but certainly not the best solution.
-   **Threshold tuning**: Speeds can vary across videos. We provided adjustment using `classifier.py`.
-   **Performance**: Each frame takes a long time to process. This is why we slowed down our processing rate and also reduced the frame sizes.

### Improvements for a Production System

-   **Improved racket detection** – Use either YOLO training or MediaPipe Pose landmark detection based on wrist joints.
-   **Player tracking with robustness** – Apply Kalman filter or SORT method.
-   **Classifier through machine learning** – Deploy either small CNN or LSTM architecture.
-   **Distinctive player identification** – Assign unique IDs based on their locations on the court.
-   **Detection of ball bounces** – Incorporate detection of changes in the vertical direction.
-   **Real-time analytics dashboard** – Offer live analytics via Streamlit or Gradio interfaces.

## Google Drive Links (Models & Output)

The following files are large and hosted on Google Drive:

-   **YOLO model weights** – `yolov8n.pt` (automatically downloaded by Ultralytics; backup link)  
    [Download from Google Drive](https://drive.google.com/file/d/1xYJqXrZPGfBu1HZZg4Nq8Pv_FY5JZlaV/view?usp=sharing)

-   **Sample output video** – `output/output_video.mp4`  
    [Watch on Google Drive](https://drive.google.com/file/d/15A2uPf2XYQGJw9IPAPAUzR_xjOzKQNT_/view?usp=sharing)

-   **All results (JSON + CSV + chart)** – Zipped folder  
    [Download from Google Drive](https://drive.google.com/drive/folders/1D1dn9D_bXfBI0ed0IUBPDXSmB_WTYYG2?usp=sharing)


## Demo Video

A short screen recording demonstrating the system in action is available here:  
[📺 Watch Demo Video](https://drive.google.com/file/d/1d0hwlTLDOCnHxOYsb_BKuIOELAfNtvj5/view?usp=sharing)

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

This code is provided for review purposes as part of a internship application. All rights reserved.

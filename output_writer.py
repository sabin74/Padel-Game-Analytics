"""
output_writer.py

Purpose:
Save shot predictions into:
- JSON
- CSV

Why this matters:
- Structured outputs make the system usable
  for analytics and evaluation.
"""

import json
import pandas as pd
import os


class OutputWriter:
    def __init__(self, output_dir="output"):

        # Initialize output directory

        self.output_dir = output_dir
        self.results = []

        # Create output folder if not exists
        os.makedirs(output_dir, exist_ok=True)

    # STORE PREDICTION
    def add_result(self, frame_id, fps, shot, player="unknown"):
        """
        Store prediction result

        Args:
            frame_id (int)
            fps (float)
            shot (str)
            player (str)
        """

        timestamp = round(frame_id / fps, 2)

        result = {
            "frame": frame_id,
            "timestamp": timestamp,
            "shot": shot,
            "player": player
        }

        self.results.append(result)

    # SAVE JSON
    def save_json(self, filename="results.json"):
        # Save results as JSON

        path = os.path.join(self.output_dir, filename)

        with open(path, "w") as f:
            json.dump(self.results, f, indent=4)

        print(f"JSON saved: {path}")

    # SAVE CSV
    def save_csv(self, filename="results.csv"):
        # Save results as CSV

        path = os.path.join(self.output_dir, filename)

        df = pd.DataFrame(self.results)
        df.to_csv(path, index=False)

        print(f"CSV saved: {path}")

    # SAVE EVERYTHING
    def save_all(self):
        # Save both JSON and CSV

        self.save_json()
        self.save_csv()

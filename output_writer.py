"""
output_writer.py
Save results as JSON and CSV
"""
import json, os, pandas as pd

class OutputWriter:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        self.results = []
        os.makedirs(output_dir, exist_ok=True)

    def add_result(self, frame_id, fps, shot, player="player_1"):
        timestamp = round(frame_id / fps, 2)
        self.results.append({
            "frame": frame_id,
            "timestamp": timestamp,
            "shot": shot,
            "player": player
        })

    def save_json(self, filename="results.json"):
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"Saved: {path}")

    def save_csv(self, filename="results.csv"):
        path = os.path.join(self.output_dir, filename)
        pd.DataFrame(self.results).to_csv(path, index=False)
        print(f"Saved: {path}")

    def save_all(self):
        self.save_json()
        self.save_csv()

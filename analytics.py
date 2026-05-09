"""
analytics.py
Shot counts and bar chart
"""
from collections import Counter
import matplotlib.pyplot as plt

class ShotAnalytics:
    def __init__(self):
        self.shots = []

    def add_shot(self, shot):
        self.shots.append(shot)

    def get_counts(self):
        return Counter(self.shots)

    def print_summary(self):
        counts = self.get_counts()
        print("\n=== Shot Analytics ===")
        for shot, cnt in counts.items():
            print(f"{shot.capitalize():10} : {cnt}")

    def plot_distribution(self, save_path="output/shot_distribution.png"):
        counts = self.get_counts()
        if not counts:
            print("No shots to plot.")
            return
        plt.figure(figsize=(6,4))
        plt.bar(counts.keys(), counts.values(), color='skyblue')
        plt.title("Shot Distribution")
        plt.xlabel("Shot Type")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        print(f"Plot saved: {save_path}")

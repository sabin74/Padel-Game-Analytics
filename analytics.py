"""
analytics.py

Purpose:
- Generate shot analytics summary
"""

from collections import Counter


class ShotAnalytics:
    def __init__(self):
        self.shots = []

    def add_shot(self, shot):
        # Store detected shot
        if shot:
            self.shots.append(shot)

    def get_counts(self):

        # Count occurrences of each shot type
        return Counter(self.shots)

    def print_summary(self):
        # Print analytics summary
        counts = self.get_counts()

        print("\nShot Analytics Summary")
        print("-" * 30)

        for shot, count in counts.items():
            print(f"{shot.capitalize()}: {count}")

# Visualize Shot Distribution
import matplotlib.pyplot as plt


def plot_shot_distribution(counts):
    """
    Plot shot frequency graph
    """

    labels = list(counts.keys())
    values = list(counts.values())

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)

    plt.xlabel("Shot Type")
    plt.ylabel("Count")
    plt.title("Shot Distribution")

    plt.tight_layout()
    plt.savefig("output/shot_distribution.png")
    plt.show()





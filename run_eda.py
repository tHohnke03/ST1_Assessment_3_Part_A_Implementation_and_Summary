"""
*******************************
Author: u3243935
Assessment 3 - Macroinvertebrate Image Analysis System
Software Technology 1 (4483/8995)
Script: run_eda.py
Description: Standalone script to run Stage 1 EDA and print summary.
             Can be executed independently of the GUI:
               python run_eda.py
Date: 2024-05-12
*******************************
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from utils.dataset_loader import DatasetLoader
from stage1_eda.eda_analysis import EDAAnalyser

DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "eda")


def main():
    print("=== Stage 1: Exploratory Data Analysis ===\n")

    # Load dataset
    loader = DatasetLoader(DATA_DIR)
    print(f"Dataset loaded: {loader.summary()['total_images']} images, "
          f"{loader.summary()['num_classes']} classes")

    # Run EDA
    analyser = EDAAnalyser(loader, OUTPUT_DIR)
    print("\nDataset Summary:")
    print(analyser.get_summary_text())

    print("\nGenerating plots…")
    saved = analyser.run_full_eda()

    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    for path in saved:
        print(f"  ✓ {os.path.basename(path)}")

    print("\nEDA complete.")


if __name__ == "__main__":
    main()

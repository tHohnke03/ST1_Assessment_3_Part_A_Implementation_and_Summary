"""
*******************************
Author: u3243935
Assessment 3 - Macroinvertebrate Image Analysis System
Software Technology 1 (4483/8995)
Script: run_classifier.py
Description: Standalone script to run Stage 2 classification training.
             Trains a Random Forest by default; pass a model name as an
             argument to use another:
               python run_classifier.py random_forest
               python run_classifier.py svm
               python run_classifier.py knn
               python run_classifier.py gradient_boost
Date: 2024-05-12
*******************************
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from utils.dataset_loader import DatasetLoader
from stage2_classification.classifier import MacroinvertebrateClassifier

DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "models")


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "random_forest"

    print(f"=== Stage 2: Classification ({model_name}) ===\n")

    # Load dataset
    loader = DatasetLoader(DATA_DIR)

    # Train
    clf = MacroinvertebrateClassifier(loader, OUTPUT_DIR, model_name=model_name)
    results = clf.train()

    # Print results
    print("\n" + clf.get_results_text())

    # Save artefacts
    model_path = clf.save_model()
    cm_path    = clf.plot_confusion_matrix()
    rep_path   = clf.plot_class_report()
    fi_path    = clf.plot_feature_importance()

    print(f"\nArtefacts saved to: {OUTPUT_DIR}")
    print(f"  ✓ {os.path.basename(model_path)}")
    print(f"  ✓ {os.path.basename(cm_path)}")
    print(f"  ✓ {os.path.basename(rep_path)}")
    if fi_path:
        print(f"  ✓ {os.path.basename(fi_path)}")

    print("\nClassification complete.")


if __name__ == "__main__":
    main()

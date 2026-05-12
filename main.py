"""
*******************************
Author: u3243935
Assessment 3 - Macroinvertebrate Image Analysis System
Software Technology 1 (4483/8995)
Date: 2024-05-12
*******************************
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from stage3_gui.app import MacroinvertebrateApp


def main():
    """
    Entry point for the Macroinvertebrate Image Analysis System.
    Launches the Tkinter GUI application which integrates all three stages:
      - Stage 1: EDA (Exploratory Data Analysis)
      - Stage 2: Classification (Predictive Analytics)
      - Stage 3: GUI Application Deployment
    """
    app = MacroinvertebrateApp()
    app.run()


if __name__ == "__main__":
    main()

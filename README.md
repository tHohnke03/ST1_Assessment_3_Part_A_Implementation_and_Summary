# Macroinvertebrate Image Analysis System

**Software Technology 1 (4483/8995) — Assessment 3**  
**Student ID:** u3243935  
**Date:** 2024-05-12

---

## Overview

A Python desktop application for classification and analysis of freshwater macroinvertebrate images across 17 species. The system implements three integrated stages:

- **Stage 1:** Exploratory Data Analysis (EDA)
- **Stage 2:** Predictive Classification (Machine Learning)
- **Stage 3:** Desktop GUI Deployment (Tkinter)

Dataset: [Kaggle — Stream Macroinvertebrates](https://www.kaggle.com/datasets/kennethtm/stream-macroinvertebrates)

---

## Quick Start (Dataset Included)

The full dataset is included in this repository under `data/images/`.

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd Assessment3SoftwareTech

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the GUI
python3 main.py
```

That's it. The application will load the dataset automatically on startup.

---

## Alternative Setup (Download Dataset Manually)

If the dataset is not present in `data/images/`, download it from Kaggle and place the species folders inside `data/images/`:

1. Download from: https://www.kaggle.com/datasets/kennethtm/stream-macroinvertebrates
2. Unzip and copy each species folder into `data/images/` so the structure looks like:

```
data/images/
├── Asellus sp/
├── Baetidae sp/
├── Elmis sp/
├── Ephemerellidae/
├── Erpobdella sp/
├── Gammarus sp/
├── Hydropsychidae sp/
├── Leptophlebiidae sp/
├── Leuctra sp/
├── Limnius sp/
├── Lymnea sp/
├── Nemoura sp/
├── Oligochaeta sp/
├── Sericostomatidae sp/
├── Sialis sp/
├── Simuliidae sp/
└── Sphaerium sp/
```

3. Then run as above (`python3 main.py`). The `metadata.csv` file will be generated automatically on first run.

---

## Species Classes (17)

| Species | Species | Species |
|---|---|---|
| Asellus sp | Baetidae sp | Elmis sp |
| Ephemerellidae | Erpobdella sp | Gammarus sp |
| Hydropsychidae sp | Leptophlebiidae sp | Leuctra sp |
| Limnius sp | Lymnea sp | Nemoura sp |
| Oligochaeta sp | Sericostomatidae sp | Sialis sp |
| Simuliidae sp | Sphaerium sp | |

---

## Project Structure

```
Assessment3SoftwareTech/
├── main.py                         # Entry point — launches GUI
├── run_eda.py                      # Standalone EDA script (no GUI)
├── run_classifier.py               # Standalone classifier training (no GUI)
├── requirements.txt
├── README.md
├── IMPLEMENTATION_SUMMARY.md
│
├── utils/
│   ├── dataset_loader.py           # DatasetLoader class
│   └── image_processor.py          # ImageProcessor class (OpenCV)
│
├── stage1_eda/
│   └── eda_analysis.py             # EDAAnalyser class
│
├── stage2_classification/
│   └── classifier.py               # MacroinvertebrateClassifier class
│
├── stage3_gui/
│   └── app.py                      # MacroinvertebrateApp (Tkinter GUI)
│
├── data/
│   ├── metadata.csv                # Auto-generated on first run
│   └── images/                     # 2,665 images across 17 species
│
└── outputs/
    ├── eda/                        # EDA plots (generated on run)
    └── models/                     # Classifier artefacts (generated on run)
```

---

## Running Individual Stages (No GUI)

```bash
# Stage 1 — EDA only
python3 run_eda.py

# Stage 2 — Train classifier only
python3 run_classifier.py random_forest   # or: svm, knn, gradient_boost
```

---

## Using the GUI

| Tab | What to do |
|---|---|
| 📊 Dashboard | Loads automatically — shows dataset stats |
| 🔍 EDA | Click **Run Full EDA** to generate 6 analysis plots |
| 🤖 Classifier | Select a model, click **Train Model** |
| 🔮 Predict | Click **Browse…**, select any image, click **Predict** |
| ℹ️ About | Project and dataset information |

---

## Technology Stack

| Library | Purpose |
|---|---|
| `pandas` | Metadata management, tabular EDA |
| `numpy` | Array operations, feature engineering |
| `opencv-python` | Image loading, HOG, edge detection, colour spaces |
| `scikit-learn` | ML pipeline, Random Forest, SVM, KNN, evaluation |
| `matplotlib` | All plot generation |
| `seaborn` | Styled statistical visualisations |
| `Pillow` | Image I/O, thumbnail generation |
| `tkinter` | Desktop GUI framework (stdlib) |

---

## Classification Results (Random Forest, full dataset)

- **Test Accuracy:** 78.99%
- **CV Accuracy:** 80.08% ± 1.61%
- Note: Minority classes (Leuctra sp: 19 images, Leptophlebiidae sp: 9 images) show lower recall due to limited training data — this is a known characteristic of the dataset.

---

## References

IEEE format references are included in `IMPLEMENTATION_SUMMARY.md`.

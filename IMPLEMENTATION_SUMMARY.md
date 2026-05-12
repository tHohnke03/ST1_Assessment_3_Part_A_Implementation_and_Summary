# Implementation Summary

**Project Title:** Macroinvertebrate Image Analysis System  
**Unit:** Software Technology 1 (4483/8995) — Assessment 3  
**Student ID:** u3243935  
**Date:** 2024-05-12  
**Submission:** GitHub Repository Link

---

## 1. Project Goal

To design and implement a Python-based desktop application that classifies freshwater macroinvertebrate images into 17 species using the Kaggle Stream Macroinvertebrates dataset. The system integrates exploratory data analysis, machine learning classification, and a graphical user interface into a unified, well-structured software solution.

---

## 2. System Design

The application follows a layered, modular architecture with clear separation of concerns:

```
Presentation Layer   →  stage3_gui/app.py          (Tkinter GUI)
Business Logic       →  stage1_eda, stage2_classification
Data / Infrastructure→  utils/ (DatasetLoader, ImageProcessor)
```

All components are designed as Python **classes** with clearly defined responsibilities. Background threads are used for long-running tasks (EDA, training) to keep the UI responsive.

---

## 3. Class and Module Overview

### `utils/dataset_loader.py` — `DatasetLoader`
Loads image metadata from CSV or builds it automatically from the directory structure. Provides image arrays on demand and exposes dataset summary queries. Used by both EDA and classification stages.

### `utils/image_processor.py` — `ImageProcessor`
Wraps all OpenCV operations: preprocessing, colour space conversion, HOG feature extraction, colour histograms, texture statistics, edge detection, and data augmentation. A single reusable component shared across all stages.

### `stage1_eda/eda_analysis.py` — `EDAAnalyser`
Performs EDA by generating six plot types: class distribution bar chart, sample image grid, per-channel pixel intensity distributions, average colour swatches per class, image statistics and correlation heat-map, and Canny edge detection examples. Plots are saved as PNG files and browsable in the GUI.

### `stage2_classification/classifier.py` — `MacroinvertebrateClassifier`
Orchestrates the full ML pipeline: feature extraction (HOG + colour histogram + texture), train/test split, sklearn Pipeline training (StandardScaler + estimator), 5-fold cross-validation, confusion matrix, classification report, feature importance, and model persistence via pickle.

### `stage3_gui/app.py` — `MacroinvertebrateApp`
Tkinter application with five tabs: Dashboard, EDA, Classifier, Predict, and About. Integrates all other classes. Uses threading for non-blocking operations and embeds Matplotlib figures directly in the GUI.

---

## 4. Key Features Implemented

- **17-species classification** of stream macroinvertebrate images
- **HOG + colour histogram + texture** feature extraction pipeline
- **Four model options:** Random Forest, SVM, KNN, Gradient Boosting (selectable in GUI)
- **5-fold cross-validation** reporting
- **Confusion matrix + per-class classification report** visualisations
- **6 EDA plots** covering distribution, colour analysis, texture, and edge detection
- **Real-time image prediction** with probability bar chart in GUI
- **Model save/load** (pickle) for persistence between sessions
- **Dark-themed Tkinter GUI** with threaded background tasks

---

## 5. Tools and Libraries Used

| Library | Version (min) | Use |
|---|---|---|
| pandas | 2.0 | Metadata, tabular EDA |
| numpy | 1.24 | Array ops, feature math |
| opencv-python | 4.8 | HOG, histograms, edges, colour spaces |
| scikit-learn | 1.3 | ML pipeline, classifiers, metrics |
| matplotlib | 3.7 | All plots |
| seaborn | 0.13 | Styled heat-maps, distributions |
| Pillow | 10.0 | Image I/O, GUI thumbnails |
| tkinter | stdlib | Desktop GUI |

---

## 6. Dataset

**Kaggle Stream Macroinvertebrates** — 17 freshwater species photographed in laboratory conditions. Images are PNG format. The `DatasetLoader` class auto-discovers class folders and builds `metadata.csv` on first run, so no manual configuration is required when the dataset is in `data/images/`.

**Species included:**
Asellus sp, Baetidae sp, Elmis sp, Ephemerellidae, Erpobdella sp, Gammarus sp, Hydropsychidae sp, Leptophlebiidae sp, Leuctra sp, Limnius sp, Lymnea sp, Nemoura sp, Oligochaeta sp, Sericostomatidae sp, Sialis sp, Simuliidae sp, Sphaerium sp.

---

## 7. Testing Summary

Testing was performed at two levels:

**Unit-level:** Each class was exercised independently via the standalone scripts `run_eda.py` and `run_classifier.py` before GUI integration. Edge cases checked include missing class folders, unsupported image extensions, and missing model file on prediction.

**Integration-level:** The GUI was run end-to-end: dataset loading → EDA → model training → prediction on held-out images. All tabs were exercised and results verified against the standalone scripts.

---

## 8. Acknowledgement of Reused or Adapted Code

- The HOG descriptor parameters (win_size, block_size, cell_size, nbins) were adapted from the OpenCV HOGDescriptor documentation example [1].
- The Canny edge detection example was adapted from the OpenCV Canny tutorial [2].
- No unit tutorial or lab code was directly reused; all classes and methods are original implementations applying concepts taught in the unit.

---

## 9. Work Division

This project was completed individually (solo group).

| Component | Contribution |
|---|---|
| `utils/` (DatasetLoader, ImageProcessor) | 100% — data/image management foundation |
| `stage1_eda/` (EDAAnalyser) | 100% — all EDA plots and statistics |
| `stage2_classification/` (Classifier) | 100% — feature extraction and ML pipeline |
| `stage3_gui/` (App) | 100% — full Tkinter GUI integration |
| Documentation (README, Summary) | 100% |

---

## 10. References

[1] OpenCV Development Team, "cv::HOGDescriptor Class Reference," *OpenCV 4.x Documentation*, 2024. [Online]. Available: https://docs.opencv.org/4.x/d5/d33/structcv_1_1HOGDescriptor.html

[2] OpenCV Development Team, "Canny Edge Detector," *OpenCV-Python Tutorials*, 2024. [Online]. Available: https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html

[3] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011. [Online]. Available: https://scikit-learn.org

[4] W. McKinney, "Data Structures for Statistical Computing in Python," in *Proc. 9th Python in Science Conf.*, 2010, pp. 51–56. [Online]. Available: https://pandas.pydata.org

[5] K. T. M., "Stream Macroinvertebrates Dataset," *Kaggle*, 2023. [Online]. Available: https://www.kaggle.com/datasets/kennethtm/stream-macroinvertebrates

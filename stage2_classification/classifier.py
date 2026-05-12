"""
*******************************
Author: u3243935
Assessment 3 - Macroinvertebrate Image Analysis System
Software Technology 1 (4483/8995)
Module: stage2_classification/classifier.py
Description: MacroinvertebrateClassifier class — trains, evaluates, and
             persists a Scikit-learn classification pipeline using HOG +
             colour histogram + texture features. Supports Random Forest,
             SVM, KNN, and Gradient Boosting. Handles class imbalance via
             class_weight='balanced'. Includes confusion matrix and
             classification report visualisations.
Date: 2024-05-12
*******************************
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

from utils.dataset_loader import DatasetLoader
from utils.image_processor import ImageProcessor


class MacroinvertebrateClassifier:
    """
    Encapsulates the full classification pipeline for macroinvertebrate images.

    Workflow:
      1. Feature extraction (HOG + colour histogram + texture)
      2. Stratified train/test split
      3. Model training with class-weight balancing
      4. Evaluation and visualisation
      5. Model persistence (save/load)

    The dataset is imbalanced (Gammarus sp: 987 images vs
    Leptophlebiidae sp: 9 images), so class_weight='balanced'
    is applied where supported to avoid biasing predictions toward
    majority classes.

    Attributes:
        loader (DatasetLoader): Dataset loader instance.
        processor (ImageProcessor): Image processor for feature extraction.
        output_dir (str): Directory where model/results are saved.
        model_name (str): Active model identifier.
        pipeline (Pipeline): Trained sklearn pipeline.
        label_encoder (LabelEncoder): Encodes class names to integers.
        results (dict): Latest evaluation results.
    """

    # Models with class_weight='balanced' where supported to handle
    # the significant class imbalance in the stream macroinvertebrates dataset.
    AVAILABLE_MODELS = {
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "svm": SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=42,
        ),
        "knn": KNeighborsClassifier(
            n_neighbors=5,
            metric="euclidean",
        ),
        "gradient_boost": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
        ),
    }

    def __init__(
        self,
        loader: DatasetLoader,
        output_dir: str,
        model_name: str = "random_forest",
    ):
        """
        Initialise the classifier.

        Args:
            loader (DatasetLoader): Initialised dataset loader.
            output_dir (str): Directory where artefacts are saved.
            model_name (str): Key from AVAILABLE_MODELS to use.

        Raises:
            ValueError: If model_name is not recognised.
        """
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Choose from {list(self.AVAILABLE_MODELS.keys())}."
            )

        self.loader = loader
        self.processor = ImageProcessor()
        self.output_dir = output_dir
        self.model_name = model_name
        self.pipeline: Pipeline | None = None
        self.label_encoder = LabelEncoder()
        self.results: dict = {}

        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_features(self, images: np.ndarray) -> np.ndarray:
        """
        Extract a combined feature vector from each image.

        Features per image:
          - Colour histogram  : 96 dims  (32 bins × 3 channels, L1-normalised)
          - HOG descriptor    : variable (shape/texture gradients)
          - Texture statistics: 3 dims   (Laplacian variance, Sobel mean/std)

        Args:
            images (np.ndarray): Array of shape (N, H, W, 3), uint8.

        Returns:
            np.ndarray: Feature matrix of shape (N, D).
        """
        return np.array([
            self.processor.extract_all_features(img) for img in images
        ])

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        test_size: float = 0.2,
        image_size: tuple[int, int] = (64, 64),
    ) -> dict:
        """
        Load data, extract features, train the model, and evaluate.

        Uses stratified splitting to ensure every class is represented
        in both train and test sets, even for minority classes.

        Args:
            test_size (float): Fraction of data reserved for testing.
            image_size (tuple): Resize target before feature extraction.

        Returns:
            dict: Evaluation results (accuracy, cv_mean, cv_std, report,
                  confusion_matrix, class_names).
        """
        # 1. Load all images
        print(f"[Classifier] Loading {len(self.loader.metadata)} images…")
        X_raw, y = self.loader.load_all_images(size=image_size)
        class_names = self.loader.class_names

        # 2. Extract features
        print("[Classifier] Extracting features…")
        X = self.extract_features(X_raw)

        # 3. Stratified split — preserves class proportions in both sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # 4. Build sklearn pipeline
        estimator = self.AVAILABLE_MODELS[self.model_name]
        self.pipeline = Pipeline([
            ("scaler",     StandardScaler()),
            ("classifier", estimator),
        ])

        # 5. Fit
        print(f"[Classifier] Training {self.model_name}…")
        self.pipeline.fit(X_train, y_train)

        # 6. Evaluate on test set
        #    zero_division=0 silences warnings for minority classes that
        #    have no predicted samples in the test split.
        y_pred = self.pipeline.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test, y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(y_test, y_pred)

        # 7. 5-fold cross-validation on full dataset
        cv_scores = cross_val_score(
            self.pipeline, X, y, cv=5, scoring="accuracy"
        )

        self.results = {
            "model_name":       self.model_name,
            "accuracy":         acc,
            "cv_mean":          cv_scores.mean(),
            "cv_std":           cv_scores.std(),
            "report":           report,
            "confusion_matrix": cm,
            "class_names":      class_names,
            "X_test":           X_test,
            "y_test":           y_test,
            "y_pred":           y_pred,
        }

        print(f"[Classifier] Test accuracy: {acc:.4f}")
        print(f"[Classifier] CV accuracy  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        return self.results

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, image: np.ndarray) -> tuple[str, float, dict]:
        """
        Predict the species of a single image.

        Args:
            image (np.ndarray): Single image (H, W, 3), uint8.

        Returns:
            tuple:
                predicted_class (str): Species label.
                confidence (float): Probability of predicted class.
                all_probs (dict): {species: probability} for all classes.

        Raises:
            RuntimeError: If model has not been trained or loaded.
        """
        if self.pipeline is None:
            raise RuntimeError("Model not trained. Call train() or load_model() first.")

        feat       = self.processor.extract_all_features(image).reshape(1, -1)
        pred_idx   = self.pipeline.predict(feat)[0]
        probs      = self.pipeline.predict_proba(feat)[0]
        class_names = self.loader.class_names

        predicted_class = class_names[pred_idx]
        confidence      = float(probs[pred_idx])
        all_probs       = {cls: float(p) for cls, p in zip(class_names, probs)}

        return predicted_class, confidence, all_probs

    # ------------------------------------------------------------------
    # Visualisations
    # ------------------------------------------------------------------

    def plot_confusion_matrix(self) -> str:
        """
        Save a heat-map of the confusion matrix from the last training run.

        Returns:
            str: Path to saved figure.
        """
        if not self.results:
            raise RuntimeError("No results available. Call train() first.")

        cm          = self.results["confusion_matrix"]
        class_names = self.results["class_names"]

        fig, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.4, ax=ax,
        )
        ax.set_title(
            f"Confusion Matrix — {self.model_name.replace('_', ' ').title()}\n"
            f"Test Accuracy: {self.results['accuracy']:.2%}",
            fontsize=13, fontweight="bold",
        )
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        plt.xticks(rotation=40, ha="right", fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        path = os.path.join(self.output_dir, "confusion_matrix.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_class_report(self) -> str:
        """
        Bar chart of per-class precision, recall, and F1-score.

        Returns:
            str: Path to saved figure.
        """
        if not self.results:
            raise RuntimeError("No results available. Call train() first.")

        report      = self.results["report"]
        class_names = self.results["class_names"]

        metrics = {
            cls: {
                "Precision": report[cls]["precision"],
                "Recall":    report[cls]["recall"],
                "F1":        report[cls]["f1-score"],
            }
            for cls in class_names
        }
        df    = pd.DataFrame(metrics).T
        x     = np.arange(len(class_names))
        width = 0.26

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.bar(x - width, df["Precision"], width, label="Precision", color="#3498db", alpha=0.85)
        ax.bar(x,          df["Recall"],   width, label="Recall",    color="#2ecc71", alpha=0.85)
        ax.bar(x + width,  df["F1"],       width, label="F1-Score",  color="#e74c3c", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, 1.2)
        ax.set_ylabel("Score")
        ax.set_title(
            f"Per-Class Metrics — {self.model_name.replace('_', ' ').title()}",
            fontsize=13, fontweight="bold",
        )
        ax.legend()
        ax.axhline(0.8, color="gray", linestyle="--", linewidth=0.8)
        plt.tight_layout()
        path = os.path.join(self.output_dir, "class_report.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_feature_importance(self) -> str | None:
        """
        Bar chart of top-20 feature importances (Random Forest / GB only).

        Returns:
            str | None: Path to saved figure, or None if not supported.
        """
        if not self.results or self.pipeline is None:
            return None

        estimator = self.pipeline.named_steps["classifier"]
        if not hasattr(estimator, "feature_importances_"):
            return None

        importances = estimator.feature_importances_
        indices     = np.argsort(importances)[::-1][:20]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(20), importances[indices], color="#9b59b6", alpha=0.85)
        ax.set_title("Top 20 Feature Importances", fontsize=13, fontweight="bold")
        ax.set_xlabel("Feature Index (HOG + Colour Histogram + Texture)")
        ax.set_ylabel("Importance")
        plt.tight_layout()
        path = os.path.join(self.output_dir, "feature_importance.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, filename: str = "model.pkl") -> str:
        """
        Serialise the trained pipeline to disk using pickle.

        Args:
            filename (str): Output filename inside output_dir.

        Returns:
            str: Full path to saved model file.
        """
        if self.pipeline is None:
            raise RuntimeError("No trained model to save.")
        path = os.path.join(self.output_dir, filename)
        with open(path, "wb") as f:
            pickle.dump({
                "pipeline":    self.pipeline,
                "label_encoder": self.label_encoder,
                "model_name":  self.model_name,
                "class_names": self.loader.class_names,
            }, f)
        print(f"[Classifier] Model saved → {path}")
        return path

    def load_model(self, filepath: str) -> None:
        """
        Load a previously saved pipeline from disk.

        Args:
            filepath (str): Path to the .pkl model file.
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.pipeline      = data["pipeline"]
        self.label_encoder = data["label_encoder"]
        self.model_name    = data["model_name"]
        print(f"[Classifier] Model loaded ← {filepath}")

    # ------------------------------------------------------------------
    # Text summary
    # ------------------------------------------------------------------

    def get_results_text(self) -> str:
        """
        Return a formatted text summary of the last training run.

        Returns:
            str: Multi-line results summary.
        """
        if not self.results:
            return "No training results available."

        lines = [
            f"=== Classification Results ===",
            f"Model         : {self.results['model_name'].replace('_', ' ').title()}",
            f"Test Accuracy : {self.results['accuracy']:.4f}",
            f"CV Accuracy   : {self.results['cv_mean']:.4f} ± {self.results['cv_std']:.4f}",
            "",
            "Per-class metrics (P=Precision, R=Recall, F1=F1-Score):",
            f"  Note: Classes with very few images (e.g. Leptophlebiidae sp: 9 images)",
            f"        may show 0.00 scores due to insufficient test samples.",
            "",
        ]
        for cls in self.results["class_names"]:
            r = self.results["report"][cls]
            lines.append(
                f"  {cls:<22}  P={r['precision']:.2f}  R={r['recall']:.2f}  F1={r['f1-score']:.2f}"
            )
        return "\n".join(lines)

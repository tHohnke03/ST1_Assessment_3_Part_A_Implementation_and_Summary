"""
*******************************
Author: u3243935
Assessment 3 - Macroinvertebrate Image Analysis System
Software Technology 1 (4483/8995)
Module: utils/feature_extractor.py
Description: FeatureExtractor class — dedicated module for extracting
             structured feature vectors from macroinvertebrate images.
             Separates feature engineering concerns from the classifier.
             Uses OpenCV HOG, colour histograms, and texture statistics.
Date: 2024-05-12
*******************************
"""

import numpy as np
import cv2


class FeatureExtractor:
    """
    Responsible for extracting numerical feature vectors from images
    for use in machine learning classification.

    Separating feature extraction into its own class keeps the
    classifier focused on model training and evaluation, and makes
    it easy to swap or extend the feature set independently.

    Feature vector composition:
      1. Colour histogram  — 96 dims  (32 bins × 3 RGB channels, L1-normalised)
      2. HOG descriptor    — ~1764 dims (gradient orientation histogram)
      3. Texture stats     — 3 dims   (Laplacian variance, Sobel mean/std)

    Total feature vector length: ~1863 dimensions per image.
    """

    # HOG parameters — fixed for consistent feature vector length across all images
    HOG_WIN_SIZE     = (64, 64)
    HOG_BLOCK_SIZE   = (16, 16)
    HOG_BLOCK_STRIDE = (8, 8)
    HOG_CELL_SIZE    = (8, 8)
    HOG_NBINS        = 9
    HIST_BINS        = 32

    def __init__(self):
        """
        Initialise the HOG descriptor with fixed parameters.
        Fixed parameters ensure every image produces the same length
        feature vector regardless of original image size.
        """
        self._hog = cv2.HOGDescriptor(
            self.HOG_WIN_SIZE,
            self.HOG_BLOCK_SIZE,
            self.HOG_BLOCK_STRIDE,
            self.HOG_CELL_SIZE,
            self.HOG_NBINS,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract a combined feature vector from a single image.

        Args:
            image (np.ndarray): RGB image array (H, W, 3), uint8.

        Returns:
            np.ndarray: 1D feature vector of consistent length.
        """
        colour  = self._colour_histogram(image)
        hog     = self._hog_features(image)
        texture = self._texture_features(image)
        return np.concatenate([colour, hog, texture])

    def extract_batch(self, images: np.ndarray) -> np.ndarray:
        """
        Extract feature vectors from a batch of images.

        Args:
            images (np.ndarray): Array of shape (N, H, W, 3), uint8.

        Returns:
            np.ndarray: Feature matrix of shape (N, D).
        """
        return np.array([self.extract(img) for img in images])

    def feature_names(self) -> list[str]:
        """
        Return human-readable names for the feature groups.
        Useful for feature importance interpretation.

        Returns:
            list[str]: ['colour_hist_0..95', 'hog_0..N', 'texture_0..2']
        """
        names = (
            [f"colour_hist_{i}" for i in range(self.HIST_BINS * 3)] +
            [f"hog_{i}" for i in range(self._hog_feature_length())] +
            ["texture_laplacian_var", "texture_sobel_mean", "texture_sobel_std"]
        )
        return names

    # ------------------------------------------------------------------
    # Private feature methods
    # ------------------------------------------------------------------

    def _colour_histogram(self, image: np.ndarray) -> np.ndarray:
        """
        Compute an L1-normalised RGB colour histogram.

        Captures the overall colour distribution of the image,
        which varies meaningfully between species (e.g. Oligochaeta sp
        are typically darker than Gammarus sp).

        Args:
            image (np.ndarray): RGB image (H, W, 3), uint8.

        Returns:
            np.ndarray: Normalised histogram, length = HIST_BINS * 3.
        """
        features = []
        for channel in range(3):
            hist = cv2.calcHist([image], [channel], None, [self.HIST_BINS], [0, 256])
            features.append(hist.flatten())
        vec = np.concatenate(features)
        return vec / (vec.sum() + 1e-8)  # L1 normalise

    def _hog_features(self, image: np.ndarray) -> np.ndarray:
        """
        Compute Histogram of Oriented Gradients (HOG) features.

        HOG captures shape and texture through gradient orientations,
        which is effective for distinguishing body morphology between
        macroinvertebrate species (e.g. elongated Erpobdella vs
        rounded Sphaerium).

        Args:
            image (np.ndarray): RGB image (H, W, 3), uint8.

        Returns:
            np.ndarray: HOG feature vector.
        """
        resized = cv2.resize(image, self.HOG_WIN_SIZE)
        gray    = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        return self._hog.compute(gray).flatten()

    def _texture_features(self, image: np.ndarray) -> np.ndarray:
        """
        Compute texture statistics via Laplacian variance and Sobel gradients.

        Laplacian variance measures image sharpness/complexity.
        Sobel mean/std captures average edge strength and its spread.
        Together these distinguish smooth-bodied species (e.g. worm-like
        Oligochaeta) from textured ones (e.g. Sericostomatidae cases).

        Args:
            image (np.ndarray): RGB image (H, W, 3), uint8.

        Returns:
            np.ndarray: [laplacian_variance, sobel_mean, sobel_std]
        """
        gray      = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        lap_var   = cv2.Laplacian(gray, cv2.CV_64F).var()
        sobelx    = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely    = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        return np.array([lap_var, magnitude.mean(), magnitude.std()])

    def _hog_feature_length(self) -> int:
        """Return the HOG descriptor length for a dummy image."""
        dummy = np.zeros((*self.HOG_WIN_SIZE, 3), dtype=np.uint8)
        return len(self._hog_features(dummy))

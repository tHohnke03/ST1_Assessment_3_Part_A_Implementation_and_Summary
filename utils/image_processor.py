"""
*******************************
Author: u3243935
Assessment 3 - Macroinvertebrate Image Analysis System
Software Technology 1 (4483/8995)
Module: utils/image_processor.py
Description: ImageProcessor class providing OpenCV-based
             preprocessing and feature extraction utilities.
Date: 2024-05-12
*******************************
"""

import cv2
import numpy as np
from PIL import Image


class ImageProcessor:
    """
    Provides static and instance methods for image preprocessing,
    augmentation, and feature extraction using OpenCV.

    Used by both the EDA stage (for visualisations) and the
    classification stage (for feature engineering).
    """

    # Colour space options supported by to_colourspace()
    SUPPORTED_COLOURSPACES = ("RGB", "GRAY", "HSV", "LAB")

    def __init__(self, target_size: tuple[int, int] = (128, 128)):
        """
        Initialise the processor with a default target image size.

        Args:
            target_size (tuple): (width, height) to resize images to.
        """
        self.target_size = target_size

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Standard preprocessing pipeline:
          1. Resize to target_size
          2. Normalise pixel values to [0, 1] float32

        Args:
            image (np.ndarray): Raw image array (H, W, 3) uint8.

        Returns:
            np.ndarray: Preprocessed image, float32 in [0, 1].
        """
        resized = cv2.resize(image, self.target_size)
        return resized.astype(np.float32) / 255.0

    def preprocess_batch(self, images: np.ndarray) -> np.ndarray:
        """
        Apply preprocess() to a batch of images.

        Args:
            images (np.ndarray): Array of shape (N, H, W, 3).

        Returns:
            np.ndarray: Preprocessed batch, shape (N, target_h, target_w, 3).
        """
        return np.array([self.preprocess(img) for img in images])

    # ------------------------------------------------------------------
    # Colour space conversion
    # ------------------------------------------------------------------

    @staticmethod
    def to_colourspace(image: np.ndarray, space: str) -> np.ndarray:
        """
        Convert an RGB image to the specified colour space.

        Args:
            image (np.ndarray): Input image (H, W, 3), RGB, uint8.
            space (str): One of 'GRAY', 'HSV', 'LAB', 'RGB'.

        Returns:
            np.ndarray: Converted image array.

        Raises:
            ValueError: If an unsupported colour space is requested.
        """
        if space == "RGB":
            return image
        elif space == "GRAY":
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif space == "HSV":
            return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif space == "LAB":
            return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        else:
            raise ValueError(
                f"Unsupported colour space '{space}'. "
                f"Choose from {ImageProcessor.SUPPORTED_COLOURSPACES}."
            )

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_colour_histogram(
        image: np.ndarray, bins: int = 32
    ) -> np.ndarray:
        """
        Compute a flattened RGB colour histogram feature vector.

        Args:
            image (np.ndarray): Input image (H, W, 3), uint8.
            bins (int): Number of histogram bins per channel.

        Returns:
            np.ndarray: Feature vector of length 3 * bins, normalised.
        """
        features = []
        for channel in range(3):
            hist = cv2.calcHist(
                [image], [channel], None, [bins], [0, 256]
            )
            features.append(hist.flatten())
        vec = np.concatenate(features)
        # L1 normalise so brightness doesn't dominate
        norm = vec.sum()
        return vec / (norm + 1e-8)

    @staticmethod
    def extract_hog_features(image: np.ndarray) -> np.ndarray:
        """
        Compute Histogram of Oriented Gradients (HOG) features.
        Resizes image to 64x64 internally.

        Args:
            image (np.ndarray): Input image (H, W, 3), uint8.

        Returns:
            np.ndarray: HOG feature vector.
        """
        resized = cv2.resize(image, (64, 64))
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        win_size = (64, 64)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        return hog.compute(gray).flatten()

    @staticmethod
    def extract_texture_features(image: np.ndarray) -> np.ndarray:
        """
        Compute Laplacian variance (a proxy for texture/sharpness)
        and Sobel gradient statistics.

        Args:
            image (np.ndarray): Input image (H, W, 3), uint8.

        Returns:
            np.ndarray: Feature vector [laplacian_var, sobel_mean, sobel_std].
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        return np.array([lap_var, magnitude.mean(), magnitude.std()])

    def extract_all_features(self, image: np.ndarray) -> np.ndarray:
        """
        Combine colour histogram, HOG, and texture features into one vector.

        Args:
            image (np.ndarray): Input image (H, W, 3), uint8.

        Returns:
            np.ndarray: Concatenated feature vector.
        """
        colour = self.extract_colour_histogram(image)
        hog = self.extract_hog_features(image)
        texture = self.extract_texture_features(image)
        return np.concatenate([colour, hog, texture])

    # ------------------------------------------------------------------
    # Augmentation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def augment(image: np.ndarray) -> np.ndarray:
        """
        Apply a random combination of augmentations (flip, rotate, brightness)
        for training data diversity.

        Args:
            image (np.ndarray): Input image (H, W, 3), uint8.

        Returns:
            np.ndarray: Augmented image.
        """
        rng = np.random.default_rng()
        img = image.copy()
        # Random horizontal flip
        if rng.random() > 0.5:
            img = cv2.flip(img, 1)
        # Random rotation ±15 degrees
        angle = rng.uniform(-15, 15)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h))
        # Random brightness
        factor = rng.uniform(0.7, 1.3)
        img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return img

    # ------------------------------------------------------------------
    # Edge / segmentation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def detect_edges(image: np.ndarray) -> np.ndarray:
        """
        Apply Canny edge detection to a grayscale version of the image.

        Args:
            image (np.ndarray): Input image (H, W, 3), uint8.

        Returns:
            np.ndarray: Binary edge map (H, W), uint8.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.Canny(blurred, threshold1=30, threshold2=100)

    @staticmethod
    def pil_to_cv(pil_image: Image.Image) -> np.ndarray:
        """Convert a PIL Image to an OpenCV-compatible NumPy array (RGB)."""
        return np.array(pil_image.convert("RGB"))

    @staticmethod
    def cv_to_pil(cv_image: np.ndarray) -> Image.Image:
        """Convert a NumPy array (RGB) to a PIL Image."""
        return Image.fromarray(cv_image.astype(np.uint8))

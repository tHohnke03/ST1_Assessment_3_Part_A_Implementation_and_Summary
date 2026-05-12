"""
*******************************
Author: u3243935
Assessment 3 - Macroinvertebrate Image Analysis System
Software Technology 1 (4483/8995)
Module: tests/test_image_processor.py
Description: Unit tests for the ImageProcessor class.
             Run with: python3 -m pytest tests/ -v
Date: 2024-05-12
*******************************
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from utils.image_processor import ImageProcessor


class TestImageProcessor:
    """Unit tests for ImageProcessor."""

    def setup_method(self):
        self.processor   = ImageProcessor(target_size=(64, 64))
        self.dummy_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)

    def test_preprocess_output_shape(self):
        """preprocess() should resize to target_size."""
        result = self.processor.preprocess(self.dummy_image)
        assert result.shape == (64, 64, 3)

    def test_preprocess_normalised_range(self):
        """preprocess() should produce float32 values in [0, 1]."""
        result = self.processor.preprocess(self.dummy_image)
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_preprocess_batch_shape(self):
        """preprocess_batch() should return (N, H, W, 3)."""
        batch  = np.random.randint(0, 255, (4, 100, 100, 3), dtype=np.uint8)
        result = self.processor.preprocess_batch(batch)
        assert result.shape == (4, 64, 64, 3)

    def test_to_colourspace_gray(self):
        """GRAY conversion should return a 2D array."""
        gray = ImageProcessor.to_colourspace(self.dummy_image, "GRAY")
        assert gray.ndim == 2

    def test_to_colourspace_hsv_shape(self):
        """HSV conversion should preserve shape (H, W, 3)."""
        hsv = ImageProcessor.to_colourspace(self.dummy_image, "HSV")
        assert hsv.shape == self.dummy_image.shape

    def test_to_colourspace_invalid_raises(self):
        """Invalid colour space should raise ValueError."""
        with pytest.raises(ValueError):
            ImageProcessor.to_colourspace(self.dummy_image, "XYZ")

    def test_detect_edges_output_shape(self):
        """detect_edges() should return a 2D binary array same H/W as input."""
        edges = ImageProcessor.detect_edges(self.dummy_image)
        assert edges.ndim == 2
        assert edges.shape == (128, 128)

    def test_augment_preserves_shape(self):
        """augment() should return an image of the same shape."""
        augmented = ImageProcessor.augment(self.dummy_image)
        assert augmented.shape == self.dummy_image.shape

    def test_pil_cv_roundtrip(self):
        """Converting PIL → CV → PIL should preserve image dimensions."""
        from PIL import Image
        pil_img  = Image.fromarray(self.dummy_image)
        cv_arr   = ImageProcessor.pil_to_cv(pil_img)
        back_pil = ImageProcessor.cv_to_pil(cv_arr)
        assert back_pil.size == pil_img.size

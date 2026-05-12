"""
*******************************
Author: u3243935
Assessment 3 - Macroinvertebrate Image Analysis System
Software Technology 1 (4483/8995)
Module: tests/test_feature_extractor.py
Description: Unit tests for the FeatureExtractor class.
             Run with: python3 -m pytest tests/ -v
Date: 2024-05-12
*******************************
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from utils.feature_extractor import FeatureExtractor


class TestFeatureExtractor:
    """Unit tests for FeatureExtractor."""

    def setup_method(self):
        """Create a shared extractor and dummy test image before each test."""
        self.extractor = FeatureExtractor()
        self.dummy_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)

    def test_extract_returns_1d_array(self):
        """extract() should return a 1D numpy array."""
        result = self.extractor.extract(self.dummy_image)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1

    def test_extract_consistent_length(self):
        """Two different images should produce vectors of the same length."""
        img1 = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        vec1 = self.extractor.extract(img1)
        vec2 = self.extractor.extract(img2)
        assert len(vec1) == len(vec2), \
            "Feature vectors must be the same length regardless of input size"

    def test_extract_no_nan_or_inf(self):
        """Feature vector should contain no NaN or infinite values."""
        result = self.extractor.extract(self.dummy_image)
        assert not np.any(np.isnan(result)), "Feature vector contains NaN"
        assert not np.any(np.isinf(result)), "Feature vector contains Inf"

    def test_colour_histogram_normalised(self):
        """Colour histogram portion should be L1-normalised (sum ≈ 1.0)."""
        result = self.extractor.extract(self.dummy_image)
        hist_portion = result[:self.extractor.HIST_BINS * 3]
        assert abs(hist_portion.sum() - 1.0) < 1e-5, \
            "Colour histogram is not L1-normalised"

    def test_extract_batch_shape(self):
        """extract_batch() should return shape (N, D)."""
        batch = np.random.randint(0, 255, (5, 128, 128, 3), dtype=np.uint8)
        result = self.extractor.extract_batch(batch)
        assert result.ndim == 2
        assert result.shape[0] == 5

    def test_extract_batch_consistent_with_single(self):
        """Each row of extract_batch() should match extract() on the same image."""
        batch = np.random.randint(0, 255, (3, 128, 128, 3), dtype=np.uint8)
        batch_result = self.extractor.extract_batch(batch)
        for i in range(3):
            single = self.extractor.extract(batch[i])
            np.testing.assert_array_almost_equal(batch_result[i], single)

    def test_different_images_produce_different_vectors(self):
        """Two clearly different images should not produce identical vectors."""
        black_img = np.zeros((128, 128, 3), dtype=np.uint8)
        white_img = np.full((128, 128, 3), 255, dtype=np.uint8)
        vec_black = self.extractor.extract(black_img)
        vec_white = self.extractor.extract(white_img)
        assert not np.allclose(vec_black, vec_white), \
            "Black and white images produced identical feature vectors"

    def test_feature_names_length_matches_vector(self):
        """feature_names() length should match the feature vector length."""
        vec   = self.extractor.extract(self.dummy_image)
        names = self.extractor.feature_names()
        assert len(names) == len(vec), \
            f"feature_names() length ({len(names)}) != vector length ({len(vec)})"

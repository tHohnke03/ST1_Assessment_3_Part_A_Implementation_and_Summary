"""
*******************************
Author: u3243935
Assessment 3 - Macroinvertebrate Image Analysis System
Software Technology 1 (4483/8995)
Module: tests/test_dataset_loader.py
Description: Unit tests for the DatasetLoader class.
             Run with: python3 -m pytest tests/ -v
Date: 2024-05-12
*******************************
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import pytest
import tempfile
import shutil
from PIL import Image
from utils.dataset_loader import DatasetLoader


@pytest.fixture
def temp_dataset():
    """
    Create a temporary dataset directory with 2 classes, 3 images each.
    Yields the path and cleans up after the test.
    """
    tmp = tempfile.mkdtemp()
    classes = ["ClassA", "ClassB"]
    for cls in classes:
        cls_dir = os.path.join(tmp, "images", cls)
        os.makedirs(cls_dir)
        for i in range(3):
            img = Image.fromarray(
                np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            )
            img.save(os.path.join(cls_dir, f"{cls}_{i}.png"))
    yield tmp
    shutil.rmtree(tmp)


class TestDatasetLoader:
    """Unit tests for DatasetLoader."""

    def test_raises_on_missing_directory(self):
        """Should raise FileNotFoundError for a non-existent path."""
        with pytest.raises(FileNotFoundError):
            DatasetLoader("/nonexistent/path/to/data")

    def test_loads_correct_number_of_images(self, temp_dataset):
        """Should find all 6 images across 2 classes."""
        loader = DatasetLoader(temp_dataset)
        assert len(loader.metadata) == 6

    def test_class_names_detected_correctly(self, temp_dataset):
        """Should detect both class names from folder structure."""
        loader = DatasetLoader(temp_dataset)
        assert set(loader.class_names) == {"ClassA", "ClassB"}

    def test_metadata_has_required_columns(self, temp_dataset):
        """metadata DataFrame should have all required columns."""
        loader = DatasetLoader(temp_dataset)
        for col in ["filename", "class", "width", "height", "size_bytes", "filepath"]:
            assert col in loader.metadata.columns, f"Missing column: {col}"

    def test_class_counts_correct(self, temp_dataset):
        """class_counts() should return 3 for each class."""
        loader = DatasetLoader(temp_dataset)
        counts = loader.class_counts()
        for cls in ["ClassA", "ClassB"]:
            assert counts[cls] == 3

    def test_load_image_returns_array(self, temp_dataset):
        """load_image() should return a uint8 numpy array."""
        loader = DatasetLoader(temp_dataset)
        row    = loader.metadata.iloc[0]
        img    = loader.load_image(row["filepath"])
        assert isinstance(img, np.ndarray)
        assert img.dtype == np.uint8
        assert img.ndim == 3

    def test_load_image_respects_size(self, temp_dataset):
        """load_image() with size arg should resize the image."""
        loader = DatasetLoader(temp_dataset)
        row    = loader.metadata.iloc[0]
        img    = loader.load_image(row["filepath"], size=(32, 32))
        assert img.shape == (32, 32, 3)

    def test_load_all_images_shapes(self, temp_dataset):
        """load_all_images() should return X of shape (N, H, W, 3) and y of shape (N,)."""
        loader = DatasetLoader(temp_dataset)
        X, y   = loader.load_all_images(size=(32, 32))
        assert X.shape == (6, 32, 32, 3)
        assert y.shape == (6,)

    def test_get_sample_count(self, temp_dataset):
        """get_sample() should return at most n rows."""
        loader = DatasetLoader(temp_dataset)
        sample = loader.get_sample(n=2)
        assert len(sample) == 2

    def test_summary_keys(self, temp_dataset):
        """summary() should return all expected keys."""
        loader = DatasetLoader(temp_dataset)
        s = loader.summary()
        for key in ["total_images", "num_classes", "class_names", "avg_width", "avg_height"]:
            assert key in s

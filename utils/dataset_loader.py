"""
*******************************
Author: u3243935
Assessment 3 - Macroinvertebrate Image Analysis System
Software Technology 1 (4483/8995)
Module: utils/dataset_loader.py
Description: DatasetLoader class for loading and managing
             macroinvertebrate image data from disk.
Date: 2024-05-12
*******************************
"""

import os
import json
import pandas as pd
import numpy as np
from PIL import Image


class DatasetLoader:
    """
    Handles loading, caching, and basic querying of the
    macroinvertebrate image dataset.

    Attributes:
        data_dir (str): Root directory containing 'images/' subfolder
                        and 'metadata.csv'.
        metadata (pd.DataFrame): DataFrame of image metadata.
        class_names (list): Sorted list of unique class labels.
    """

    SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

    def __init__(self, data_dir: str):
        """
        Initialise the DatasetLoader with the path to the data directory.

        Args:
            data_dir (str): Path to the dataset root folder.

        Raises:
            FileNotFoundError: If data_dir does not exist.
        """
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "images")
        self._metadata_cache: pd.DataFrame | None = None
        self._image_cache: dict[str, np.ndarray] = {}

        self.metadata = self._load_or_build_metadata()
        self.class_names = sorted(self.metadata["class"].unique().tolist())

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _load_or_build_metadata(self) -> pd.DataFrame:
        """
        Load metadata.csv if it exists; otherwise scan the images folder
        and build metadata from the directory structure.

        Returns:
            pd.DataFrame: Metadata table with columns:
                          filename, class, width, height, size_bytes, filepath.
        """
        csv_path = os.path.join(self.data_dir, "metadata.csv")
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path)
            # Attach absolute filepath column
            df["filepath"] = df.apply(
                lambda row: os.path.join(
                    self.images_dir, row["class"], row["filename"]
                ),
                axis=1,
            )
            return df

        return self._build_metadata_from_disk()

    def _build_metadata_from_disk(self) -> pd.DataFrame:
        """
        Walk images_dir and collect per-image metadata.

        Returns:
            pd.DataFrame: Built metadata table.
        """
        records = []
        for class_name in os.listdir(self.images_dir):
            class_path = os.path.join(self.images_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            for fname in os.listdir(class_path):
                if not fname.lower().endswith(self.SUPPORTED_EXTENSIONS):
                    continue
                fpath = os.path.join(class_path, fname)
                try:
                    with Image.open(fpath) as img:
                        w, h = img.size
                except Exception:
                    w, h = 0, 0
                records.append(
                    {
                        "filename": fname,
                        "class": class_name,
                        "width": w,
                        "height": h,
                        "size_bytes": os.path.getsize(fpath),
                        "filepath": fpath,
                    }
                )
        df = pd.DataFrame(records)
        df.to_csv(os.path.join(self.data_dir, "metadata.csv"), index=False)
        return df

    # ------------------------------------------------------------------
    # Image access
    # ------------------------------------------------------------------

    def load_image(self, filepath: str, size: tuple[int, int] | None = None) -> np.ndarray:
        """
        Load a single image as a NumPy array (RGB, uint8).

        Args:
            filepath (str): Absolute path to the image file.
            size (tuple, optional): If given, resize to (width, height).

        Returns:
            np.ndarray: Image array of shape (H, W, 3).
        """
        key = filepath if size is None else f"{filepath}_{size}"
        if key not in self._image_cache:
            with Image.open(filepath) as img:
                img = img.convert("RGB")
                if size is not None:
                    img = img.resize(size, Image.LANCZOS)
                self._image_cache[key] = np.array(img)
        return self._image_cache[key]

    def load_all_images(
        self, size: tuple[int, int] = (64, 64)
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Load all images into memory as arrays.

        Args:
            size (tuple): Resize target for each image.

        Returns:
            tuple:
                X (np.ndarray): Image array, shape (N, H, W, 3).
                y (np.ndarray): Integer label array, shape (N,).
        """
        label_map = {cls: i for i, cls in enumerate(self.class_names)}
        X, y = [], []
        for _, row in self.metadata.iterrows():
            arr = self.load_image(row["filepath"], size=size)
            X.append(arr)
            y.append(label_map[row["class"]])
        return np.array(X, dtype=np.uint8), np.array(y, dtype=np.int32)

    # ------------------------------------------------------------------
    # Queries / stats
    # ------------------------------------------------------------------

    def class_counts(self) -> pd.Series:
        """Return count of images per class."""
        return self.metadata["class"].value_counts()

    def summary(self) -> dict:
        """
        Return a high-level summary dictionary of the dataset.

        Returns:
            dict: Keys include total_images, num_classes, class_names,
                  avg_width, avg_height.
        """
        return {
            "total_images": len(self.metadata),
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
            "avg_width": self.metadata["width"].mean(),
            "avg_height": self.metadata["height"].mean(),
            "avg_size_bytes": self.metadata["size_bytes"].mean(),
        }

    def get_sample(self, class_name: str | None = None, n: int = 9) -> pd.DataFrame:
        """
        Return a random sample of metadata rows.

        Args:
            class_name (str, optional): Filter to one class.
            n (int): Number of samples.

        Returns:
            pd.DataFrame: Sampled metadata rows.
        """
        df = self.metadata
        if class_name is not None:
            df = df[df["class"] == class_name]
        return df.sample(min(n, len(df)), random_state=42).reset_index(drop=True)

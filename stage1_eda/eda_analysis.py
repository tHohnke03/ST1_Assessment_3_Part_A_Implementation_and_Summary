"""
*******************************
Author: u3243935
Assessment 3 - Macroinvertebrate Image Analysis System
Software Technology 1 (4483/8995)
Module: stage1_eda/eda_analysis.py
Description: EDAAnalyser class — performs Exploratory Data Analysis on the
             macroinvertebrate dataset including class distribution, image
             statistics, colour analysis, and visualisations using Pandas,
             NumPy, Matplotlib, Seaborn, and OpenCV.
Date: 2024-05-12
*******************************
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend for file saving
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image

from utils.dataset_loader import DatasetLoader
from utils.image_processor import ImageProcessor


class EDAAnalyser:
    """
    Performs Exploratory Data Analysis on the macroinvertebrate dataset.

    Generates:
      - Class distribution bar chart
      - Sample image grid per class
      - Per-channel pixel intensity distributions
      - Average colour per class heat-map
      - Correlation of image statistics
      - Edge detection examples

    All plots are saved to `output_dir`.
    """

    # Seaborn palette used throughout
    PALETTE = "Set2"

    def __init__(self, loader: DatasetLoader, output_dir: str):
        """
        Initialise EDAAnalyser.

        Args:
            loader (DatasetLoader): Initialised dataset loader.
            output_dir (str): Directory where plot images are saved.
        """
        self.loader = loader
        self.output_dir = output_dir
        self.processor = ImageProcessor()
        os.makedirs(output_dir, exist_ok=True)

        # Computed lazily
        self._pixel_stats: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_full_eda(self) -> list[str]:
        """
        Run all EDA routines and return a list of saved file paths.

        Returns:
            list[str]: Paths to all generated plot files.
        """
        saved = []
        saved.append(self.plot_class_distribution())
        saved.append(self.plot_sample_grid())
        saved.append(self.plot_pixel_intensity_distributions())
        saved.append(self.plot_average_colour_per_class())
        saved.append(self.plot_image_size_stats())
        saved.append(self.plot_edge_examples())
        return saved

    # ------------------------------------------------------------------
    # Individual plots
    # ------------------------------------------------------------------

    def plot_class_distribution(self) -> str:
        """
        Bar chart of image counts per class.

        Returns:
            str: Path to saved figure.
        """
        counts = self.loader.class_counts().sort_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(
            counts.index,
            counts.values,
            color=sns.color_palette(self.PALETTE, len(counts)),
            edgecolor="black",
            linewidth=0.7,
        )
        ax.bar_label(bars, padding=3, fontsize=10)
        ax.set_title("Class Distribution — Macroinvertebrate Dataset", fontsize=14, fontweight="bold")
        ax.set_xlabel("Order", fontsize=12)
        ax.set_ylabel("Image Count", fontsize=12)
        ax.set_ylim(0, counts.max() * 1.2)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        path = os.path.join(self.output_dir, "01_class_distribution.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_sample_grid(self, n_cols: int = 5) -> str:
        """
        Display a grid of sample images — one row per class.

        Args:
            n_cols (int): Number of sample images per class.

        Returns:
            str: Path to saved figure.
        """
        classes = self.loader.class_names
        fig, axes = plt.subplots(
            len(classes), n_cols, figsize=(n_cols * 2.2, len(classes) * 2.2)
        )
        fig.suptitle("Sample Images by Class", fontsize=14, fontweight="bold", y=1.01)

        for row_idx, cls in enumerate(classes):
            samples = self.loader.get_sample(class_name=cls, n=n_cols)
            for col_idx in range(n_cols):
                ax = axes[row_idx][col_idx]
                if col_idx < len(samples):
                    fpath = samples.iloc[col_idx]["filepath"]
                    img = self.loader.load_image(fpath, size=(96, 96))
                    ax.imshow(img)
                ax.axis("off")
                if col_idx == 0:
                    ax.set_ylabel(cls, rotation=0, labelpad=60, fontsize=9, va="center")

        plt.tight_layout()
        path = os.path.join(self.output_dir, "02_sample_grid.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_pixel_intensity_distributions(self) -> str:
        """
        Plot per-channel (R, G, B) mean pixel intensity distributions
        across all images, split by class.

        Returns:
            str: Path to saved figure.
        """
        records = self._compute_pixel_stats()
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        channel_names = ["Red", "Green", "Blue"]
        colours = ["#e74c3c", "#2ecc71", "#3498db"]

        for idx, (ch_col, title, colour) in enumerate(
            zip(["mean_r", "mean_g", "mean_b"], channel_names, colours)
        ):
            ax = axes[idx]
            for cls in self.loader.class_names:
                vals = records[records["class"] == cls][ch_col]
                ax.hist(
                    vals,
                    bins=20,
                    alpha=0.55,
                    label=cls,
                    edgecolor="none",
                )
            ax.set_title(f"{title} Channel Intensity", fontsize=12)
            ax.set_xlabel("Mean Pixel Value (0–255)")
            ax.set_ylabel("Frequency")
            ax.legend(fontsize=8)

        fig.suptitle("Per-Channel Pixel Intensity Distributions by Class", fontsize=13, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(self.output_dir, "03_pixel_intensity.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_average_colour_per_class(self) -> str:
        """
        Compute the mean RGB colour for each class and visualise as
        a colour swatch grid and a grouped bar chart.

        Returns:
            str: Path to saved figure.
        """
        class_mean_rgb = {}
        for cls in self.loader.class_names:
            sample = self.loader.get_sample(class_name=cls, n=20)
            pixels = []
            for _, row in sample.iterrows():
                img = self.loader.load_image(row["filepath"], size=(64, 64))
                pixels.append(img.reshape(-1, 3))
            all_pixels = np.vstack(pixels).astype(np.float32)
            class_mean_rgb[cls] = all_pixels.mean(axis=0)

        classes = list(class_mean_rgb.keys())
        means = np.array([class_mean_rgb[c] for c in classes])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # Colour swatches
        for i, cls in enumerate(classes):
            colour_patch = means[i] / 255.0
            ax1.add_patch(plt.Rectangle((i, 0), 0.9, 0.9, color=colour_patch))
            ax1.text(i + 0.45, -0.1, cls, ha="center", va="top", fontsize=8, rotation=20)
        ax1.set_xlim(0, len(classes))
        ax1.set_ylim(-0.3, 1.0)
        ax1.set_title("Mean Colour Swatch per Class", fontsize=12)
        ax1.axis("off")

        # Grouped bar chart
        x = np.arange(len(classes))
        width = 0.25
        ax2.bar(x - width, means[:, 0], width, label="Red",   color="#e74c3c", alpha=0.8)
        ax2.bar(x,          means[:, 1], width, label="Green", color="#2ecc71", alpha=0.8)
        ax2.bar(x + width, means[:, 2], width, label="Blue",   color="#3498db", alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(classes, rotation=20, ha="right", fontsize=9)
        ax2.set_title("Mean RGB Channel Values per Class", fontsize=12)
        ax2.set_ylabel("Mean Pixel Value (0–255)")
        ax2.legend()

        fig.suptitle("Average Colour Analysis", fontsize=13, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(self.output_dir, "04_avg_colour_per_class.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_image_size_stats(self) -> str:
        """
        Box-plot of image file sizes and a correlation heat-map of
        numeric metadata columns.

        Returns:
            str: Path to saved figure.
        """
        df = self.loader.metadata.copy()
        df["size_kb"] = df["size_bytes"] / 1024

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # Box plot of file size by class
        classes = self.loader.class_names
        data_by_class = [df[df["class"] == c]["size_kb"].values for c in classes]
        bp = ax1.boxplot(data_by_class, labels=classes, patch_artist=True)
        colours = sns.color_palette(self.PALETTE, len(classes))
        for patch, colour in zip(bp["boxes"], colours):
            patch.set_facecolor(colour)
        ax1.set_title("Image File Size Distribution by Class", fontsize=12)
        ax1.set_xlabel("Class")
        ax1.set_ylabel("File Size (KB)")
        ax1.tick_params(axis="x", rotation=20)

        # Correlation heat-map
        numeric_cols = ["width", "height", "size_bytes"]
        stats = self._compute_pixel_stats()
        merged = df.merge(stats[["filepath", "mean_r", "mean_g", "mean_b", "std_r"]], on="filepath", how="left")
        corr_cols = ["width", "height", "size_bytes", "mean_r", "mean_g", "mean_b", "std_r"]
        corr = merged[corr_cols].corr()
        sns.heatmap(
            corr,
            ax=ax2,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )
        ax2.set_title("Feature Correlation Heat-map", fontsize=12)
        plt.setp(ax2.get_xticklabels(), rotation=30, ha="right", fontsize=8)

        fig.suptitle("Image Statistics", fontsize=13, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(self.output_dir, "05_image_stats.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_edge_examples(self) -> str:
        """
        Show original vs. Canny edge-detected versions for one sample
        from each class.

        Returns:
            str: Path to saved figure.
        """
        classes = self.loader.class_names
        fig, axes = plt.subplots(len(classes), 2, figsize=(6, len(classes) * 2.5))
        fig.suptitle("Edge Detection Examples (Canny)", fontsize=13, fontweight="bold")

        for row_idx, cls in enumerate(classes):
            sample = self.loader.get_sample(class_name=cls, n=1)
            fpath = sample.iloc[0]["filepath"]
            img = self.loader.load_image(fpath, size=(128, 128))
            edges = self.processor.detect_edges(img)

            axes[row_idx][0].imshow(img)
            axes[row_idx][0].set_title(f"{cls} (original)", fontsize=8)
            axes[row_idx][0].axis("off")

            axes[row_idx][1].imshow(edges, cmap="gray")
            axes[row_idx][1].set_title(f"{cls} (edges)", fontsize=8)
            axes[row_idx][1].axis("off")

        plt.tight_layout()
        path = os.path.join(self.output_dir, "06_edge_examples.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_pixel_stats(self) -> pd.DataFrame:
        """
        Compute per-image mean and std for each RGB channel.
        Result is cached after first call.

        Returns:
            pd.DataFrame: Rows matching metadata, extra columns for stats.
        """
        if self._pixel_stats is not None:
            return self._pixel_stats

        records = []
        for _, row in self.loader.metadata.iterrows():
            img = self.loader.load_image(row["filepath"], size=(64, 64)).astype(np.float32)
            records.append(
                {
                    "filepath": row["filepath"],
                    "class": row["class"],
                    "mean_r": img[:, :, 0].mean(),
                    "mean_g": img[:, :, 1].mean(),
                    "mean_b": img[:, :, 2].mean(),
                    "std_r": img[:, :, 0].std(),
                    "std_g": img[:, :, 1].std(),
                    "std_b": img[:, :, 2].std(),
                }
            )
        self._pixel_stats = pd.DataFrame(records)
        return self._pixel_stats

    def get_summary_text(self) -> str:
        """
        Return a human-readable text summary of the dataset.

        Returns:
            str: Multi-line summary string.
        """
        s = self.loader.summary()
        lines = [
            "=== Dataset Summary ===",
            f"Total images  : {s['total_images']}",
            f"Classes       : {s['num_classes']} — {', '.join(s['class_names'])}",
            f"Avg width     : {s['avg_width']:.1f} px",
            f"Avg height    : {s['avg_height']:.1f} px",
            f"Avg file size : {s['avg_size_bytes'] / 1024:.1f} KB",
            "",
            "Class counts:",
        ]
        for cls, count in self.loader.class_counts().items():
            lines.append(f"  {cls:<20} {count} images")
        return "\n".join(lines)

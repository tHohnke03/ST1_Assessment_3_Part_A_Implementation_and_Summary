"""
*******************************
Author: u3243935
Assessment 3 - Macroinvertebrate Image Analysis System
Software Technology 1 (4483/8995)
Module: stage3_gui/app.py
Description: MacroinvertebrateApp — Tkinter desktop GUI integrating all
             three stages (EDA, Classification, Prediction) into a
             tabbed application with live progress feedback.
Date: 2024-05-12
*******************************
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Project imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from utils.dataset_loader import DatasetLoader
from utils.image_processor import ImageProcessor
from stage1_eda.eda_analysis import EDAAnalyser
from stage2_classification.classifier import MacroinvertebrateClassifier


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
DARK_BG  = "#1e1e2e"
PANEL_BG = "#2a2a3e"
ACCENT   = "#7c3aed"
ACCENT2  = "#10b981"
TEXT_FG  = "#e2e8f0"
MUTED    = "#94a3b8"
CARD_BG  = "#313149"


class MacroinvertebrateApp:
    """
    Main Tkinter application for the Macroinvertebrate Image Analysis System.

    Tabs:
      1. Dashboard  — dataset overview and quick stats
      2. EDA        — exploratory analysis plots
      3. Classifier — train a model and view results
      4. Predict    — upload an image and get a prediction
      5. About      — project information
    """

    APP_TITLE   = "Macroinvertebrate Image Analysis System"
    WINDOW_SIZE = "1100x720"
    DATA_DIR    = os.path.join(ROOT_DIR, "data")
    OUTPUT_EDA  = os.path.join(ROOT_DIR, "outputs", "eda")
    OUTPUT_MDL  = os.path.join(ROOT_DIR, "outputs", "models")

    def __init__(self):
        """Initialise the application window and core components."""
        self.root = tk.Tk()
        self.root.title(self.APP_TITLE)
        self.root.geometry(self.WINDOW_SIZE)
        self.root.configure(bg=DARK_BG)
        self.root.resizable(True, True)

        self.loader:     DatasetLoader | None = None
        self.eda:        EDAAnalyser   | None = None
        self.classifier: MacroinvertebrateClassifier | None = None
        self.processor   = ImageProcessor()

        self._eda_paths: list[str] = []
        self._eda_idx:   int = 0

        self._build_ui()
        self._load_dataset_async()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self):
        """Start the Tkinter event loop."""
        self.root.mainloop()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        self._build_header()
        self._build_notebook()
        self._build_status_bar()

    def _build_header(self):
        header = tk.Frame(self.root, bg=ACCENT, height=56)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(
            header,
            text="🦟  Macroinvertebrate Image Analysis System",
            font=("Helvetica", 16, "bold"),
            bg=ACCENT, fg="white",
        ).pack(side="left", padx=20, pady=10)
        tk.Label(
            header,
            text="Software Technology 1 — Assessment 3",
            font=("Helvetica", 10),
            bg=ACCENT, fg="#ddd6fe",
        ).pack(side="right", padx=20)

    def _build_notebook(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Custom.TNotebook", background=DARK_BG, tabmargins=[4, 4, 0, 0])
        style.configure(
            "Custom.TNotebook.Tab",
            background=PANEL_BG, foreground=TEXT_FG,
            padding=[14, 8], font=("Helvetica", 10, "bold"),
        )
        style.map(
            "Custom.TNotebook.Tab",
            background=[("selected", ACCENT)],
            foreground=[("selected", "white")],
        )
        self.notebook = ttk.Notebook(self.root, style="Custom.TNotebook")
        self.notebook.pack(fill="both", expand=True, padx=8, pady=6)
        self._build_dashboard_tab()
        self._build_eda_tab()
        self._build_classifier_tab()
        self._build_predict_tab()
        self._build_about_tab()

    def _build_status_bar(self):
        self.status_var = tk.StringVar(value="Initialising…")
        bar = tk.Frame(self.root, bg=PANEL_BG, height=28)
        bar.pack(fill="x", side="bottom")
        tk.Label(bar, textvariable=self.status_var, bg=PANEL_BG, fg=MUTED,
                 font=("Helvetica", 9), anchor="w").pack(side="left", padx=12)

    # ------------------------------------------------------------------
    # Tab: Dashboard
    # ------------------------------------------------------------------

    def _build_dashboard_tab(self):
        frame = tk.Frame(self.notebook, bg=DARK_BG)
        self.notebook.add(frame, text="📊  Dashboard")
        tk.Label(frame, text="Dataset Overview", font=("Helvetica", 15, "bold"),
                 bg=DARK_BG, fg=TEXT_FG).pack(pady=(20, 6))
        self.dash_cards_frame = tk.Frame(frame, bg=DARK_BG)
        self.dash_cards_frame.pack(pady=10)
        self.dash_text = tk.Text(frame, bg=CARD_BG, fg=TEXT_FG, font=("Courier", 11),
                                 height=18, relief="flat", padx=14, pady=10)
        self.dash_text.pack(fill="both", expand=True, padx=20, pady=(0, 16))
        self.dash_text.insert("end", "Loading dataset…")
        self.dash_text.config(state="disabled")

    def _update_dashboard(self):
        if self.loader is None:
            return
        summary = self.loader.summary()
        for w in self.dash_cards_frame.winfo_children():
            w.destroy()
        for label, value, icon in [
            ("Total Images", summary["total_images"], "🖼"),
            ("Species",      summary["num_classes"],  "🔖"),
            ("Avg Width",    f"{summary['avg_width']:.0f} px",  "↔"),
            ("Avg Height",   f"{summary['avg_height']:.0f} px", "↕"),
        ]:
            card = tk.Frame(self.dash_cards_frame, bg=CARD_BG, width=170, height=90)
            card.pack(side="left", padx=10)
            card.pack_propagate(False)
            tk.Label(card, text=icon,        font=("Helvetica", 20), bg=CARD_BG, fg=ACCENT).pack(pady=(10, 2))
            tk.Label(card, text=str(value),  font=("Helvetica", 16, "bold"), bg=CARD_BG, fg=TEXT_FG).pack()
            tk.Label(card, text=label,       font=("Helvetica", 9), bg=CARD_BG, fg=MUTED).pack()

        text = EDAAnalyser(self.loader, self.OUTPUT_EDA).get_summary_text()
        self.dash_text.config(state="normal")
        self.dash_text.delete("1.0", "end")
        self.dash_text.insert("end", text)
        self.dash_text.config(state="disabled")

    # ------------------------------------------------------------------
    # Tab: EDA
    # ------------------------------------------------------------------

    def _build_eda_tab(self):
        frame = tk.Frame(self.notebook, bg=DARK_BG)
        self.notebook.add(frame, text="🔍  EDA")
        ctrl = tk.Frame(frame, bg=DARK_BG)
        ctrl.pack(fill="x", padx=16, pady=12)
        self._style_button(ctrl, "Run Full EDA", self._run_eda_async, ACCENT2).pack(side="left", padx=6)
        self._style_button(ctrl, "◀ Prev",       self._eda_prev,      PANEL_BG).pack(side="left", padx=4)
        self._style_button(ctrl, "Next ▶",       self._eda_next,      PANEL_BG).pack(side="left", padx=4)
        self.eda_label_var = tk.StringVar(value="No plots yet — click Run Full EDA")
        tk.Label(ctrl, textvariable=self.eda_label_var, bg=DARK_BG, fg=MUTED,
                 font=("Helvetica", 10)).pack(side="left", padx=12)
        self.eda_canvas_frame = tk.Frame(frame, bg=DARK_BG)
        self.eda_canvas_frame.pack(fill="both", expand=True, padx=16, pady=(0, 12))
        self.eda_img_label = tk.Label(self.eda_canvas_frame, bg=DARK_BG, text="")
        self.eda_img_label.pack(fill="both", expand=True)

    def _run_eda_async(self):
        self._set_status("Running EDA — please wait…")
        threading.Thread(target=self._run_eda, daemon=True).start()

    def _run_eda(self):
        if self.loader is None:
            messagebox.showerror("Error", "Dataset not loaded yet.")
            return
        self.eda = EDAAnalyser(self.loader, self.OUTPUT_EDA)
        paths = self.eda.run_full_eda()
        self._eda_paths = paths
        self._eda_idx = 0
        self.root.after(0, self._show_eda_plot)
        self.root.after(0, lambda: self._set_status(f"EDA complete — {len(paths)} plots generated."))

    def _show_eda_plot(self):
        if not self._eda_paths:
            return
        path = self._eda_paths[self._eda_idx]
        self.eda_label_var.set(f"Plot {self._eda_idx + 1}/{len(self._eda_paths)}: {os.path.basename(path)}")
        self._display_image_in_label(self.eda_img_label, path, max_size=(950, 560))

    def _eda_prev(self):
        if self._eda_paths:
            self._eda_idx = (self._eda_idx - 1) % len(self._eda_paths)
            self._show_eda_plot()

    def _eda_next(self):
        if self._eda_paths:
            self._eda_idx = (self._eda_idx + 1) % len(self._eda_paths)
            self._show_eda_plot()

    # ------------------------------------------------------------------
    # Tab: Classifier
    # ------------------------------------------------------------------

    def _build_classifier_tab(self):
        frame = tk.Frame(self.notebook, bg=DARK_BG)
        self.notebook.add(frame, text="🤖  Classifier")
        ctrl = tk.Frame(frame, bg=DARK_BG)
        ctrl.pack(fill="x", padx=16, pady=12)
        tk.Label(ctrl, text="Model:", bg=DARK_BG, fg=TEXT_FG,
                 font=("Helvetica", 10)).pack(side="left", padx=(0, 6))
        self.model_var = tk.StringVar(value="random_forest")
        ttk.Combobox(
            ctrl, textvariable=self.model_var,
            values=list(MacroinvertebrateClassifier.AVAILABLE_MODELS.keys()),
            state="readonly", width=18,
        ).pack(side="left", padx=6)
        self._style_button(ctrl, "Train Model", self._train_async, ACCENT2).pack(side="left", padx=10)
        self.clf_text = tk.Text(frame, bg=CARD_BG, fg=TEXT_FG, font=("Courier", 10),
                                height=10, relief="flat", padx=12, pady=8)
        self.clf_text.pack(fill="x", padx=16, pady=(0, 8))
        self.clf_plots_frame = tk.Frame(frame, bg=DARK_BG)
        self.clf_plots_frame.pack(fill="both", expand=True, padx=16, pady=(0, 12))
        self.clf_cm_label  = tk.Label(self.clf_plots_frame, bg=DARK_BG)
        self.clf_cm_label.pack(side="left", expand=True, fill="both")
        self.clf_rep_label = tk.Label(self.clf_plots_frame, bg=DARK_BG)
        self.clf_rep_label.pack(side="left", expand=True, fill="both")

    def _train_async(self):
        self._set_status("Training model — this may take a moment…")
        threading.Thread(target=self._train, daemon=True).start()

    def _train(self):
        if self.loader is None:
            messagebox.showerror("Error", "Dataset not loaded yet.")
            return
        self.classifier = MacroinvertebrateClassifier(
            self.loader, self.OUTPUT_MDL, model_name=self.model_var.get()
        )
        self.classifier.train()
        self.classifier.save_model()
        cm_path  = self.classifier.plot_confusion_matrix()
        rep_path = self.classifier.plot_class_report()
        self.classifier.plot_feature_importance()

        def update_ui():
            self.clf_text.config(state="normal")
            self.clf_text.delete("1.0", "end")
            self.clf_text.insert("end", self.classifier.get_results_text())
            self.clf_text.config(state="disabled")
            self._display_image_in_label(self.clf_cm_label,  cm_path,  max_size=(480, 340))
            self._display_image_in_label(self.clf_rep_label, rep_path, max_size=(480, 340))
            self._set_status("Training complete.")

        self.root.after(0, update_ui)

    # ------------------------------------------------------------------
    # Tab: Predict
    # ------------------------------------------------------------------

    def _build_predict_tab(self):
        frame = tk.Frame(self.notebook, bg=DARK_BG)
        self.notebook.add(frame, text="🔮  Predict")

        left = tk.Frame(frame, bg=PANEL_BG, width=380)
        left.pack(side="left", fill="both", padx=(16, 8), pady=16)
        left.pack_propagate(False)
        tk.Label(left, text="Upload Image", font=("Helvetica", 13, "bold"),
                 bg=PANEL_BG, fg=TEXT_FG).pack(pady=(16, 8))
        self._style_button(left, "Browse…", self._browse_image, ACCENT).pack(pady=4)
        self.pred_img_label = tk.Label(left, bg=PANEL_BG, text="No image selected",
                                       fg=MUTED, font=("Helvetica", 10))
        self.pred_img_label.pack(expand=True, fill="both", padx=12, pady=12)

        right = tk.Frame(frame, bg=DARK_BG)
        right.pack(side="left", fill="both", expand=True, padx=(0, 16), pady=16)
        tk.Label(right, text="Prediction Results", font=("Helvetica", 13, "bold"),
                 bg=DARK_BG, fg=TEXT_FG).pack(pady=(16, 8))
        self.pred_class_var = tk.StringVar(value="—")
        tk.Label(right, textvariable=self.pred_class_var, font=("Helvetica", 18, "bold"),
                 bg=DARK_BG, fg=ACCENT).pack()
        self.pred_conf_var = tk.StringVar(value="")
        tk.Label(right, textvariable=self.pred_conf_var, font=("Helvetica", 12),
                 bg=DARK_BG, fg=MUTED).pack(pady=4)
        self.pred_chart_frame = tk.Frame(right, bg=DARK_BG)
        self.pred_chart_frame.pack(fill="both", expand=True, pady=8)
        self._style_button(right, "Predict", self._predict, ACCENT2).pack(pady=8)
        self._pred_image_path: str | None = None

    def _browse_image(self):
        path = filedialog.askopenfilename(
            title="Select Macroinvertebrate Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")],
        )
        if path:
            self._pred_image_path = path
            self._display_image_in_label(self.pred_img_label, path, max_size=(340, 300))
            self.pred_class_var.set("—")
            self.pred_conf_var.set("")

    def _predict(self):
        if self._pred_image_path is None:
            messagebox.showwarning("No Image", "Please select an image first.")
            return
        if self.classifier is None or self.classifier.pipeline is None:
            model_path = os.path.join(self.OUTPUT_MDL, "model.pkl")
            if not os.path.isfile(model_path):
                messagebox.showwarning("No Model", "Please train the classifier first (Classifier tab).")
                return
            self.classifier = MacroinvertebrateClassifier(self.loader, self.OUTPUT_MDL)
            self.classifier.load_model(model_path)

        img = np.array(Image.open(self._pred_image_path).convert("RGB"))
        predicted, confidence, all_probs = self.classifier.predict(img)
        self.pred_class_var.set(predicted)
        self.pred_conf_var.set(f"Confidence: {confidence:.1%}")
        self._draw_prob_bars(all_probs)
        self._set_status(f"Prediction: {predicted} ({confidence:.1%})")

    def _draw_prob_bars(self, probs: dict):
        """Draw a Matplotlib probability bar chart embedded in the predict tab."""
        for w in self.pred_chart_frame.winfo_children():
            w.destroy()
        fig, ax = plt.subplots(figsize=(5, 6), facecolor=DARK_BG)
        ax.set_facecolor(DARK_BG)
        classes = list(probs.keys())
        values  = [probs[c] for c in classes]
        best    = max(probs, key=probs.get)
        colours = [ACCENT if c == best else "#475569" for c in classes]
        bars = ax.barh(classes, values, color=colours, height=0.6)
        ax.bar_label(bars, fmt="%.1%%", padding=4, color=TEXT_FG, fontsize=7)
        ax.set_xlim(0, 1.2)
        ax.set_title("Class Probabilities", color=TEXT_FG, fontsize=10)
        ax.tick_params(colors=TEXT_FG, labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")
        plt.tight_layout(pad=1.0)
        canvas = FigureCanvasTkAgg(fig, master=self.pred_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Tab: About
    # ------------------------------------------------------------------

    def _build_about_tab(self):
        frame = tk.Frame(self.notebook, bg=DARK_BG)
        self.notebook.add(frame, text="ℹ️  About")
        about_text = """
Macroinvertebrate Image Analysis System
Software Technology 1 — Assessment 3 (4483/8995)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Author:  u3243935
Date:    2024-05-12

─── Overview ───────────────────────────────────────────

This application classifies freshwater macroinvertebrate
images across 17 species from the Kaggle Stream
Macroinvertebrates dataset:

  • Asellus sp          • Baetidae sp
  • Elmis sp            • Ephemerellidae
  • Erpobdella sp       • Gammarus sp
  • Hydropsychidae sp   • Leptophlebiidae sp
  • Leuctra sp          • Limnius sp
  • Lymnea sp           • Nemoura sp
  • Oligochaeta sp      • Sericostomatidae sp
  • Sialis sp           • Simuliidae sp
  • Sphaerium sp

─── Stages ──────────────────────────────────────────────

  Stage 1 — EDA
    Exploratory Data Analysis using Pandas, NumPy, OpenCV,
    Matplotlib, and Seaborn. Includes class distributions,
    pixel intensity plots, colour analysis, and edge detection.

  Stage 2 — Classification
    Feature extraction (HOG + colour histograms + texture) fed
    into a Scikit-learn pipeline. Supports Random Forest, SVM,
    KNN, and Gradient Boosting. Evaluates with confusion matrix,
    classification report, and 5-fold cross-validation.

  Stage 3 — GUI Deployment
    Tkinter desktop application providing a unified interface
    for all three stages.

─── Libraries ───────────────────────────────────────────

  pandas · numpy · opencv-python · scikit-learn
  matplotlib · seaborn · pillow · tkinter

─── Dataset ─────────────────────────────────────────────

  Kaggle Stream Macroinvertebrates
  https://www.kaggle.com/datasets/kennethtm/stream-macroinvertebrates
"""
        w = tk.Text(frame, bg=CARD_BG, fg=TEXT_FG, font=("Courier", 11),
                    relief="flat", padx=20, pady=16)
        w.pack(fill="both", expand=True, padx=20, pady=20)
        w.insert("end", about_text)
        w.config(state="disabled")

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def _load_dataset_async(self):
        threading.Thread(target=self._load_dataset, daemon=True).start()

    def _load_dataset(self):
        try:
            self.loader = DatasetLoader(self.DATA_DIR)
            self.root.after(0, self._update_dashboard)
            self.root.after(0, lambda: self._set_status(
                f"Dataset loaded — {self.loader.summary()['total_images']} images, "
                f"{self.loader.summary()['num_classes']} species."
            ))
        except Exception as e:
            self.root.after(0, lambda: self._set_status(f"Error loading dataset: {e}"))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_status(self, msg: str):
        self.status_var.set(msg)

    @staticmethod
    def _style_button(parent, text: str, command, bg_colour: str) -> tk.Button:
        return tk.Button(
            parent, text=text, command=command,
            bg=bg_colour, fg="white", font=("Helvetica", 10, "bold"),
            relief="flat", padx=14, pady=6, cursor="hand2",
            activebackground=bg_colour, activeforeground="white",
        )

    def _display_image_in_label(self, label: tk.Label, path: str, max_size: tuple[int, int]):
        """Load an image and display it in a tk.Label, scaled to fit."""
        try:
            img = Image.open(path)
            img.thumbnail(max_size, Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            label.config(image=photo, text="")
            label._photo = photo  # type: ignore[attr-defined]
        except Exception as e:
            label.config(text=f"Could not load image:\n{e}", image="")

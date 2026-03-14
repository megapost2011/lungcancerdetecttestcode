#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
肺癌量子診断システム - 起動安定化版
Tkinter GUI / Windowsダブルクリック対策版

主な修正:
- Qiskit依存を除去（現コードでは実処理に未使用のため）
- 起動時エラーをログ保存
- 必須ライブラリ不足をGUI表示
- scikit-image未導入時のフォールバック
- PCA次元数を自動調整
- OpenCVの日本語パス対策
- スレッドUI更新をroot.after経由に統一
"""

import sys
import os
import json
import traceback
import warnings
from datetime import datetime
import threading

warnings.filterwarnings("ignore")

# =========================
# Tkinter
# =========================
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

# =========================
# 依存関係チェック付き import
# =========================
MISSING = []
IMPORT_ERRORS = []

try:
    import numpy as np
except Exception as e:
    MISSING.append("numpy")
    IMPORT_ERRORS.append(f"numpy: {e}")

try:
    import cv2
except Exception as e:
    MISSING.append("opencv-python (cv2)")
    IMPORT_ERRORS.append(f"cv2: {e}")

try:
    from scipy.stats import skew, kurtosis
except Exception as e:
    MISSING.append("scipy")
    IMPORT_ERRORS.append(f"scipy: {e}")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
except Exception as e:
    MISSING.append("scikit-learn")
    IMPORT_ERRORS.append(f"scikit-learn: {e}")

SKIMAGE_AVAILABLE = True
try:
    from skimage import restoration
    from skimage.feature import graycomatrix, graycoprops
except Exception as e:
    SKIMAGE_AVAILABLE = False
    IMPORT_ERRORS.append(f"scikit-image: {e}")

PYDICOM_AVAILABLE = True
try:
    import pydicom
except Exception:
    PYDICOM_AVAILABLE = False

MATPLOTLIB_AVAILABLE = True
try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
except Exception as e:
    MATPLOTLIB_AVAILABLE = False
    IMPORT_ERRORS.append(f"matplotlib: {e}")


def get_app_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except Exception:
        return os.getcwd()


def write_error_log(err_text: str):
    log_path = os.path.join(get_app_dir(), "lung_gui_error.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(err_text)
    return log_path


def safe_imread_gray(path):
    """日本語パス対策付き画像読込"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        return img
    except Exception:
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


class QuantumImageReconstructor:
    """量子画像再構成エンジン（実装は安定動作用のクラシカル近似）"""

    def __init__(self, n_qubits=6):
        self.n_qubits = n_qubits

    def quantum_super_resolution(self, low_res_image, scale_factor=2):
        h, w = low_res_image.shape
        new_h, new_w = h * scale_factor, w * scale_factor

        initial_sr = cv2.resize(
            low_res_image,
            (new_w, new_h),
            interpolation=cv2.INTER_CUBIC
        )

        sobel_x = cv2.Sobel(initial_sr, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(initial_sr, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        enhanced = initial_sr.astype(np.float64) + 0.3 * edges
        enhanced = np.clip(enhanced, 0, 255)

        if SKIMAGE_AVAILABLE:
            try:
                restored = restoration.wiener(enhanced, np.ones((5, 5)) / 25)
                restored = (
                    (restored - restored.min())
                    / (restored.max() - restored.min() + 1e-8)
                    * 255
                )
                return restored.astype(np.uint8)
            except Exception:
                pass

        # フォールバック
        blurred = cv2.GaussianBlur(enhanced.astype(np.uint8), (3, 3), 0)
        return blurred.astype(np.uint8)


class QuantumUnsupervisedLearner:
    """量子教師無し学習エンジン（安定動作用のクラシカル版）"""

    def __init__(self, n_qubits=4, n_clusters=3):
        self.n_qubits = n_qubits
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()

        self.cluster_labels = {
            0: "正常組織",
            1: "良性結節",
            2: "悪性結節（疑い）"
        }

    def extract_nodule_features(self, image):
        preprocessed = self._preprocess_lung_ct(image)
        lung_mask = self._segment_lung(preprocessed)
        nodule_candidates = self._detect_nodule_candidates(preprocessed, lung_mask)

        all_features = []
        nodules = []

        for candidate in nodule_candidates:
            features = self._extract_single_nodule_features(preprocessed, candidate)
            all_features.append(features)
            nodules.append(candidate)

        if len(all_features) == 0:
            return np.array([]), []

        return np.array(all_features), nodules

    def _preprocess_lung_ct(self, image):
        image = image.astype(np.float32)

        # 8bit画像ならそのまま扱う
        if image.max() <= 255 and image.min() >= 0:
            normalized = image.astype(np.uint8)
        else:
            normalized = np.clip(image, -1000, 400)
            normalized = (normalized + 1000) / 1400 * 255
            normalized = normalized.astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(normalized)

        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        return denoised

    def _segment_lung(self, image):
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = np.ones((5, 5), np.uint8)
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed)

        if num_labels <= 1:
            return np.zeros_like(image)

        areas = stats[1:, cv2.CC_STAT_AREA]
        if len(areas) == 0:
            return np.zeros_like(image)

        top_indices = np.argsort(areas)[-2:] + 1

        lung_mask = np.zeros_like(labels, dtype=np.uint8)
        for idx in top_indices:
            lung_mask[labels == idx] = 255

        return lung_mask

    def _detect_nodule_candidates(self, image, lung_mask):
        lung_region = cv2.bitwise_and(image, image, mask=lung_mask)

        log_filtered = cv2.Laplacian(lung_region, cv2.CV_64F, ksize=5)
        log_filtered = np.abs(log_filtered)

        threshold = np.mean(log_filtered) + 2 * np.std(log_filtered)
        _, candidates = cv2.threshold(
            log_filtered.astype(np.uint8),
            int(max(0, min(255, threshold))),
            255,
            cv2.THRESH_BINARY
        )

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidates)

        nodule_candidates = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            if 10 < area < 1000:
                aspect_ratio = max(w, h) / (min(w, h) + 1e-8)
                if aspect_ratio < 3:
                    nodule_candidates.append({
                        "bbox": (x, y, w, h),
                        "centroid": centroids[i],
                        "area": int(area)
                    })

        return nodule_candidates

    def _extract_single_nodule_features(self, image, candidate):
        x, y, w, h = candidate["bbox"]

        padding = 5
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)

        roi = image[y1:y2, x1:x2]

        if roi.size == 0:
            return np.zeros(30, dtype=np.float32)

        features = []

        # 幾何学特徴
        features.append(float(candidate["area"]))
        features.append(float(w / (h + 1e-8)))
        features.append(float(4 * np.pi * candidate["area"] / ((w + h) ** 2 + 1e-8)))

        # 強度特徴
        flat = roi.flatten().astype(np.float32)
        features.extend([
            float(np.mean(roi)),
            float(np.std(roi)),
            float(np.median(roi)),
            float(np.max(roi) - np.min(roi)),
            float(skew(flat)) if flat.size > 2 else 0.0,
            float(kurtosis(flat)) if flat.size > 3 else 0.0
        ])

        # GLCM特徴
        if SKIMAGE_AVAILABLE:
            try:
                roi_norm = ((roi - roi.min()) / (roi.max() - roi.min() + 1e-8) * 255).astype(np.uint8)
                glcm = graycomatrix(roi_norm, [1], [0], levels=256, symmetric=True, normed=True)
                features.extend([
                    float(graycoprops(glcm, "contrast")[0, 0]),
                    float(graycoprops(glcm, "dissimilarity")[0, 0]),
                    float(graycoprops(glcm, "homogeneity")[0, 0]),
                    float(graycoprops(glcm, "energy")[0, 0]),
                    float(graycoprops(glcm, "correlation")[0, 0]),
                ])
            except Exception:
                features.extend([0.0] * 5)
        else:
            features.extend([0.0] * 5)

        # 形状特徴
        try:
            _, th = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                cnt = max(contours, key=cv2.contourArea)
                perimeter = cv2.arcLength(cnt, True)
                features.append(float(perimeter))

                moments = cv2.moments(cnt)
                hu = cv2.HuMoments(moments).flatten()
                features.extend([float(v) for v in hu[:5]])
            else:
                features.extend([0.0] * 6)
        except Exception:
            features.extend([0.0] * 6)

        # エッジ特徴
        edges = cv2.Canny(roi, 50, 150)
        features.append(float(np.sum(edges > 0) / (roi.size + 1e-8)))

        # 放射状特徴
        center_x, center_y = roi.shape[1] // 2, roi.shape[0] // 2
        radial_profile = []
        r = max(1, min(roi.shape[0], roi.shape[1]) // 2 - 1)
        for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            dx = int(center_x + r * np.cos(angle))
            dy = int(center_y + r * np.sin(angle))
            if 0 <= dx < roi.shape[1] and 0 <= dy < roi.shape[0]:
                radial_profile.append(float(roi[dy, dx]))

        features.append(float(np.std(radial_profile)) if radial_profile else 0.0)

        # 長さ固定
        while len(features) < 30:
            features.append(0.0)

        return np.array(features[:30], dtype=np.float32)

    def _reduce_features(self, features, target_dim=8):
        if len(features) == 0:
            return np.array([]), None

        features_scaled = self.scaler.fit_transform(features)

        max_comp = min(target_dim, features_scaled.shape[0], features_scaled.shape[1])
        if max_comp < 1:
            return features_scaled, None

        pca = PCA(n_components=max_comp)
        reduced = pca.fit_transform(features_scaled)
        return reduced, pca

    def quantum_clustering(self, features):
        if len(features) == 0:
            return np.array([]), np.array([])

        reduced, _ = self._reduce_features(features, target_dim=8)

        n_clusters = max(2, min(self.n_clusters, len(reduced)))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(reduced)
        centers = kmeans.cluster_centers_
        return labels, centers

    def quantum_anomaly_detection(self, features):
        if len(features) == 0:
            return np.array([]), np.array([])

        reduced, pca = self._reduce_features(features, target_dim=8)

        if pca is None:
            scores = np.zeros(len(features), dtype=np.float32)
            flags = np.zeros(len(features), dtype=bool)
            return scores, flags

        features_scaled = self.scaler.fit_transform(features)
        reconstructed = pca.inverse_transform(reduced)
        errors = np.sum((features_scaled[:, :reconstructed.shape[1]] - reconstructed) ** 2, axis=1)

        threshold = np.mean(errors) + 2 * np.std(errors)
        is_anomaly = errors > threshold
        anomaly_scores = (errors - errors.min()) / (errors.max() - errors.min() + 1e-8)

        return anomaly_scores, is_anomaly


class LungCancerQuantumGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("肺癌量子診断システム - Tkinter版")
        self.root.geometry("1400x900")

        self.current_image = None
        self.current_result = None
        self.processing = False

        self.reconstructor = QuantumImageReconstructor(n_qubits=6)
        self.learner = QuantumUnsupervisedLearner(n_qubits=4, n_clusters=3)

        self.setup_ui()

    def setup_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        title_frame = ttk.Frame(main_frame)
        title_frame.grid(row=0, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))

        title_label = ttk.Label(
            title_frame,
            text="肺癌量子診断システム - Quantum Unsupervised Learning",
            font=("Arial", 18, "bold"),
            foreground="#2c3e50"
        )
        title_label.pack()

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        self.diagnosis_tab = ttk.Frame(self.notebook)
        self.settings_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.diagnosis_tab, text="診断")
        self.notebook.add(self.settings_tab, text="設定")

        self.setup_diagnosis_tab()
        self.setup_settings_tab()

        self.status_bar = ttk.Label(main_frame, text="準備完了", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

    def setup_diagnosis_tab(self):
        control_frame = ttk.LabelFrame(self.diagnosis_tab, text="コントロール", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=5)

        ttk.Button(control_frame, text="画像を開く", command=self.load_image).grid(row=0, column=0, padx=5)

        ttk.Label(control_frame, text="解析モード:").grid(row=0, column=1, padx=5)

        self.mode_var = tk.StringVar(value="clustering")
        mode_combo = ttk.Combobox(
            control_frame,
            textvariable=self.mode_var,
            values=["clustering", "anomaly"],
            state="readonly",
            width=15
        )
        mode_combo.grid(row=0, column=2, padx=5)

        self.process_btn = ttk.Button(
            control_frame,
            text="解析実行",
            command=self.run_analysis,
            state=tk.DISABLED
        )
        self.process_btn.grid(row=0, column=3, padx=5)

        ttk.Button(control_frame, text="結果保存", command=self.save_results).grid(row=0, column=4, padx=5)

        self.progress = ttk.Progressbar(self.diagnosis_tab, mode="determinate", length=400)
        self.progress.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=5)

        self.status_label = ttk.Label(self.diagnosis_tab, text="処理待機中...", foreground="#7f8c8d")
        self.status_label.grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5)

        paned = ttk.PanedWindow(self.diagnosis_tab, orient=tk.HORIZONTAL)
        paned.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)

        ttk.Label(left_frame, text="入力画像:", font=("Arial", 10, "bold")).pack(pady=5)

        self.original_fig = Figure(figsize=(6, 5), dpi=80)
        self.original_ax = self.original_fig.add_subplot(111)
        self.original_canvas = FigureCanvasTkAgg(self.original_fig, left_frame)
        self.original_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        ttk.Label(left_frame, text="処理結果:", font=("Arial", 10, "bold")).pack(pady=5)

        self.result_fig = Figure(figsize=(6, 5), dpi=80)
        self.result_ax = self.result_fig.add_subplot(111)
        self.result_canvas = FigureCanvasTkAgg(self.result_fig, left_frame)
        self.result_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=1)

        stats_frame = ttk.LabelFrame(right_frame, text="検出統計", padding="10")
        stats_frame.pack(fill=tk.BOTH, padx=5, pady=5)

        self.stats_text = scrolledtext.ScrolledText(stats_frame, height=8, width=50, font=("Courier", 9))
        self.stats_text.pack(fill=tk.BOTH, expand=True)

        ttk.Label(right_frame, text="クラスタ分布:", font=("Arial", 10, "bold")).pack(pady=5)

        self.cluster_fig = Figure(figsize=(6, 4), dpi=80)
        self.cluster_ax = self.cluster_fig.add_subplot(111)
        self.cluster_canvas = FigureCanvasTkAgg(self.cluster_fig, right_frame)
        self.cluster_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        nodule_frame = ttk.LabelFrame(right_frame, text="検出結節", padding="10")
        nodule_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.nodule_listbox = tk.Listbox(nodule_frame, height=10, font=("Courier", 9))
        scrollbar = ttk.Scrollbar(nodule_frame, orient=tk.VERTICAL, command=self.nodule_listbox.yview)
        self.nodule_listbox.configure(yscrollcommand=scrollbar.set)

        self.nodule_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.diagnosis_tab.columnconfigure(0, weight=1)
        self.diagnosis_tab.rowconfigure(3, weight=1)

    def setup_settings_tab(self):
        quantum_frame = ttk.LabelFrame(self.settings_tab, text="量子パラメータ", padding="10")
        quantum_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=10, pady=10)

        ttk.Label(quantum_frame, text="量子ビット数:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.qubits_var = tk.IntVar(value=6)
        ttk.Spinbox(quantum_frame, from_=3, to=8, textvariable=self.qubits_var, width=10).grid(row=0, column=1, pady=5)

        cluster_frame = ttk.LabelFrame(self.settings_tab, text="クラスタリング設定", padding="10")
        cluster_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=10, pady=10)

        ttk.Label(cluster_frame, text="クラスタ数:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.clusters_var = tk.IntVar(value=3)
        ttk.Spinbox(cluster_frame, from_=2, to=5, textvariable=self.clusters_var, width=10).grid(row=0, column=1, pady=5)

        info_frame = ttk.LabelFrame(self.settings_tab, text="システム情報", padding="10")
        info_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)

        info_text = scrolledtext.ScrolledText(info_frame, height=15, width=60)
        info_text.pack(fill=tk.BOTH, expand=True)

        missing_text = ", ".join(MISSING) if MISSING else "なし"

        system_info = f"""
バージョン: 1.1.0 (起動安定化版)
Python: {sys.version.split()[0]}
UI: Tkinter

必須ライブラリ不足:
{missing_text}

scikit-image:
{"利用可能" if SKIMAGE_AVAILABLE else "未導入（フォールバック動作）"}

pydicom:
{"利用可能" if PYDICOM_AVAILABLE else "未導入（DICOMは開けません）"}

作成者:unknown
"""
        info_text.insert("1.0", system_info)
        info_text.config(state=tk.DISABLED)

        self.settings_tab.columnconfigure(0, weight=1)
        self.settings_tab.rowconfigure(2, weight=1)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="CT画像を選択",
            filetypes=[
                ("画像ファイル", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.dcm *.npy"),
                ("すべてのファイル", "*.*")
            ]
        )

        if not file_path:
            return

        try:
            ext = os.path.splitext(file_path)[1].lower()

            if ext == ".dcm":
                if not PYDICOM_AVAILABLE:
                    raise RuntimeError("pydicom が未インストールです。DICOMは読み込めません。")
                ds = pydicom.dcmread(file_path)
                self.current_image = ds.pixel_array.astype(np.float32)

            elif ext == ".npy":
                self.current_image = np.load(file_path)

            else:
                self.current_image = safe_imread_gray(file_path)

            if self.current_image is None:
                raise ValueError("画像読み込み失敗")

            if self.current_image.ndim != 2:
                raise ValueError("グレースケール画像ではありません")

            self.plot_image(self.original_ax, self.current_image, "入力CT画像")
            self.original_canvas.draw()

            self.process_btn.config(state=tk.NORMAL)
            self.status_bar.config(text=f"読み込み完了: {os.path.basename(file_path)}")

        except Exception as e:
            messagebox.showerror("エラー", f"画像読み込み失敗:\n{e}")

    def run_analysis(self):
        if self.current_image is None or self.processing:
            return

        self.processing = True
        self.process_btn.config(state=tk.DISABLED)
        self.progress["value"] = 0

        self.learner.n_clusters = self.clusters_var.get()
        self.reconstructor.n_qubits = self.qubits_var.get()

        thread = threading.Thread(target=self.process_image, daemon=True)
        thread.start()

    def process_image(self):
        try:
            mode = self.mode_var.get()

            self.update_status(10, "画像再構成開始...")

            if self.current_image.shape[0] < 128:
                enhanced = self.reconstructor.quantum_super_resolution(self.current_image, scale_factor=2)
                self.update_status(30, "量子超解像完了")
            else:
                enhanced = self.current_image.copy()

            self.update_status(40, "特徴抽出中...")

            features, nodules = self.learner.extract_nodule_features(enhanced)

            if len(features) == 0:
                self.root.after(0, lambda: messagebox.showinfo("情報", "結節が検出されませんでした"))
                self.root.after(0, self._finish_processing)
                return

            self.update_status(60, f"{len(nodules)}個の結節候補を検出")

            if mode == "clustering":
                self.update_status(70, "量子クラスタリング実行中...")
                labels, centers = self.learner.quantum_clustering(features)
                result = {
                    "enhanced": enhanced,
                    "nodules": nodules,
                    "labels": labels,
                    "centers": centers,
                    "features": features,
                    "mode": "clustering"
                }
            else:
                self.update_status(70, "量子異常検知実行中...")
                anomaly_scores, is_anomaly = self.learner.quantum_anomaly_detection(features)
                result = {
                    "enhanced": enhanced,
                    "nodules": nodules,
                    "anomaly_scores": anomaly_scores,
                    "is_anomaly": is_anomaly,
                    "features": features,
                    "mode": "anomaly"
                }

            self.update_status(90, "処理完了")
            self.root.after(0, lambda: self.display_result(result))

        except Exception as e:
            err = traceback.format_exc()
            log_path = write_error_log(err)
            self.root.after(0, lambda: messagebox.showerror(
                "エラー",
                f"処理中にエラー:\n{e}\n\nログ:\n{log_path}"
            ))
        finally:
            self.root.after(0, self._finish_processing)

    def _finish_processing(self):
        self.processing = False
        self.process_btn.config(state=tk.NORMAL)

    def update_status(self, progress, message):
        self.root.after(0, lambda: self.progress.config(value=progress))
        self.root.after(0, lambda: self.status_label.config(text=message))

    def display_result(self, result):
        self.current_result = result

        if "labels" in result:
            self.plot_nodules(
                self.result_ax,
                result["enhanced"],
                result["nodules"],
                result["labels"],
                "クラスタリング結果"
            )

            stats_text = f"検出結節数: {len(result['nodules'])}\n\nクラスタ分布:\n"
            for i in np.unique(result["labels"]):
                count = int(np.sum(result["labels"] == i))
                stats_text += f"  クラスタ {i}: {count}個\n"

            self.stats_text.delete("1.0", tk.END)
            self.stats_text.insert("1.0", stats_text)

            if len(result["features"]) >= 2:
                self.plot_cluster_distribution(self.cluster_ax, result["features"], result["labels"])
            else:
                self.cluster_ax.clear()
                self.cluster_ax.text(0.5, 0.5, "2件以上で分布表示", ha="center", va="center")
                self.cluster_ax.set_axis_off()

            self.nodule_listbox.delete(0, tk.END)
            for i, (nodule, label) in enumerate(zip(result["nodules"], result["labels"])):
                item_text = f"結節 #{i+1} - クラスタ {label} - 面積: {nodule['area']}"
                self.nodule_listbox.insert(tk.END, item_text)

        else:
            anomaly_labels = result["is_anomaly"].astype(int)
            self.plot_nodules(
                self.result_ax,
                result["enhanced"],
                result["nodules"],
                anomaly_labels,
                "異常検知結果"
            )

            n_anomalies = int(np.sum(result["is_anomaly"]))
            stats_text = f"""検出結節数: {len(result['nodules'])}
異常検出数: {n_anomalies}
正常: {len(result['nodules']) - n_anomalies}

異常スコア統計:
  平均: {np.mean(result['anomaly_scores']):.3f}
  最大: {np.max(result['anomaly_scores']):.3f}
  最小: {np.min(result['anomaly_scores']):.3f}
"""
            self.stats_text.delete("1.0", tk.END)
            self.stats_text.insert("1.0", stats_text)

            self.cluster_ax.clear()
            self.cluster_ax.hist(result["anomaly_scores"], bins=10)
            self.cluster_ax.set_title("異常スコア分布")
            self.cluster_ax.set_xlabel("score")
            self.cluster_ax.set_ylabel("count")

            self.nodule_listbox.delete(0, tk.END)
            for i, (nodule, score, is_anom) in enumerate(zip(
                result["nodules"],
                result["anomaly_scores"],
                result["is_anomaly"]
            )):
                status = "異常" if is_anom else "正常"
                item_text = f"結節 #{i+1} - {status} - スコア: {score:.3f}"
                self.nodule_listbox.insert(tk.END, item_text)

        self.result_canvas.draw()
        self.cluster_canvas.draw()

        self.progress["value"] = 100
        self.status_bar.config(text="解析完了")

    def plot_image(self, ax, image, title):
        ax.clear()
        ax.imshow(image, cmap="gray")
        ax.set_title(title, fontweight="bold")
        ax.axis("off")

    def plot_nodules(self, ax, image, nodules, labels, title):
        ax.clear()
        ax.imshow(image, cmap="gray")

        colors = ["red", "yellow", "cyan", "lime", "magenta"]

        for i, nodule in enumerate(nodules):
            x, y, w, h = nodule["bbox"]

            if labels is not None and i < len(labels):
                color = colors[int(labels[i]) % len(colors)]
                label_text = f"C{int(labels[i])}"
            else:
                color = "red"
                label_text = f"N{i}"

            rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor="none")
            ax.add_patch(rect)

            ax.text(
                x, max(0, y - 5), label_text,
                color=color,
                fontsize=8,
                fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.5)
            )

        ax.set_title(title, fontweight="bold")
        ax.axis("off")

    def plot_cluster_distribution(self, ax, features, labels):
        ax.clear()

        if len(features) < 2:
            ax.text(0.5, 0.5, "データ不足", ha="center", va="center")
            ax.set_axis_off()
            return

        n_comp = min(2, features.shape[0], features.shape[1])
        if n_comp < 2:
            ax.text(0.5, 0.5, "2次元PCA不可", ha="center", va="center")
            ax.set_axis_off()
            return

        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)

        colors = ["red", "blue", "green", "orange", "purple"]
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            ax.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                c=colors[int(cluster_id) % len(colors)],
                label=f"Cluster {int(cluster_id)}",
                alpha=0.6,
                s=100
            )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("特徴空間でのクラスタ分布", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def save_results(self):
        if self.current_result is None:
            messagebox.showwarning("警告", "保存する結果がありません")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("すべて", "*.*")],
            initialfile=f"lung_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        if not file_path:
            return

        try:
            save_data = {
                "timestamp": datetime.now().isoformat(),
                "mode": self.current_result.get("mode", "unknown"),
                "n_nodules": len(self.current_result.get("nodules", []))
            }

            if "labels" in self.current_result:
                save_data["cluster_distribution"] = {
                    int(i): int(np.sum(self.current_result["labels"] == i))
                    for i in np.unique(self.current_result["labels"])
                }

            if "anomaly_scores" in self.current_result:
                save_data["anomaly_stats"] = {
                    "mean": float(np.mean(self.current_result["anomaly_scores"])),
                    "max": float(np.max(self.current_result["anomaly_scores"])),
                    "n_anomalies": int(np.sum(self.current_result["is_anomaly"]))
                }

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)

            messagebox.showinfo("保存完了", "結果を保存しました")

        except Exception as e:
            messagebox.showerror("エラー", f"保存失敗:\n{e}")


def main():
    if MISSING:
        root = tk.Tk()
        root.withdraw()
        detail = "\n".join(IMPORT_ERRORS) if IMPORT_ERRORS else "不明"
        messagebox.showerror(
            "起動エラー",
            "必要ライブラリが不足しています。\n\n"
            f"不足候補: {', '.join(MISSING)}\n\n"
            f"詳細:\n{detail}"
        )
        root.destroy()
        return

    if not MATPLOTLIB_AVAILABLE:
        root = tk.Tk()
        root.withdraw()
        detail = "\n".join(IMPORT_ERRORS)
        messagebox.showerror("起動エラー", f"matplotlib/Tk の初期化に失敗しました。\n\n{detail}")
        root.destroy()
        return

    root = tk.Tk()
    app = LungCancerQuantumGUI(root)
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()
        log_path = write_error_log(err)
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "致命的エラー",
                f"起動時にエラーが発生しました。\n\nログ:\n{log_path}"
            )
            root.destroy()
        except Exception:
            pass
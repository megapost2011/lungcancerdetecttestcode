import os
import threading
import traceback
from datetime import datetime

import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


class LungCTSampleGenerator:
    """肺CT画像サンプル生成クラス"""

    def __init__(self, output_dir='lung_ct_samples', log_func=None):
        self.output_dir = output_dir
        self.image_size = (512, 512)
        self.log = log_func if log_func else print
        os.makedirs(output_dir, exist_ok=True)

    def log_msg(self, text):
        self.log(text)

    def generate_normal_lung(self):
        self.log_msg("正常肺CT画像を生成中...")
        image = np.zeros(self.image_size, dtype=np.float32)

        cv2.ellipse(image, (256, 256), (200, 220), 0, 0, 360, -1000, -1)
        cv2.ellipse(image, (320, 256), (100, 150), 0, 0, 360, -800, -1)
        cv2.ellipse(image, (192, 256), (90, 140), 0, 0, 360, -800, -1)
        cv2.ellipse(image, (256, 280), (40, 60), 0, 0, 360, -200, -1)

        self._add_vasculature(image, (320, 200), depth=3, intensity=-700)
        self._add_vasculature(image, (192, 200), depth=3, intensity=-700)

        cv2.line(image, (256, 180), (256, 280), -850, 8)
        cv2.circle(image, (256, 180), 15, -900, -1)

        noise = np.random.normal(-800, 50, self.image_size)
        image = image + noise * 0.3

        image = np.clip(image, -1000, 400)
        normalized = (image + 1000) / 1400 * 255
        return normalized.astype(np.uint8)

    def generate_benign_nodule_lung(self):
        self.log_msg("良性結節CT画像を生成中...")
        image = self.generate_normal_lung().astype(np.float32)
        image = (image / 255.0 * 1400) - 1000

        nodule_positions = [
            (350, 220),
            (180, 300)
        ]

        for pos in nodule_positions[:np.random.randint(1, 3)]:
            radius = np.random.randint(8, 15)
            cv2.circle(image, pos, radius, -300, -1)

            if np.random.random() > 0.5:
                calc_pos = (
                    pos[0] + np.random.randint(-3, 3),
                    pos[1] + np.random.randint(-3, 3)
                )
                cv2.circle(image, calc_pos, 2, 400, -1)

        noise = np.random.normal(0, 30, self.image_size)
        image = image + noise

        image = np.clip(image, -1000, 400)
        normalized = (image + 1000) / 1400 * 255
        return normalized.astype(np.uint8)

    def generate_malignant_nodule_lung(self):
        self.log_msg("悪性結節CT画像を生成中...")
        image = self.generate_normal_lung().astype(np.float32)
        image = (image / 255.0 * 1400) - 1000

        nodule_center = (340, 240)

        for _ in range(5):
            offset_x = np.random.randint(-10, 10)
            offset_y = np.random.randint(-10, 10)
            center = (nodule_center[0] + offset_x, nodule_center[1] + offset_y)
            axes = (np.random.randint(12, 20), np.random.randint(10, 18))
            angle = np.random.randint(0, 180)
            cv2.ellipse(image, center, axes, angle, 0, 360, -200, -1)

        for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            length = np.random.randint(15, 30)
            end_x = int(nodule_center[0] + length * np.cos(angle))
            end_y = int(nodule_center[1] + length * np.sin(angle))
            cv2.line(image, nodule_center, (end_x, end_y), -400, 2)

        ggo_mask = np.zeros_like(image)
        cv2.circle(ggo_mask, nodule_center, 35, 1, -1)
        image = image - ggo_mask * 200

        cv2.line(
            image,
            nodule_center,
            (nodule_center[0] + 50, nodule_center[1] - 40),
            -500,
            3
        )

        noise = np.random.normal(0, 35, self.image_size)
        image = image + noise

        image = np.clip(image, -1000, 400)
        normalized = (image + 1000) / 1400 * 255
        return normalized.astype(np.uint8)

    def generate_multiple_nodules_lung(self):
        self.log_msg("多発結節CT画像を生成中...")
        image = self.generate_normal_lung().astype(np.float32)
        image = (image / 255.0 * 1400) - 1000

        n_nodules = np.random.randint(3, 7)

        for _ in range(n_nodules):
            if np.random.random() > 0.5:
                x = np.random.randint(270, 400)
            else:
                x = np.random.randint(130, 240)

            y = np.random.randint(180, 330)
            nodule_type = np.random.choice(['benign', 'suspicious', 'malignant'])

            if nodule_type == 'benign':
                radius = np.random.randint(5, 10)
                cv2.circle(image, (x, y), radius, -300, -1)

            elif nodule_type == 'suspicious':
                radius = np.random.randint(8, 15)
                cv2.circle(image, (x, y), radius, -250, -1)
                for angle in np.linspace(0, 2*np.pi, 6, endpoint=False):
                    dx = int(x + radius * 1.5 * np.cos(angle))
                    dy = int(y + radius * 1.5 * np.sin(angle))
                    cv2.line(image, (x, y), (dx, dy), -400, 1)

            else:
                for _ in range(3):
                    offset = np.random.randint(-5, 5)
                    cv2.circle(
                        image,
                        (x + offset, y + offset),
                        np.random.randint(8, 12),
                        -200,
                        -1
                    )

        noise = np.random.normal(0, 30, self.image_size)
        image = image + noise

        image = np.clip(image, -1000, 400)
        normalized = (image + 1000) / 1400 * 255
        return normalized.astype(np.uint8)

    def _add_vasculature(self, image, start_pos, depth, intensity, angle=90):
        if depth == 0:
            return

        length = 30 - depth * 5
        thickness = max(1, 4 - depth)

        rad = np.deg2rad(angle)
        end_x = int(start_pos[0] + length * np.cos(rad))
        end_y = int(start_pos[1] + length * np.sin(rad))

        cv2.line(image, start_pos, (end_x, end_y), intensity, thickness)

        if depth > 1:
            self._add_vasculature(image, (end_x, end_y), depth - 1, intensity, angle - 30)
            self._add_vasculature(image, (end_x, end_y), depth - 1, intensity, angle + 30)

    def add_ct_artifacts(self, image):
        center = (256, 256)
        y, x = np.ogrid[:512, :512]
        distance = np.sqrt((x - center[0])**2 + (y - center[1])**2)

        beam_hardening = 1 + 0.1 * (distance / 256)
        image = np.clip(image.astype(np.float32) * beam_hardening, 0, 255).astype(np.uint8)

        for r in [100, 150, 200]:
            mask = np.abs(distance - r) < 2
            delta = np.random.randint(-5, 5)
            image[mask] = np.clip(image[mask].astype(np.int16) + delta, 0, 255).astype(np.uint8)

        for angle in np.linspace(0, np.pi, 4, endpoint=False):
            for i in range(1, 200):
                xx = int(256 + i * np.cos(angle))
                yy = int(256 + i * np.sin(angle))
                if 0 <= xx < 512 and 0 <= yy < 512:
                    delta = np.random.randint(-3, 3)
                    image[yy, xx] = np.clip(int(image[yy, xx]) + delta, 0, 255)

        return image

    def generate_all_samples(self, normal_count, benign_count, malignant_count, multiple_count):
        self.log_msg("=" * 60)
        self.log_msg("肺癌CTサンプル画像生成")
        self.log_msg("=" * 60)

        samples = []

        for i in range(normal_count):
            img = self.add_ct_artifacts(self.generate_normal_lung())
            filename = f'normal_lung_{i+1}.png'
            save_path = os.path.join(self.output_dir, filename)
            ok = cv2.imwrite(save_path, img)
            if not ok:
                raise RuntimeError(f"画像保存失敗: {save_path}")
            samples.append(('正常', filename))
            self.log_msg(f"✓ {filename}")

        for i in range(benign_count):
            img = self.add_ct_artifacts(self.generate_benign_nodule_lung())
            filename = f'benign_nodule_{i+1}.png'
            save_path = os.path.join(self.output_dir, filename)
            ok = cv2.imwrite(save_path, img)
            if not ok:
                raise RuntimeError(f"画像保存失敗: {save_path}")
            samples.append(('良性結節', filename))
            self.log_msg(f"✓ {filename}")

        for i in range(malignant_count):
            img = self.add_ct_artifacts(self.generate_malignant_nodule_lung())
            filename = f'malignant_nodule_{i+1}.png'
            save_path = os.path.join(self.output_dir, filename)
            ok = cv2.imwrite(save_path, img)
            if not ok:
                raise RuntimeError(f"画像保存失敗: {save_path}")
            samples.append(('悪性結節', filename))
            self.log_msg(f"✓ {filename}")

        for i in range(multiple_count):
            img = self.add_ct_artifacts(self.generate_multiple_nodules_lung())
            filename = f'multiple_nodules_{i+1}.png'
            save_path = os.path.join(self.output_dir, filename)
            ok = cv2.imwrite(save_path, img)
            if not ok:
                raise RuntimeError(f"画像保存失敗: {save_path}")
            samples.append(('多発結節', filename))
            self.log_msg(f"✓ {filename}")

        self._write_summary(samples)
        self._save_npy_files(samples)

        self.log_msg("")
        self.log_msg("=" * 60)
        self.log_msg(f"生成完了！合計 {len(samples)} 枚")
        self.log_msg(f"保存先: {self.output_dir}")
        self.log_msg("=" * 60)

    def _write_summary(self, samples):
        summary_path = os.path.join(self.output_dir, 'DATASET_INFO.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("肺癌CTサンプルデータセット\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"画像数: {len(samples)}\n")
            f.write("画像サイズ: 512×512\n\n")

            f.write("カテゴリ別内訳:\n")
            f.write("-" * 60 + "\n")
            categories = {}
            for cat, _ in samples:
                categories[cat] = categories.get(cat, 0) + 1
            for cat, count in categories.items():
                f.write(f"{cat}: {count}枚\n")

            f.write("\nファイル一覧:\n")
            f.write("-" * 60 + "\n")
            for cat, filename in samples:
                f.write(f"{filename:30s} - {cat}\n")

            f.write("\n特徴:\n")
            f.write("-" * 60 + "\n")
            f.write("- 正常肺: リアルな血管・気管支構造\n")
            f.write("- 良性結節: 境界明瞭、円形、石灰化の可能性\n")
            f.write("- 悪性結節: Spiculation、不規則形状、GGO\n")
            f.write("- 多発結節: 複数タイプの結節が混在\n\n")

            f.write("CTアーティファクト:\n")
            f.write("-" * 60 + "\n")
            f.write("- ビームハードニング\n")
            f.write("- リングアーティファクト\n")
            f.write("- ストリークアーティファクト\n\n")

            f.write("注意事項:\n")
            f.write("-" * 60 + "\n")
            f.write("このデータセットは合成画像です。\n")
            f.write("教育・研究目的でのみ使用してください。\n")
            f.write("実際の臨床診断には使用しないでください。\n")

        self.log_msg(f"データセット情報: {summary_path}")

    def _save_npy_files(self, samples):
        self.log_msg("NumPy形式でも保存中...")
        for _, filename in samples:
            img_path = os.path.join(self.output_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"画像読込失敗: {img_path}")
            npy_filename = filename.replace('.png', '.npy')
            np.save(os.path.join(self.output_dir, npy_filename), img)
        self.log_msg("✓ NumPy形式の保存完了")


class LungSampleGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("肺癌CTサンプル生成GUI")
        self.root.geometry("900x700")

        self.output_dir = tk.StringVar(value=os.path.join(os.getcwd(), "lung_ct_samples"))
        self.normal_count = tk.IntVar(value=3)
        self.benign_count = tk.IntVar(value=3)
        self.malignant_count = tk.IntVar(value=3)
        self.multiple_count = tk.IntVar(value=2)
        self.status_var = tk.StringVar(value="待機中")

        self._build_ui()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        top = ttk.LabelFrame(main, text="出力設定", padding=10)
        top.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(top, text="保存先フォルダ").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(top, textvariable=self.output_dir, width=70).grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(top, text="参照", command=self.select_folder).grid(row=0, column=2, padx=5, pady=5)
        top.columnconfigure(1, weight=1)

        counts = ttk.LabelFrame(main, text="生成枚数", padding=10)
        counts.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(counts, text="正常肺").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Spinbox(counts, from_=0, to=100, textvariable=self.normal_count, width=10).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(counts, text="良性結節").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        ttk.Spinbox(counts, from_=0, to=100, textvariable=self.benign_count, width=10).grid(row=0, column=3, padx=5, pady=5)

        ttk.Label(counts, text="悪性結節").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ttk.Spinbox(counts, from_=0, to=100, textvariable=self.malignant_count, width=10).grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(counts, text="多発結節").grid(row=1, column=2, padx=5, pady=5, sticky="w")
        ttk.Spinbox(counts, from_=0, to=100, textvariable=self.multiple_count, width=10).grid(row=1, column=3, padx=5, pady=5)

        btns = ttk.Frame(main)
        btns.pack(fill=tk.X, pady=(0, 10))

        self.start_btn = ttk.Button(btns, text="生成開始", command=self.start_generation)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(btns, text="保存先を開く", command=self.open_output_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(btns, text="ログ消去", command=self.clear_log).pack(side=tk.LEFT, padx=5)

        status_frame = ttk.Frame(main)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(status_frame, text="状態:").pack(side=tk.LEFT)
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=5)

        log_frame = ttk.LabelFrame(main, text="ログ", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(log_frame, wrap="none", height=25)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        y_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=y_scroll.set)

    def log(self, text):
        self.root.after(0, self._append_log, text)

    def _append_log(self, text):
        self.log_text.insert(tk.END, str(text) + "\n")
        self.log_text.see(tk.END)

    def clear_log(self):
        self.log_text.delete("1.0", tk.END)

    def select_folder(self):
        folder = filedialog.askdirectory(initialdir=self.output_dir.get() or os.getcwd())
        if folder:
            self.output_dir.set(folder)

    def open_output_folder(self):
        folder = self.output_dir.get().strip()
        if not folder:
            messagebox.showwarning("警告", "保存先フォルダが空です。")
            return

        os.makedirs(folder, exist_ok=True)
        try:
            os.startfile(folder)
        except Exception as e:
            messagebox.showerror("エラー", f"フォルダを開けませんでした。\n{e}")

    def start_generation(self):
        output_dir = self.output_dir.get().strip()
        if not output_dir:
            messagebox.showwarning("警告", "保存先フォルダを指定してください。")
            return

        total = (
            self.normal_count.get() +
            self.benign_count.get() +
            self.malignant_count.get() +
            self.multiple_count.get()
        )
        if total <= 0:
            messagebox.showwarning("警告", "生成枚数を1枚以上にしてください。")
            return

        self.start_btn.config(state=tk.DISABLED)
        self.status_var.set("生成中...")
        self.clear_log()

        thread = threading.Thread(target=self._run_generation, daemon=True)
        thread.start()

    def _run_generation(self):
        try:
            output_dir = self.output_dir.get().strip()
            os.makedirs(output_dir, exist_ok=True)

            generator = LungCTSampleGenerator(output_dir=output_dir, log_func=self.log)
            generator.generate_all_samples(
                normal_count=self.normal_count.get(),
                benign_count=self.benign_count.get(),
                malignant_count=self.malignant_count.get(),
                multiple_count=self.multiple_count.get()
            )

            self.root.after(0, self._generation_success)

        except Exception as e:
            err = traceback.format_exc()
            self.log("エラーが発生しました。")
            self.log(str(e))
            self.log(err)
            self.root.after(0, lambda: self._generation_error(str(e)))

    def _generation_success(self):
        self.start_btn.config(state=tk.NORMAL)
        self.status_var.set("完了")
        messagebox.showinfo("完了", "サンプル画像の生成が完了しました。")

    def _generation_error(self, msg):
        self.start_btn.config(state=tk.NORMAL)
        self.status_var.set("エラー")
        messagebox.showerror("エラー", f"生成中にエラーが発生しました。\n\n{msg}")


def main():
    try:
        root = tk.Tk()
        app = LungSampleGUI(root)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("起動エラー", f"GUI起動に失敗しました。\n\n{e}")


if __name__ == "__main__":
    main()
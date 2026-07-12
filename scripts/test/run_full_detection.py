"""
全量检测脚本：对 data/test 下所有图片进行裁剪 + PaDiM 推理。

裁剪和推理均在内存中完成，不生成中间 patch 文件。
仅对 NG 图片保存可视化结果。

用法：
    python scripts/test/run_full_detection.py
"""

import os
os.environ["TRUST_REMOTE_CODE"] = "1"

import sys
import io
import time
import glob
import warnings
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision.transforms.v2.functional import to_dtype, to_image
from anomalib.deploy import TorchInferencer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from lib.roi import extract_roi_from_image

warnings.filterwarnings("ignore", category=UserWarning, module="anomalib")

# ============================================================
# 参数配置
# ============================================================
ANGLE_OFFSET = 2.8
OFFSET_FROM_PEAK = 33
ROI_WIDTH = 55
PATCH_SIZE = 256
FLIP_VERTICAL = True
SCORE_THRESHOLD = 42.77  # P99


def slice_image_to_patches(img):
    """在内存中将图片裁剪为 patches，不写磁盘。"""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        result = extract_roi_from_image(
            img,
            rotate_angle=None,
            angle_offset=ANGLE_OFFSET,
            offset_from_peak=OFFSET_FROM_PEAK,
            roi_width=ROI_WIDTH,
            angle_search_range=(-8.0, 8.0),
            angle_search_step=0.2,
            flip_vertical=FLIP_VERTICAL,
        )
    finally:
        sys.stdout = old_stdout

    if result is None:
        return []

    roi_strip = result["roi_strip"]
    window_w = roi_strip.shape[0]
    stride = int(window_w * 0.5)
    x_start = 0
    patches = []

    while (x_start + window_w) <= roi_strip.shape[1]:
        square_patch = roi_strip[:, x_start:x_start + window_w]
        resized_patch = cv2.resize(square_patch, (PATCH_SIZE, PATCH_SIZE),
                                   interpolation=cv2.INTER_CUBIC)
        patches.append(resized_patch)
        x_start += stride

    return patches


def infer_patch(inferencer, patch_bgr):
    """对单个 patch 做推理，返回 (score, predictions)"""
    lightning_model = inferencer.model
    pre_processor = getattr(lightning_model, "pre_processor", None)

    img_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = to_dtype(to_image(img_rgb), torch.float32, scale=True)

    with torch.no_grad():
        proc_input = img_tensor.unsqueeze(0).to(inferencer.device)
        proc_output = pre_processor(proc_input) if pre_processor is not None else proc_input
        predictions = lightning_model.model(proc_output)

    return float(predictions.pred_score), predictions


def safe_makedirs(path):
    """尝试创建目录，若权限不足则回退到临时目录"""
    try:
        os.makedirs(path, exist_ok=True)
        test_file = os.path.join(path, ".write_test")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        return path
    except (PermissionError, OSError):
        fallback = os.path.join(os.path.expanduser("~"), ".trae-cn", "work",
                                "6a50fd0baa3e386cea3adf32", "all_frames_results")
        os.makedirs(os.path.join(fallback, "ng_visualizations"), exist_ok=True)
        print(f"⚠️ 项目目录不可写，结果输出到临时目录: {fallback}")
        return fallback


def main():
    MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "models", "padim", "torch", "model.pt")
    TEST_DIR = os.path.join(PROJECT_ROOT, "data", "test", "images")

    preferred_output = os.path.join(PROJECT_ROOT, "outputs", "patches", "_all_frames_results")
    OUTPUT_DIR = safe_makedirs(preferred_output)
    ng_vis_dir = os.path.join(OUTPUT_DIR, "ng_visualizations")
    os.makedirs(ng_vis_dir, exist_ok=True)

    # 加载模型
    print("=" * 60)
    print("🚀 加载 PaDiM 模型...")
    inferencer = TorchInferencer(path=MODEL_PATH, device="cpu")
    print("✅ 模型加载完成")

    # 收集所有图片
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(TEST_DIR, ext)))
    image_paths = sorted(image_paths)

    total = len(image_paths)
    print(f"📁 共 {total} 张待检测图片")
    print(f"📐 阈值: {SCORE_THRESHOLD} (P99)")
    print(f"💾 结果输出至: {OUTPUT_DIR}")
    print("=" * 60)
    sys.stdout.flush()

    all_results = []
    ng_images = []
    start_time = time.time()

    for idx, img_path in enumerate(image_paths):
        img_name = os.path.basename(img_path)

        elapsed = time.time() - start_time
        if idx > 0:
            eta = elapsed / idx * (total - idx)
            print(f"[{idx+1}/{total}] {img_name} (已用{elapsed:.0f}s, 预计剩余{eta:.0f}s)", end="", flush=True)
        else:
            print(f"[{idx+1}/{total}] {img_name}", end="", flush=True)

        img = cv2.imread(img_path)
        if img is None:
            print(" ❌ 无法读取")
            all_results.append({"image": img_name, "patch_count": 0,
                                "max_score": 0, "mean_score": 0,
                                "ng_patches": 0, "status": "ERROR"})
            continue

        patches = slice_image_to_patches(img)
        if not patches:
            print(" ❌ 无法定位ROI")
            all_results.append({"image": img_name, "patch_count": 0,
                                "max_score": 0, "mean_score": 0,
                                "ng_patches": 0, "status": "ERROR"})
            continue

        patch_scores = []
        max_score = 0
        max_patch_idx = 0
        max_predictions = None
        max_patch_img = None

        for pidx, patch_img in enumerate(patches):
            score, predictions = infer_patch(inferencer, patch_img)
            patch_scores.append(score)
            if score > max_score:
                max_score = score
                max_patch_idx = pidx
                max_predictions = predictions
                max_patch_img = patch_img

        is_ng = max_score >= SCORE_THRESHOLD
        mean_score = float(np.mean(patch_scores))
        ng_patch_count = sum(1 for s in patch_scores if s >= SCORE_THRESHOLD)

        status = "❌ NG" if is_ng else "✅ OK"
        print(f" | {len(patches)}patches | max={max_score:.2f} | {status}", flush=True)

        if is_ng and max_predictions is not None:
            try:
                heatmap = max_predictions.anomaly_map.detach().cpu().numpy().squeeze()
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                axes[0].set_title(f"Original: {img_name}")
                axes[0].axis("off")

                axes[1].imshow(cv2.cvtColor(max_patch_img, cv2.COLOR_BGR2RGB))
                axes[1].set_title(f"NG Patch #{max_patch_idx} (score={max_score:.2f})")
                axes[1].axis("off")

                im = axes[2].imshow(heatmap, cmap="jet")
                axes[2].set_title("Anomaly Heatmap")
                axes[2].axis("off")
                fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

                base_name = os.path.splitext(img_name)[0]
                save_path = os.path.join(ng_vis_dir, f"{base_name}_ng.jpg")
                plt.tight_layout()
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                print(f"   ⚠️ 保存可视化失败: {e}")

        all_results.append({
            "image": img_name,
            "patch_count": len(patches),
            "max_score": max_score,
            "mean_score": mean_score,
            "ng_patches": ng_patch_count,
            "status": "NG" if is_ng else "OK",
        })

        if is_ng:
            ng_images.append(img_name)

        if (idx + 1) % 100 == 0:
            _save_csv(all_results, OUTPUT_DIR)

    # 汇总
    total_time = time.time() - start_time
    total_patches = sum(r["patch_count"] for r in all_results)
    ok_count = sum(1 for r in all_results if r["status"] == "OK")
    ng_count = sum(1 for r in all_results if r["status"] == "NG")
    error_count = sum(1 for r in all_results if r["status"] == "ERROR")

    print("\n" + "=" * 60)
    print("📊 全量检测结果汇总")
    print("=" * 60)
    print(f"  总图片数:    {total}")
    print(f"  正常 (OK):   {ok_count}")
    print(f"  异常 (NG):   {ng_count}")
    print(f"  错误:        {error_count}")
    print(f"  总 patch 数: {total_patches}")
    print(f"  耗时:        {total_time:.1f}s ({total_time/max(total,1):.2f}s/张)")

    if ng_images:
        print(f"\n📋 NG 图片列表 ({ng_count} 张):")
        for name in ng_images:
            r = next(x for x in all_results if x["image"] == name)
            print(f"  ❌ {name}  (max={r['max_score']:.2f}, ng_patches={r['ng_patches']}/{r['patch_count']})")

    _save_csv(all_results, OUTPUT_DIR)
    csv_path = os.path.join(OUTPUT_DIR, "detection_results.csv")
    print(f"\n📝 详细结果 CSV: {csv_path}")
    if ng_count > 0:
        print(f"🖼️ NG 可视化: {ng_vis_dir}")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print("=" * 60)


def _save_csv(results, output_dir):
    csv_path = os.path.join(output_dir, "detection_results.csv")
    try:
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("image,patch_count,max_score,mean_score,ng_patches,status\n")
            for r in results:
                f.write(f"{r['image']},{r['patch_count']},{r['max_score']:.6f},"
                        f"{r['mean_score']:.6f},{r['ng_patches']},{r['status']}\n")
    except PermissionError:
        fallback = os.path.join(os.path.expanduser("~"), ".trae-cn", "work",
                                "6a50fd0baa3e386cea3adf32", "all_frames_results")
        os.makedirs(fallback, exist_ok=True)
        csv_path = os.path.join(fallback, "detection_results.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("image,patch_count,max_score,mean_score,ng_patches,status\n")
            for r in results:
                f.write(f"{r['image']},{r['patch_count']},{r['max_score']:.6f},"
                        f"{r['mean_score']:.6f},{r['ng_patches']},{r['status']}\n")


if __name__ == "__main__":
    main()
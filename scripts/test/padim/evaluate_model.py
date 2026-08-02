"""
模型评估脚本：对训练集、测试正常集、NG 异常集分别推理打分。

用法：
    python scripts/test/evaluate_model.py
"""

import os
os.environ["TRUST_REMOTE_CODE"] = "1"

import sys
import io
import glob
import random
import warnings

import cv2
import numpy as np
import torch
from torchvision.transforms.v2.functional import to_dtype, to_image
from anomalib.deploy import TorchInferencer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from lib.roi import extract_roi_from_image

warnings.filterwarnings("ignore")

# 训练集参数（与 build_dataset.py 一致）
TRAIN_FLIP = False
TRAIN_ROTATE = -2.0
TRAIN_ANGLE_OFFSET = 0.0

# 测试/NG 推理参数（与 run_full_detection.py 一致）
TEST_FLIP = True
TEST_ANGLE_OFFSET = 2.8

OFFSET_FROM_PEAK = 33
ROI_WIDTH = 55
PATCH_SIZE = 256
SAMPLE_SIZE = 30


def extract_patches(img, flip_vertical, rotate_angle=None, angle_offset=0.0):
    """提取 ROI 并切 patch（无重叠，加速评估）。"""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        result = extract_roi_from_image(
            img,
            rotate_angle=rotate_angle,
            angle_offset=angle_offset,
            offset_from_peak=OFFSET_FROM_PEAK,
            roi_width=ROI_WIDTH,
            flip_vertical=flip_vertical,
        )
    finally:
        sys.stdout = old_stdout

    if result is None:
        return []

    roi_strip = result["roi_strip"]
    window_w = roi_strip.shape[0]
    stride = window_w  # 无重叠
    patches = []
    x_start = 0
    while x_start + window_w <= roi_strip.shape[1]:
        patch = roi_strip[:, x_start:x_start + window_w]
        patch = cv2.resize(patch, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_CUBIC)
        patches.append(patch)
        x_start += stride
    return patches


def infer_patch(inferencer, patch_bgr):
    """对单个 patch 推理，返回异常分数。"""
    lightning_model = inferencer.model
    pre_processor = getattr(lightning_model, "pre_processor", None)
    img_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = to_dtype(to_image(img_rgb), torch.float32, scale=True)
    with torch.no_grad():
        proc_input = img_tensor.unsqueeze(0).to(inferencer.device)
        proc_output = pre_processor(proc_input) if pre_processor is not None else proc_input
        predictions = lightning_model.model(proc_output)
    return float(predictions.pred_score)


def evaluate(image_paths, label, flip_vertical, rotate_angle=None, angle_offset=0.0):
    """评估一组图片，返回每张图的 max patch score。"""
    random.seed(42)
    sampled = random.sample(image_paths, min(SAMPLE_SIZE, len(image_paths)))
    scores = []
    for path in sampled:
        img = cv2.imread(path)
        if img is None:
            continue
        patches = extract_patches(img, flip_vertical, rotate_angle, angle_offset)
        if not patches:
            continue
        patch_scores = [infer_patch(inferencer, p) for p in patches]
        scores.append(max(patch_scores))
    return scores


def print_stats(label, scores):
    if not scores:
        print(f"  {label}: 无有效数据")
        return
    arr = np.array(scores)
    print(f"  {label:12s} | n={len(arr):3d} | "
          f"min={arr.min():7.2f} | max={arr.max():7.2f} | "
          f"mean={arr.mean():7.2f} | median={np.median(arr):7.2f} | "
          f"std={arr.std():6.2f}")


if __name__ == "__main__":
    print("=" * 70)
    print("📊 模型评估：训练集 vs 测试正常 vs NG 异常")
    print("=" * 70)

    # 加载模型
    print("🔄 加载模型...")
    inferencer = TorchInferencer(
        path=os.path.join(PROJECT_ROOT, "outputs", "models", "padim", "torch", "model.pt"),
        device="cpu",
    )
    print("✅ 模型加载完成\n")

    # 收集图片
    train_imgs = []
    for ext in ("*.jpg", "*.png"):
        for subdir in glob.glob(os.path.join(PROJECT_ROOT, "data", "train", "*")):
            train_imgs.extend(glob.glob(os.path.join(subdir, ext)))

    test_normal_imgs = []
    for ext in ("*.jpg", "*.png"):
        test_normal_imgs.extend(glob.glob(os.path.join(PROJECT_ROOT, "data", "test", "images", ext)))

    ng_imgs = glob.glob(os.path.join(PROJECT_ROOT, "data", "ng", "ng_ng", "*.jpg"))

    print(f"📁 训练集图片池: {len(train_imgs)} 张")
    print(f"📁 测试正常图片池: {len(test_normal_imgs)} 张")
    print(f"📁 NG 异常图片池: {len(ng_imgs)} 张")
    print(f"🔧 每组采样: {SAMPLE_SIZE} 张\n")

    # 评估三组
    print("🔄 正在推理训练集...")
    train_scores = evaluate(train_imgs, "train", TRAIN_FLIP, TRAIN_ROTATE, TRAIN_ANGLE_OFFSET)

    print("🔄 正在推理测试正常集...")
    test_scores = evaluate(test_normal_imgs, "test_normal", TEST_FLIP, None, TEST_ANGLE_OFFSET)

    print("🔄 正在推理 NG 异常集...")
    ng_scores = evaluate(ng_imgs, "ng", TEST_FLIP, None, TEST_ANGLE_OFFSET)

    # 汇总
    print("\n" + "=" * 70)
    print("📈 评估结果汇总（每张图取所有 patch 的最高异常分数）")
    print("=" * 70)
    print_stats("训练集", train_scores)
    print_stats("测试正常", test_scores)
    print_stats("NG异常", ng_scores)

    # 分析区分能力
    print("\n" + "=" * 70)
    print("🔍 区分能力分析")
    print("=" * 70)
    if train_scores and test_scores and ng_scores:
        train_max = max(train_scores)
        test_max = max(test_scores)
        ng_min = min(ng_scores)
        ng_mean = np.mean(ng_scores)
        normal_max = max(max(train_scores), max(test_scores))

        print(f"  正常样本最高分: {normal_max:.2f}")
        print(f"  NG样本最低分:   {ng_min:.2f}")
        print(f"  NG样本均分:     {ng_mean:.2f}")

        if ng_min > normal_max:
            print(f"\n  ✅ 模型可以完美区分：正常最高分({normal_max:.2f}) < NG最低分({ng_min:.2f})")
            print(f"     建议阈值: {(normal_max + ng_min) / 2:.2f}")
        else:
            overlap = normal_max - ng_min
            print(f"\n  ⚠️ 存在分数重叠区间: [{ng_min:.2f}, {normal_max:.2f}]")
            print(f"     重叠程度: {overlap:.2f}")
            # 计算某个阈值下的准确率
            threshold = np.median(train_scores + test_scores + ng_scores)
            normal_correct = sum(1 for s in train_scores + test_scores if s < threshold)
            ng_correct = sum(1 for s in ng_scores if s >= threshold)
            total = len(train_scores) + len(test_scores) + len(ng_scores)
            print(f"     以中位数 {threshold:.2f} 为阈值:")
            print(f"       正常检出率: {normal_correct}/{len(train_scores) + len(test_scores)}")
            print(f"       NG检出率:   {ng_correct}/{len(ng_scores)}")
            print(f"       总体准确率: {(normal_correct + ng_correct) / total * 100:.1f}%")

    print("=" * 70)

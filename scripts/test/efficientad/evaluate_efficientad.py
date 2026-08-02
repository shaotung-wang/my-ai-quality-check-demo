"""
EfficientAd 模型评估脚本：对训练集、测试正常集、NG 异常集分别推理打分。

与 evaluate_model.py 的差异：
  - 加载 EfficientAd 而非 PaDiM
  - 默认加载按正确 ROI 续训后的 v1 Lightning checkpoint
  - 默认用 P95 聚合 patch 分数，抑制孤立反光造成的最大值尖峰

用法：
    .venv/bin/python scripts/test/efficientad/evaluate_efficientad.py
"""

import os
os.environ["TRUST_REMOTE_CODE"] = "1"

import sys
import io
import glob
import random
import warnings
import argparse

import cv2
import numpy as np
import torch
from torchvision.transforms.v2.functional import to_dtype, to_image
from anomalib.models import EfficientAd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from lib.roi import extract_roi_from_image, roi_settings_for_path
from lib.scoring import aggregate_patch_scores

warnings.filterwarnings("ignore")

OFFSET_FROM_PEAK = 33
ROI_WIDTH = 55
PATCH_SIZE = 256
DEFAULT_SAMPLE_SIZE = 30


def extract_patches(img, image_path):
    """按文件名选择正确采集面，提取 ROI 并切 patch（无重叠）。"""
    settings = roi_settings_for_path(image_path)
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        result = extract_roi_from_image(
            img,
            rotate_angle=settings["rotate_angle"],
            angle_offset=settings["angle_offset"],
            offset_from_peak=OFFSET_FROM_PEAK,
            roi_width=ROI_WIDTH,
            flip_vertical=settings["flip_vertical"],
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


def infer_patch(model, patch_bgr):
    """对单个 patch 推理，返回异常分数。"""
    pre_processor = getattr(model, "pre_processor", None)
    img_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = to_dtype(to_image(img_rgb), torch.float32, scale=True)
    with torch.no_grad():
        proc_input = img_tensor.unsqueeze(0)
        proc_output = pre_processor(proc_input) if pre_processor is not None else proc_input
        predictions = model.model(proc_output)
    return float(predictions.pred_score)


def evaluate(model, image_paths, sample_size, aggregation="p95"):
    """评估一组图片，返回聚合后的工件级分数。"""
    random.seed(42)
    sampled = random.sample(image_paths, min(sample_size, len(image_paths)))
    scores = []
    for path in sampled:
        img = cv2.imread(path)
        if img is None:
            continue
        patches = extract_patches(img, path)
        if not patches:
            continue
        patch_scores = [infer_patch(model, p) for p in patches]
        scores.append(aggregate_patch_scores(patch_scores, aggregation))
    return scores


def print_stats(label, scores):
    if not scores:
        print(f"  {label}: 无有效数据")
        return
    arr = np.array(scores)
    print(f"  {label:12s} | n={len(arr):3d} | "
          f"min={arr.min():7.4f} | max={arr.max():7.4f} | "
          f"mean={arr.mean():7.4f} | median={np.median(arr):7.4f} | "
          f"std={arr.std():6.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="按文件名采集规则评估 EfficientAD checkpoint")
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(PROJECT_ROOT, "results", "EfficientAd", "My_Metal_Project_roi_v2", "EfficientAd", "metal_roi_v2", "v1", "weights", "lightning", "model.ckpt"),
        help="EfficientAD Lightning .ckpt 路径（CPU 加载，避免旧 .pt 的 MPS 状态问题）",
    )
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--aggregation", choices=("max", "p95", "topk_mean", "mean"), default="p95")
    args = parser.parse_args()
    if args.sample_size < 1:
        parser.error("--sample-size 必须大于等于 1")
    print("=" * 70)
    print("📊 EfficientAd 模型评估：训练集 vs 测试正常 vs NG 异常")
    print("=" * 70)

    # 加载模型
    print("🔄 加载 EfficientAd 模型...")
    # checkpoint 由本地训练脚本产生，含 pathlib 配置对象；显式关闭 PyTorch
    # 2.6+ 的 weights_only 默认限制，避免无法加载可信本地 checkpoint。
    model = EfficientAd.load_from_checkpoint(args.checkpoint, map_location="cpu", weights_only=False)
    model.eval().to("cpu")
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
    print(f"🔧 每组采样: {args.sample_size} 张\n")

    # 评估三组
    print("🔄 正在推理训练集...")
    train_scores = evaluate(model, train_imgs, args.sample_size, args.aggregation)

    print("🔄 正在推理测试正常集...")
    test_scores = evaluate(model, test_normal_imgs, args.sample_size, args.aggregation)

    print("🔄 正在推理 NG 异常集...")
    ng_scores = evaluate(model, ng_imgs, args.sample_size, args.aggregation)

    # 汇总
    print("\n" + "=" * 70)
    print(f"📈 评估结果汇总（工件级聚合: {args.aggregation}）")
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

        # EfficientAd 分数尺度较小，建议用 P99 校准
        normal_all = train_scores + test_scores
        p99_threshold = float(np.percentile(normal_all, 99))

        print(f"  正常样本最高分: {normal_max:.4f}")
        print(f"  NG样本最低分:   {ng_min:.4f}")
        print(f"  NG样本均分:     {ng_mean:.4f}")
        print(f"  正常样本 P99:   {p99_threshold:.4f}（低误报参考，不满足零漏检目标）")
        safety_threshold = ng_min
        safety_normal_ok = sum(1 for s in normal_all if s < safety_threshold)
        print(f"  零漏检上限阈值: {safety_threshold:.4f}（当前 NG 全部拦截）")
        print(f"  该阈值正常放行: {safety_normal_ok}/{len(normal_all)}")

        if ng_min > normal_max:
            print(f"\n  ✅ 模型可以完美区分：正常最高分({normal_max:.4f}) < NG最低分({ng_min:.4f})")
            print(f"     建议阈值: {(normal_max + ng_min) / 2:.4f}")
        else:
            overlap = normal_max - ng_min
            print(f"\n  ⚠️ 存在分数重叠区间: [{ng_min:.4f}, {normal_max:.4f}]")
            print(f"     重叠程度: {overlap:.4f}")
            # 以 P99 为阈值评估准确率
            threshold = p99_threshold
            normal_correct = sum(1 for s in train_scores + test_scores if s < threshold)
            ng_correct = sum(1 for s in ng_scores if s >= threshold)
            total = len(train_scores) + len(test_scores) + len(ng_scores)
            print(f"     以 P99 {threshold:.4f} 为阈值:")
            print(f"       正常检出率: {normal_correct}/{len(train_scores) + len(test_scores)}")
            print(f"       NG检出率:   {ng_correct}/{len(ng_scores)}")
            print(f"       总体准确率: {(normal_correct + ng_correct) / total * 100:.1f}%")

    print("=" * 70)

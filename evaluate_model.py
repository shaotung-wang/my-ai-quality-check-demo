#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型评估工具 - 用于验证缺陷检测的召回率和精确率
"""

import os
import sys
import cv2
import argparse
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
import config


def evaluate_on_folder(model_path, image_folder, conf_threshold=None, verbose=True):
    """
    在文件夹中的所有图像上运行模型评估

    参数:
        model_path: 模型文件路径
        image_folder: 包含测试图像的文件夹
        conf_threshold: 置信度阈值（如果为None使用config中的值）
        verbose: 是否打印详细信息
    """

    if conf_threshold is None:
        conf_threshold = config.CONF_THRESHOLD

    # 加载模型
    print(f"正在加载模型: {model_path}")
    try:
        model = YOLO(model_path)
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return None

    # 获取图像列表
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [
        f for f in os.listdir(image_folder)
        if os.path.splitext(f)[1].lower() in image_extensions
    ]

    if not image_files:
        print(f"✗ 在 {image_folder} 中找不到图像文件")
        return None

    print(f"\n发现 {len(image_files)} 张图像，开始推理...\n")

    # 运行推理
    results_summary = {
        'total': 0,
        'with_defects': 0,  # 模型检测到缺陷
        'without_defects': 0,  # 模型未检测到缺陷
        'high_conf_defects': 0,  # 高置信度缺陷
        'low_conf_detections': 0,  # 低置信度检测
        'detections_by_class': defaultdict(int),
        'files': []
    }

    for idx, image_file in enumerate(image_files, 1):
        image_path = os.path.join(image_folder, image_file)

        try:
            # 读取图像
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"  {idx}. ✗ {image_file} - 无法读取")
                continue

            # 运行推理
            results = model(frame, conf=conf_threshold, verbose=False)

            # 分析检测结果
            boxes = results[0].boxes
            frame_height, frame_width = frame.shape[:2]
            image_area = frame_height * frame_width

            high_conf_count = 0
            low_conf_count = 0
            defect_info = []

            for box in boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = results[0].names[cls_id]

                # 计算框面积
                x1, y1, x2, y2 = box.xyxy[0]
                box_area = float((x2 - x1) * (y2 - y1))
                area_ratio = box_area / image_area

                results_summary['detections_by_class'][class_name] += 1

                if conf >= 0.40:
                    high_conf_count += 1
                    conf_level = "🔴 高"
                elif conf >= config.CONF_THRESHOLD:
                    conf_level = "🟡 中"
                else:
                    low_conf_count += 1
                    conf_level = "🟠 低"

                defect_info.append(f"  - {class_name}: 置信度={conf:.3f} ({conf_level}), 面积比={area_ratio:.4f}")

            results_summary['total'] += 1

            if len(boxes) > 0:
                results_summary['with_defects'] += 1
                results_summary['high_conf_defects'] += high_conf_count
                results_summary['low_conf_detections'] += low_conf_count

                status = f"✓ 🔴 NG (检测到 {len(boxes)} 个缺陷)"
                if verbose:
                    print(f"  {idx}. {image_file}")
                    print(f"     {status}")
                    for info in defect_info:
                        print(info)
            else:
                results_summary['without_defects'] += 1
                status = "✓ 🟢 OK (未检测到缺陷)"
                if verbose:
                    print(f"  {idx}. {image_file}")
                    print(f"     {status}")

            results_summary['files'].append({
                'name': image_file,
                'status': 'NG' if len(boxes) > 0 else 'OK',
                'detection_count': len(boxes),
                'details': defect_info
            })

        except Exception as e:
            print(f"  {idx}. ✗ {image_file} - 处理出错: {e}")

    # 打印统计摘要
    print("\n" + "="*60)
    print("📊 评估结果摘要")
    print("="*60)
    print(f"总图像数: {results_summary['total']}")
    print(f"检测为NG (有缺陷): {results_summary['with_defects']} ({100*results_summary['with_defects']/max(results_summary['total'],1):.1f}%)")
    print(f"检测为OK (无缺陷): {results_summary['without_defects']} ({100*results_summary['without_defects']/max(results_summary['total'],1):.1f}%)")

    if results_summary['with_defects'] > 0:
        print(f"\n  高置信度缺陷 (≥0.40): {results_summary['high_conf_defects']}")
        print(f"  低置信度检测 (<{config.CONF_THRESHOLD}): {results_summary['low_conf_detections']}")

    if results_summary['detections_by_class']:
        print(f"\n🏷  缺陷类型分布:")
        for class_name, count in sorted(results_summary['detections_by_class'].items(), key=lambda x: x[1], reverse=True):
            print(f"  - {class_name}: {count}")

    print("\n" + "="*60)
    print("💡 建议:")
    print("="*60)

    # 基于检测结果的建议
    ng_ratio = results_summary['with_defects'] / max(results_summary['total'], 1)

    if ng_ratio > 0.9:
        print("⚠  注意: 超过90%的图像被检测为有缺陷")
        print("  → 置信度阈值可能过低")
        print(f"  → 建议尝试: CONF_THRESHOLD = {config.CONF_THRESHOLD + 0.05:.2f}")
    elif ng_ratio < 0.1:
        print("ℹ  只有<10%的图像被检测为有缺陷")
        print("  → 可能正常（如果测试集中大多数是完好品）")
        print("  → 或者置信度阈值过高")
        if results_summary['low_conf_detections'] > 0:
            print(f"  → 建议尝试: CONF_THRESHOLD = {config.CONF_THRESHOLD - 0.05:.2f}")
    else:
        print("✓ 检测比率处于合理范围")

    return results_summary


def compare_thresholds(model_path, image_folder, thresholds=None):
    """
    在不同置信度阈值下评估模型

    参数:
        model_path: 模型文件路径
        image_folder: 测试图像文件夹
        thresholds: 要测试的阈值列表
    """

    if thresholds is None:
        thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    print("\n" + "="*70)
    print("🔍 多阈值对比分析")
    print("="*70)
    print(f"{'阈值':<8} {'检测为NG':<12} {'检测为OK':<12} {'高置信度':<12}")
    print("-"*70)

    for threshold in thresholds:
        # 临时替换阈值
        old_threshold = config.CONF_THRESHOLD
        config.CONF_THRESHOLD = threshold

        results = evaluate_on_folder(model_path, image_folder, threshold, verbose=False)

        if results:
            ng_count = results['with_defects']
            ok_count = results['without_defects']
            high_conf = results['high_conf_defects']

            ng_pct = 100 * ng_count / max(results['total'], 1)
            ok_pct = 100 * ok_count / max(results['total'], 1)
            high_conf_pct = 100 * high_conf / max(ng_count, 1) if ng_count > 0 else 0

            print(f"{threshold:<8.2f} {ng_count:<2}({ng_pct:>5.1f}%)  {ok_count:<2}({ok_pct:>5.1f}%)  {high_conf:<2}({high_conf_pct:>5.1f}%)")

        config.CONF_THRESHOLD = old_threshold

    print("-"*70)
    print("建议选择: 在保证召回率的同时，选择最小化误检的阈值")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='评估YOLO模型的缺陷检测性能',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认配置评估模型
  python evaluate_model.py --images ./test_images

  # 使用特定置信度阈值评估
  python evaluate_model.py --images ./test_images --conf 0.15

  # 比较多个阈值
  python evaluate_model.py --images ./test_images --compare-thresholds

  # 使用自定义模型
  python evaluate_model.py --images ./test_images --model ./custom_model.pt
        """
    )

    parser.add_argument('--images', type=str, required=True,
                       help='包含测试图像的文件夹路径')
    parser.add_argument('--model', type=str, default=None,
                       help='模型路径（默认使用config中的MODEL_PATH）')
    parser.add_argument('--conf', type=float, default=None,
                       help='置信度阈值')
    parser.add_argument('--compare-thresholds', action='store_true',
                       help='比较多个置信度阈值的效果')

    args = parser.parse_args()

    # 验证文件夹存在
    if not os.path.isdir(args.images):
        print(f"✗ 错误: {args.images} 不存在或不是文件夹")
        sys.exit(1)

    # 确定模型路径
    model_path = args.model or config.MODEL_PATH
    if not os.path.isfile(model_path):
        print(f"✗ 错误: 模型文件 {model_path} 不存在")
        print(f"  请检查 config.py 中的 MODEL_PATH 设置")
        sys.exit(1)

    # 执行评估
    if args.compare_thresholds:
        compare_thresholds(model_path, args.images)
    else:
        evaluate_on_folder(model_path, args.images, args.conf)


if __name__ == "__main__":
    main()


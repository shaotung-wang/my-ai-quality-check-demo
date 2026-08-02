"""
批量训练数据集构建脚本。

遍历 raw_images 下的各 rod 文件夹，对每张原始图像：
  1. 用共享 lib.roi 按文件名定位高光与 ROI 条带
     （imageXXX 取高光上方，movieXXX_XXX 取高光下方并翻转）
  2. 以 50% 重叠滑窗切出 256×256 的 patch
  3. 输出为 MVTec AD 格式（train/good/）

用法：
    python scripts/train/build_dataset.py

配置：
    修改 main 区块中的路径和参数。
"""

import os
import sys
import glob
import argparse
import cv2

# 确保项目根目录在 sys.path 中，以便导入 lib
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from lib.roi import extract_roi_from_image, roi_settings_for_path


def process_single_image(image_path, offset_from_peak, roi_width, patch_size=256):
    """处理单张训练图片：拉直、寻找水平高光、截取 ROI 条带。

    ROI 方向由文件名决定，不能因为该图片用于训练就一律取高光上方。
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    settings = roi_settings_for_path(image_path)
    result = extract_roi_from_image(
        img,
        rotate_angle=settings["rotate_angle"],
        angle_offset=settings["angle_offset"],
        offset_from_peak=offset_from_peak,
        roi_width=roi_width,
        flip_vertical=settings["flip_vertical"],
        verbose=False,
    )
    if result is None:
        return None

    return result["roi_strip"]


def build_dataset(input_dir, output_dir, offset_from_peak, roi_width,
                  patch_size=256, sample_every=10):
    """遍历 input_dir 下的子文件夹，批量构建 MVTec AD 格式训练集。

    参数：
        input_dir: 原始图片根目录（内含 rod1, rod2, ... 子文件夹）
        output_dir: 输出数据集根目录（将自动创建 train/good/）
        offset_from_peak: ROI 距高光峰值的偏移像素
        roi_width: ROI 条带高度
        patch_size: 输出 patch 尺寸
        sample_every: 每隔多少张原图采一张；1 表示使用全部图片
    """
    train_good_dir = os.path.join(output_dir, "train", "good")
    os.makedirs(train_good_dir, exist_ok=True)

    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG')

    print("=" * 50)
    print(f"📂 正在扫描根目录: {input_dir}")
    print(f"⚙️ ROI 规则: imageXXX=高光上方、固定旋转; movieXXX_XXX=高光下方、翻转后自动旋转")
    print(f"⚙️ 固定参数: 偏移={offset_from_peak}px, 裁剪高度={roi_width}px")
    print("=" * 50)

    subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    if not subdirs:
        print(f"❌ 未在 {input_dir} 下找到任何工件文件夹，请检查路径。")
        return

    total_images_processed = 0
    total_patches_generated = 0
    source_counts = {"image": 0, "movie": 0, "unknown": 0}

    for subdir in sorted(subdirs):
        subdir_path = os.path.join(input_dir, subdir)
        raw_images = []
        for ext in extensions:
            raw_images.extend(glob.glob(os.path.join(subdir_path, ext)))

        if not raw_images:
            continue

        print(f"📦 正在处理工件群组: {subdir} (共 {len(raw_images)} 张原图)")

        for img_idx, img_path in enumerate(raw_images):
            # 通过抽帧控制训练数据规模。
            # PaDiM 的 memory bank 存储所有样本特征，256x256 输入下每个样本约 1.6MB
            # 6000 样本 ≈ 10GB，适配 24GB 系统内存
            if img_idx % sample_every != 0:
                continue
            source_type = roi_settings_for_path(img_path)["source_type"]
            roi_strip = process_single_image(img_path, offset_from_peak, roi_width, patch_size)
            if roi_strip is None:
                print(f"  ⚠️ 警告: 图片 {os.path.basename(img_path)} 无法定位有效区域，已跳过。")
                continue

            # 从左往右滑窗（无重叠，减少 patch 总量以控制 PaDiM 训练内存）
            window_w = roi_strip.shape[0]
            stride = window_w
            x_start = 0
            patch_count = 0
            base_filename = os.path.splitext(os.path.basename(img_path))[0]

            while (x_start + window_w) <= roi_strip.shape[1]:
                square_patch = roi_strip[:, x_start:x_start + window_w]
                resized_patch = cv2.resize(square_patch, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)
                save_name = f"{subdir}_{base_filename}_patch_{patch_count:02d}.jpg"
                save_path = os.path.join(train_good_dir, save_name)
                cv2.imwrite(save_path, resized_patch)

                x_start += stride
                patch_count += 1
                total_patches_generated += 1

            total_images_processed += 1
            source_counts[source_type] += 1

    print("=" * 50)
    print(f"🎉 流水线批量执行完毕！")
    print(f"📊 成功处理原图总数: {total_images_processed} 张")
    print(f"   └─ imageXXX（高光上方）: {source_counts['image']} 张")
    print(f"   └─ movieXXX_XXX（高光下方）: {source_counts['movie']} 张")
    if source_counts["unknown"]:
        print(f"   └─ 未识别命名（按旧检测规则处理）: {source_counts['unknown']} 张")
    print(f"🖼️ 最终生成 {patch_size}×{patch_size} 样本总数: {total_patches_generated} 张")
    print(f"📁 MVTec AD 训练集就绪: {os.path.abspath(train_good_dir)}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="按采集视角构建 EfficientAD/PaDiM 训练 patch 数据集")
    parser.add_argument("--input-dir", default="data/train")
    parser.add_argument("--output-dir", default="outputs/train_intermediate/padim")
    parser.add_argument("--offset-from-peak", type=int, default=33)
    parser.add_argument("--roi-width", type=int, default=55)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--sample-every", type=int, default=10)
    args = parser.parse_args()
    if args.sample_every < 1:
        parser.error("--sample-every 必须大于等于 1")
    build_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        offset_from_peak=args.offset_from_peak,
        roi_width=args.roi_width,
        patch_size=args.patch_size,
        sample_every=args.sample_every,
    )

"""
批量训练数据集构建脚本。

遍历 raw_images 下的各 rod 文件夹，对每张原始图像：
  1. 用共享 lib.roi 自动定位高光与 ROI 条带（训练模式：高光上方）
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
import cv2

# 确保项目根目录在 sys.path 中，以便导入 lib
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from lib.roi import extract_roi_from_image


def process_single_image(image_path, rotate_angle, offset_from_peak, roi_width, patch_size=256):
    """处理单张训练图片：拉直、寻找水平高光、截取 ROI 条带。

    训练模式下 flip_vertical=False，ROI 位于高光上方。
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    result = extract_roi_from_image(
        img,
        rotate_angle=rotate_angle,
        angle_offset=0.0,
        offset_from_peak=offset_from_peak,
        roi_width=roi_width,
        flip_vertical=False,  # 训练图：ROI 在高光上方
        verbose=False,
    )
    if result is None:
        return None

    return result["roi_strip"]


def build_dataset(input_dir, output_dir, rotate_angle, offset_from_peak, roi_width,
                  patch_size=256):
    """遍历 input_dir 下的子文件夹，批量构建 MVTec AD 格式训练集。

    参数：
        input_dir: 原始图片根目录（内含 rod1, rod2, ... 子文件夹）
        output_dir: 输出数据集根目录（将自动创建 train/good/）
        rotate_angle: 手动指定拉直角度（None 则自动搜索）
        offset_from_peak: ROI 距高光峰值的偏移像素
        roi_width: ROI 条带高度
        patch_size: 输出 patch 尺寸
    """
    train_good_dir = os.path.join(output_dir, "train", "good")
    os.makedirs(train_good_dir, exist_ok=True)

    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG')

    print("=" * 50)
    print(f"📂 正在扫描根目录: {input_dir}")
    print(f"⚙️ 固定的黄金参数: 角度={rotate_angle}°, 向上偏移={offset_from_peak}px, 裁剪宽度={roi_width}px")
    print("=" * 50)

    subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    if not subdirs:
        print(f"❌ 未在 {input_dir} 下找到任何工件文件夹，请检查路径。")
        return

    total_images_processed = 0
    total_patches_generated = 0

    for subdir in sorted(subdirs):
        subdir_path = os.path.join(input_dir, subdir)
        raw_images = []
        for ext in extensions:
            raw_images.extend(glob.glob(os.path.join(subdir_path, ext)))

        if not raw_images:
            continue

        print(f"📦 正在处理工件群组: {subdir} (共 {len(raw_images)} 张原图)")

        for img_path in raw_images:
            roi_strip = process_single_image(img_path, rotate_angle, offset_from_peak, roi_width, patch_size)
            if roi_strip is None:
                print(f"  ⚠️ 警告: 图片 {os.path.basename(img_path)} 无法定位有效区域，已跳过。")
                continue

            # 从左往右滑窗
            window_w = roi_strip.shape[0]
            stride = int(window_w * 0.5)
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

    print("=" * 50)
    print(f"🎉 流水线批量执行完毕！")
    print(f"📊 成功处理原图总数: {total_images_processed} 张")
    print(f"🖼️ 最终生成 {patch_size}×{patch_size} 样本总数: {total_patches_generated} 张")
    print(f"📁 MVTec AD 训练集就绪: {os.path.abspath(train_good_dir)}")
    print("=" * 50)


if __name__ == "__main__":
    build_dataset(
        input_dir="data/raw/normal",
        output_dir="data/datasets/padim",
        rotate_angle=-2.0,
        offset_from_peak=33,
        roi_width=55,
        patch_size=256,
    )
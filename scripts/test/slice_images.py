"""
待检测图切片脚本。

将待检测图片按训练时的同样规则切为 256×256 的 patch。
对待检测图自动搜索高光、拉直并截取 ROI 条带（flip_vertical=True 以对齐训练集视角）。

用法：
    # 单张模式
    python scripts/test/slice_images.py

    # 批量模式
    python scripts/test/slice_images.py --batch

输出目录结构：
    outputs/patches/{image_name}/
        patches/         # 每张 patch_XX.jpg
        preview.jpg      # 在原图上画出所有 patch 位置的可视化
        roi_strip.jpg    # 提取出来的 ROI 条带
"""

import os
import sys
import glob
import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from lib.roi import extract_roi_from_image, manual_deskew_image


def slice_test_image(image_path,
                     output_dir,
                     rotate_angle=None,
                     angle_offset=0.0,
                     offset_from_peak=33,
                     roi_width=55,
                     patch_size=256,
                     angle_search_range=(-8.0, 8.0),
                     angle_search_step=0.2,
                     save_preview=True,
                     flip_vertical=True):
    """将单张待检测图按训练时的同样规则切成 256×256 的 patch。

    对待检测图做上下翻转（flip_vertical=True），使原图中位于高光下方的清晰区域
    在翻转后落入高光上方，与训练集视角保持一致。

    输出目录结构：
      output_dir/
        patches/         # 每张 patch_XX.jpg
        preview.jpg      # 切片位置可视化
        roi_strip.jpg    # ROI 条带本体
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取图片: {image_path}")
        return []

    print("=" * 50)
    print(f"🔪 切片待检测图: {image_path}")
    direction_label = "上方" if flip_vertical else "下方"
    print(f"⚙️ 黄金参数: 向{direction_label}偏移={offset_from_peak}px, 裁剪高度={roi_width}px, 翻转={flip_vertical}")
    print("=" * 50)

    result = extract_roi_from_image(
        img,
        rotate_angle=rotate_angle,
        angle_offset=angle_offset,
        offset_from_peak=offset_from_peak,
        roi_width=roi_width,
        angle_search_range=angle_search_range,
        angle_search_step=angle_search_step,
        flip_vertical=flip_vertical,
    )
    if result is None:
        print("❌ 无法定位有效 ROI 条带")
        return []

    roi_strip = result["roi_strip"]
    y_start = result["y_start"]
    y_end = result["y_end"]
    peak_y = result["peak_y"]
    used_angle = result["rotate_angle"]

    patches_dir = os.path.join(output_dir, "patches")
    os.makedirs(patches_dir, exist_ok=True)

    # 滑窗逻辑与训练集 builder 完全一致
    window_w = roi_strip.shape[0]
    stride = int(window_w * 0.5)
    x_start = 0
    patch_count = 0
    patch_records = []  # (save_path, x_start_in_strip)

    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    while (x_start + window_w) <= roi_strip.shape[1]:
        square_patch = roi_strip[:, x_start:x_start + window_w]
        resized_patch = cv2.resize(
            square_patch, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC
        )

        save_name = f"{base_filename}_patch_{patch_count:02d}.jpg"
        save_path = os.path.join(patches_dir, save_name)
        cv2.imwrite(save_path, resized_patch)

        patch_records.append((save_path, x_start))
        x_start += stride
        patch_count += 1

    print(f"✅ 共生成 {patch_count} 个 patch，已保存至: {os.path.abspath(patches_dir)}")

    # 可选：保存预览
    if save_preview:
        cv2.imwrite(os.path.join(output_dir, "roi_strip.jpg"), roi_strip)

        preview = manual_deskew_image(img, angle_deg=used_angle)
        cv2.rectangle(preview, (0, y_start), (preview.shape[1] - 1, y_end), (0, 255, 0), 2)
        cv2.line(preview, (0, peak_y), (preview.shape[1] - 1, peak_y), (0, 0, 255), 1)
        for _, x in patch_records:
            cv2.rectangle(preview, (x, y_start), (x + window_w, y_end), (0, 0, 255), 1)
        cv2.imwrite(os.path.join(output_dir, "preview.jpg"), preview)
        print(f"🖼️ 预览图与 ROI 条带已保存至: {os.path.abspath(output_dir)}")

    return [p for p, _ in patch_records]


def batch_slice(input_dir, output_dir, **kwargs):
    """对一个目录下所有待检测图批量切片，每张图独立一个子目录。"""
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG')
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(input_dir, ext)))

    if not images:
        print(f"❌ 目录 {input_dir} 下没有可处理的图片")
        return

    for img_path in sorted(images):
        sub_out = os.path.join(output_dir, os.path.splitext(os.path.basename(img_path))[0])
        slice_test_image(img_path, sub_out, **kwargs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="待检测图切片工具")
    parser.add_argument("--batch", action="store_true", help="批量模式：处理整个目录")
    parser.add_argument("--input", type=str, default="data/test/images",
                        help="输入图片或目录（默认: data/test/images）")
    parser.add_argument("--output", type=str, default="outputs/patches",
                        help="输出目录（默认: outputs/patches）")
    args = parser.parse_args()

    kwargs = dict(
        rotate_angle=None,
        angle_offset=2.8,
        offset_from_peak=33,
        roi_width=55,
        patch_size=256,
        save_preview=True,
        flip_vertical=True,
    )

    if args.batch:
        batch_slice(args.input, args.output, **kwargs)
    else:
        # 单张模式：取目录下第一张图
        extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG')
        images = []
        for ext in extensions:
            images.extend(glob.glob(os.path.join(args.input, ext)))
        if not images:
            print(f"❌ 目录 {args.input} 下没有可处理的图片")
            sys.exit(1)

        test_image = sorted(images)[0]
        sub_out = os.path.join(args.output, os.path.splitext(os.path.basename(test_image))[0])
        slice_test_image(test_image, sub_out, **kwargs)
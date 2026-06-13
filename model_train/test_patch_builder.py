import cv2
import numpy as np
import os
import glob
from scipy.ndimage import gaussian_filter1d


def extract_roi_strip(img, offset_from_peak, roi_width):
    """寻找水平高光 + 截取高光上方的 ROI 条带"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    crop_gray = gray[:, int(w * 0.2):int(w * 0.8)]
    row_mean = np.mean(crop_gray, axis=1)
    row_mean_smooth = gaussian_filter1d(row_mean, sigma=5)
    peak_y = int(np.argmax(row_mean_smooth))

    y_end = peak_y - offset_from_peak
    y_start = y_end - roi_width

    if y_start < 0:
        y_start = 0
    if y_end <= y_start:
        return None, None

    roi_strip = img[y_start:y_end, :]
    return roi_strip, (y_start, y_end)


def slice_test_image(image_path, output_dir, offset_from_peak, roi_width,
                     patch_size=256, save_preview=True):
    """
    将单张待检测图按训练时的同样规则切成 256x256 的 patch。
    输出目录结构：
      output_dir/
        patches/         # 每张 patch_XX.jpg（喂给推理器的输入）
        preview.jpg      # 在原图上画出所有 patch 位置的可视化（可选）
        roi_strip.jpg    # 提取出来的 ROI 条带本体（可选）
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取图片: {image_path}")
        return []

    print("=" * 50)
    print(f"🔪 切片待检测图: {image_path}")
    print(f"⚙️ 黄金参数: 向上偏移={offset_from_peak}px, 裁剪宽度={roi_width}px")
    print("=" * 50)

    roi_strip, y_range = extract_roi_strip(img, offset_from_peak, roi_width)
    if roi_strip is None:
        print("❌ 无法定位有效 ROI 条带（高光过于靠近图像顶部）")
        return []

    patches_dir = os.path.join(output_dir, "patches")
    os.makedirs(patches_dir, exist_ok=True)

    # 滑窗逻辑与训练集 builder 完全一致：以条带高度作为正方形边长，50% 重叠
    window_w = roi_strip.shape[0]
    stride = int(window_w * 0.5)
    x_start = 0
    patch_count = 0
    patch_records = []  # (save_path, x_start_in_strip)

    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    while (x_start + window_w) <= roi_strip.shape[1]:
        square_patch = roi_strip[:, x_start:x_start + window_w]
        resized_patch = cv2.resize(square_patch, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)

        save_name = f"{base_filename}_patch_{patch_count:02d}.jpg"
        save_path = os.path.join(patches_dir, save_name)
        cv2.imwrite(save_path, resized_patch)

        patch_records.append((save_path, x_start))
        x_start += stride
        patch_count += 1

    print(f"✅ 共生成 {patch_count} 个 patch，已保存至: {os.path.abspath(patches_dir)}")

    # 可选：保存 ROI 条带本体与原图上的切片位置预览
    if save_preview:
        cv2.imwrite(os.path.join(output_dir, "roi_strip.jpg"), roi_strip)

        preview = img.copy()
        y_start, y_end = y_range
        # 整条 ROI 用绿色框出
        cv2.rectangle(preview, (0, y_start), (preview.shape[1] - 1, y_end), (0, 255, 0), 2)
        # 每个 patch 用红色框
        for _, x in patch_records:
            cv2.rectangle(preview, (x, y_start), (x + window_w, y_end), (0, 0, 255), 1)
        cv2.imwrite(os.path.join(output_dir, "preview.jpg"), preview)
        print(f"🖼️ 预览图与 ROI 条带已保存至: {os.path.abspath(output_dir)}")

    return [p for p, _ in patch_records]


def batch_slice_test_images(input_dir, output_dir, offset_from_peak, roi_width,
                            patch_size=256, save_preview=True):
    """对一个目录下所有待检测图批量切片，每张图独立一个子目录"""
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG')
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(input_dir, ext)))

    if not images:
        print(f"❌ 目录 {input_dir} 下没有可处理的图片")
        return

    for img_path in sorted(images):
        sub_out = os.path.join(output_dir, os.path.splitext(os.path.basename(img_path))[0])
        slice_test_image(img_path, sub_out, offset_from_peak, roi_width,
                         patch_size=patch_size, save_preview=save_preview)


if __name__ == "__main__":
    # ---- 单张待检测图模式 ----
    TEST_IMAGE_PATH = "../ng/ng_test.jpg"
    OUTPUT_DIR = "../ng/ng_test_patches"

    slice_test_image(
        image_path=TEST_IMAGE_PATH,
        output_dir=OUTPUT_DIR,
        offset_from_peak=33,
        roi_width=55,
        patch_size=256,
        save_preview=True,
    )

    # ---- 批量模式：取消下面注释即可对一个文件夹下所有图切片 ----
    # batch_slice_test_images(
    #     input_dir="./ng",
    #     output_dir="./ng/_sliced",
    #     offset_from_peak=33,
    #     roi_width=55,
    #     patch_size=256,
    # )

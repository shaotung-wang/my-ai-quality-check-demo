import cv2
import os
import glob
import sys

# 确保无论从哪个工作目录运行，都能导入同目录下的 find_highlight_test
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import find_highlight_test


def manual_deskew_image(img, angle_deg):
    """固定角度纠偏：正角度为 OpenCV 默认逆时针方向"""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )


def extract_roi_strip(img,
                      rotate_angle=None,
                      angle_offset=0.0,
                      offset_from_peak=33,
                      roi_width=55,
                      angle_search_range=(-8.0, 8.0),
                      angle_search_step=0.2,
                      flip_vertical=True):
    """
    寻找水平高光并截取 ROI 条带。

    本函数直接复用 find_highlight_test.extract_roi_from_image，确保高光搜索、
    角度校准和 ROI 定位逻辑只维护一份。

    参数：
      rotate_angle: 手动指定拉直角度；None 则自动搜索。
      angle_offset: 未翻转时正数表示顺时针；翻转后正数表示逆时针。
      offset_from_peak: ROI 起始行与高光峰值行的垂直距离。
      roi_width: ROI 条带高度。
      flip_vertical: 是否先上下翻转图像，使清晰区位于高光上方。
    返回：
      roi_strip, (y_start, y_end, peak_y, used_angle)
    """
    result = find_highlight_test.extract_roi_from_image(
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
        return None, None

    y_start = result["y_start"]
    y_end = result["y_end"]
    peak_y = result["peak_y"]
    used_angle = result["rotate_angle"]
    roi_strip = result["roi_strip"]
    return roi_strip, (y_start, y_end, peak_y, used_angle)


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
    """
    将单张待检测图按训练时的同样规则切成 256x256 的 patch。

    说明：
      若 flip_vertical=True（默认），会先把待检测图上下翻转，使原图中位于
      高光下方的清晰区域，在翻转后落入高光上方，与训练集视角保持一致。

    输出目录结构：
      output_dir/
        patches/         # 每张 patch_XX.jpg（喂给推理器的输入）
        preview.jpg      # 在翻转后的图上画出所有 patch 位置的可视化（可选）
        roi_strip.jpg    # 提取出来的 ROI 条带本体（可选）
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

    roi_strip, meta = extract_roi_strip(
        img,
        rotate_angle=rotate_angle,
        angle_offset=angle_offset,
        offset_from_peak=offset_from_peak,
        roi_width=roi_width,
        angle_search_range=angle_search_range,
        angle_search_step=angle_search_step,
        flip_vertical=flip_vertical,
    )
    if roi_strip is None:
        print("❌ 无法定位有效 ROI 条带")
        return []

    y_start, y_end, peak_y, used_angle = meta
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

    # 可选：保存 ROI 条带本体与原图上的切片位置预览
    if save_preview:
        cv2.imwrite(os.path.join(output_dir, "roi_strip.jpg"), roi_strip)

        preview = manual_deskew_image(img, angle_deg=used_angle)
        # 整条 ROI 用绿色框出
        cv2.rectangle(preview, (0, y_start), (preview.shape[1] - 1, y_end), (0, 255, 0), 2)
        # 高光峰值用红色虚线示意（这里画一条实线）
        cv2.line(preview, (0, peak_y), (preview.shape[1] - 1, peak_y), (0, 0, 255), 1)
        # 每个 patch 用红色框
        for _, x in patch_records:
            cv2.rectangle(preview, (x, y_start), (x + window_w, y_end), (0, 0, 255), 1)
        cv2.imwrite(os.path.join(output_dir, "preview.jpg"), preview)
        print(f"🖼️ 预览图与 ROI 条带已保存至: {os.path.abspath(output_dir)}")

    return [p for p, _ in patch_records]


def batch_slice_test_images(input_dir,
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
        slice_test_image(
            img_path,
            sub_out,
            rotate_angle=rotate_angle,
            angle_offset=angle_offset,
            offset_from_peak=offset_from_peak,
            roi_width=roi_width,
            patch_size=patch_size,
            angle_search_range=angle_search_range,
            angle_search_step=angle_search_step,
            save_preview=save_preview,
            flip_vertical=flip_vertical,
        )


if __name__ == "__main__":
    # ---- 单张待检测图模式 ----
    TEST_IMAGE_PATH = "./all_frames/movie270_00000000.png"
    OUTPUT_DIR = "./test_patches/movie270_00000000"

    slice_test_image(
        image_path=TEST_IMAGE_PATH,
        output_dir=OUTPUT_DIR,
        rotate_angle=None,      # None = 自动搜索最佳角度
        angle_offset=2.8,       # flip_vertical=True 时：正数 = 逆时针再转几度
        offset_from_peak=33,    # 距离高光峰值多少像素开始截取目标区域
        roi_width=55,           # 提取的条带高度
        patch_size=256,
        save_preview=True,
        flip_vertical=True,     # 默认 True：上下翻转以对齐训练集（高光上方为清晰区）
    )

    # ---- 批量模式：取消下面注释即可对一个文件夹下所有图切片 ----
    # batch_slice_test_images(
    #     input_dir="./all_frames",
    #     output_dir="./test_patches/_all",
    #     rotate_angle=None,
    #     angle_offset=2.8,
    #     offset_from_peak=33,
    #     roi_width=55,
    #     patch_size=256,
    #     flip_vertical=True,
    # )

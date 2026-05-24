import cv2
import numpy as np
import os
import glob


def manual_deskew_image(img, angle_deg):
    """固定角度纠偏模块（顺时针旋转 2 度）"""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def process_single_image(image_path, rotate_angle, offset_from_peak, roi_width, patch_size=256):
    """处理单张图片：拉直、寻找水平高光、截取【上方】区域"""
    img = cv2.imread(image_path)
    if img is None:
        return None

    # 1. 旋转拉直
    straight_img = manual_deskew_image(img, angle_deg=rotate_angle)

    # 2. 按行求平均，寻找水平高光中心 Y 坐标 (Peak Y)
    gray = cv2.cvtColor(straight_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    crop_gray = gray[:, int(w * 0.2):int(w * 0.8)]  # 避开左右暗角
    row_mean = np.mean(crop_gray, axis=1)

    from scipy.ndimage import gaussian_filter1d
    row_mean_smooth = gaussian_filter1d(row_mean, sigma=5)
    peak_y = int(np.argmax(row_mean_smooth))

    # 3. 计算裁剪的上下边界（严格采用你调通的【高光上方】反向递减逻辑）
    y_end = peak_y - offset_from_peak
    y_start = y_end - roi_width

    # 边界安全保护
    if y_start < 0:
        y_start = 0
    if y_end <= y_start:
        return None

    # 执行裁剪
    return straight_img[y_start:y_end, :]


def batch_build_efficientad_dataset(base_input_dir, base_output_dir, rotate_angle, offset_from_peak, roi_width,
                                    patch_size=256):
    """
    遍历 rod1-rod7，批量切割并组织为 EfficientAD 训练目录结构
    """
    # 自动创建标准 MVTec AD 训练目录：train/good
    train_good_dir = os.path.join(base_output_dir, "train", "good")
    os.makedirs(train_good_dir, exist_ok=True)

    # 支持的图片格式扩展名
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG')

    print("=" * 50)
    print(f"📂 正在扫描根目录: {base_input_dir}")
    print(f"⚙️ 固定的黄金参数: 角度={rotate_angle}°, 向上偏移={offset_from_peak}px, 裁剪宽度={roi_width}px")
    print("=" * 50)

    # 获取根目录下所有的子文件夹（rod1, rod2 ...）
    subdirs = [d for d in os.listdir(base_input_dir) if os.path.isdir(os.path.join(base_input_dir, d))]

    if not subdirs:
        print(f"❌ 未在 {base_input_dir} 下找到任何工件文件夹，请检查路径。")
        return

    total_images_processed = 0
    total_patches_generated = 0

    # 遍历每一个轴的文件夹
    for subdir in sorted(subdirs):
        subdir_path = os.path.join(base_input_dir, subdir)

        # 搜集该文件夹下的所有图片
        raw_images = []
        for ext in extensions:
            raw_images.extend(glob.glob(os.path.join(subdir_path, ext)))

        if not raw_images:
            continue

        print(f"📦 正在处理工件群组: {subdir} (共 {len(raw_images)} 张原图)")

        # 处理该轴下的每一张照片
        for img_path in raw_images:
            roi_strip = process_single_image(img_path, rotate_angle, offset_from_peak, roi_width, patch_size)
            if roi_strip is None:
                print(f"  ⚠️ 警告: 图片 {os.path.basename(img_path)} 无法定位有效区域，已跳过。")
                continue

            # 从左往右的滑窗逻辑
            window_w = roi_strip.shape[0]  # 高度作为正方形边长 (55px)
            stride = int(window_w * 0.5)  # 步长设为 50% 重叠率，增加批次间特征稠密度
            x_start = 0
            patch_count = 0

            base_filename = os.path.splitext(os.path.basename(img_path))[0]

            while (x_start + window_w) <= roi_strip.shape[1]:
                square_patch = roi_strip[:, x_start:x_start + window_w]

                # 缩放到 EfficientAD 强限制的 256x256
                resized_patch = cv2.resize(square_patch, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)

                # 核心：将子目录名 (如 rod1) 融入文件名，防止重名覆盖
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
    print(f"🖼️ 最终生成 256x256 样本总数: {total_patches_generated} 张")
    print(f"📁 EfficientAD 训练集就绪: {os.path.abspath(train_good_dir)}")
    print("=" * 50)


if __name__ == "__main__":
    # 配置你的实际路径
    INPUT_NORMAL_DIR = "./normal"  # 包含 rod1-rod7 的父目录
    OUTPUT_DATASET_DIR = "./My_Metal_Project"  # 模型训练的目标根目录

    # 传入你验证完全正确的黄金参数
    batch_build_efficientad_dataset(
        base_input_dir=INPUT_NORMAL_DIR,
        base_output_dir=OUTPUT_DATASET_DIR,
        rotate_angle=-2.0,
        offset_from_peak=33,
        roi_width=55,
        patch_size=256
    )
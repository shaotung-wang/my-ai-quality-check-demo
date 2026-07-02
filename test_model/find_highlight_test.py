import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def manual_deskew_image(img, angle_deg):
    """手动固定角度纠偏（与 find_highlight.py 完全一致）"""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def _row_brightness(img, sigma=5):
    """对一张图按行计算平滑后的亮度投影（中间 60% 宽度，避开暗角）"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, w = gray.shape
    crop = gray[:, int(w * 0.2):int(w * 0.8)]
    row_mean = np.mean(crop, axis=1)
    return gaussian_filter1d(row_mean, sigma=sigma)


def _peak_sharpness(row_mean_smooth):
    """
    评价高光峰值的"锐度"：峰值 - 背景均值，再除以背景标准差。
    高光越是被拉直成一条水平线，峰值越尖锐、得分越高。
    """
    peak = float(row_mean_smooth.max())
    bg_mean = float(row_mean_smooth.mean())
    bg_std = float(row_mean_smooth.std()) + 1e-6
    return (peak - bg_mean) / bg_std


def auto_search_rotate_angle(img, angle_range=(-8.0, 8.0), step=0.2):
    """
    在 [angle_range[0], angle_range[1]] 内以 step° 步长扫描，
    返回让水平高光峰值最锐的角度。
    """
    angles = np.arange(angle_range[0], angle_range[1] + step / 2, step)
    scores = []
    for a in angles:
        rotated = manual_deskew_image(img, angle_deg=float(a))
        rm = _row_brightness(rotated)
        scores.append(_peak_sharpness(rm))
    scores = np.array(scores)
    best_idx = int(np.argmax(scores))
    return float(angles[best_idx]), angles, scores


def extract_roi_from_image(img,
                           rotate_angle=None,
                           angle_offset=0.0,
                           offset_from_peak=33,
                           roi_width=55,
                           angle_search_range=(-8.0, 8.0),
                           angle_search_step=0.2,
                           flip_vertical=True):
    """
    对输入图像自动搜索高光、拉直并截取 ROI 条带（无可视化）。

    参数与 find_highlight_for_test_image 保持一致。
    返回 dict 包含 roi_strip 及定位元信息；若 ROI 无效则返回 None。
    """
    # 关键修正：对待检测图做垂直翻转，对齐训练照片（高光上方为清晰区）
    if flip_vertical:
        img = cv2.flip(img, 0)
        print("🔄 已对待检测图做上下翻转，以匹配训练集视角（高光上方为清晰区）")

    # 1. 自动搜索旋转角（除非用户已手动指定）
    if rotate_angle is None:
        best_angle, angles, scores = auto_search_rotate_angle(
            img, angle_range=angle_search_range, step=angle_search_step
        )
        print(f"🔍 自动搜索得到最佳旋转角度: {best_angle:+.2f}°")
    else:
        best_angle = float(rotate_angle)
        angles, scores = None, None
        print(f"📌 使用手动指定的旋转角度: {best_angle:+.2f}°")

    # 1.5 叠加人工校准偏移
    if angle_offset != 0.0:
        if flip_vertical:
            # 翻转后坐标系上下镜像：正 angle_offset 表示逆时针
            best_angle += float(angle_offset)
            direction = "逆时针"
        else:
            # 未翻转：正 angle_offset 表示顺时针
            best_angle -= float(angle_offset)
            direction = "顺时针"
        print(f"🔄 叠加校准偏移: {angle_offset:+.2f}°（{direction}），实际使用角度: {best_angle:+.2f}°")

    # 2. 用最佳角度拉直
    straight_img = manual_deskew_image(img, angle_deg=best_angle)

    # 3. 找高光峰值 Y
    row_mean_smooth = _row_brightness(straight_img)
    peak_y = int(np.argmax(row_mean_smooth))

    # 4. 计算 ROI 条带
    if flip_vertical:
        # 翻转后清晰区位于高光上方，与训练集一致
        y_end = peak_y - offset_from_peak
        y_start = y_end - roi_width
        img_h = straight_img.shape[0]
        if y_start < 0:
            print(f"⚠️ 警告: ROI 已撞到图像顶部，自动截断到 y=0")
            y_start = 0
        if y_end <= y_start:
            print("❌ ROI 高度无效，无法定位有效区域")
            return None
    else:
        # 未翻转：清晰区位于高光下方
        y_start = peak_y + offset_from_peak
        y_end = y_start + roi_width
        img_h = straight_img.shape[0]
        if y_end > img_h:
            print(f"⚠️ 警告: ROI 已撞到图像底部，自动截断到 y={img_h}")
            y_end = img_h
        if y_end <= y_start:
            print("❌ ROI 高度无效，无法定位有效区域")
            return None

    direction_label = "上方" if flip_vertical else "下方"
    print(f"✅ 高光峰值行: Y={peak_y}    ROI 条带（高光{direction_label}）: Y=[{y_start}:{y_end}] (高度 {y_end - y_start}px)")
    print("=" * 60)
    print("👉 把下面这组参数直接喂给 test_patch_builder.py:")
    print(f"     rotate_angle    = {best_angle:+.2f}")
    print(f"     offset_from_peak = {offset_from_peak}")
    print(f"     roi_width       = {roi_width}")
    print("=" * 60)

    roi_strip = straight_img[y_start:y_end, :]

    return {
        "rotate_angle": best_angle,
        "peak_y": peak_y,
        "y_start": y_start,
        "y_end": y_end,
        "offset_from_peak": offset_from_peak,
        "roi_width": roi_width,
        "roi_strip": roi_strip,
        "straight_img": straight_img,
        "processed_img": img,
        "row_mean_smooth": row_mean_smooth,
        "angles": angles,
        "scores": scores,
    }


def find_highlight_for_test_image(image_path,
                                  rotate_angle=None,
                                  angle_offset=0.0,
                                  offset_from_peak=33,
                                  roi_width=55,
                                  angle_search_range=(-8.0, 8.0),
                                  angle_search_step=0.2,
                                  flip_vertical=True):
    """
    针对待检测工件图，自动搜索高光所在的旋转角，并定位高光下方的 ROI 条带。

    说明：
      若 flip_vertical=True（默认），会先把待检测图上下翻转，使原图中位于
      高光下方的清晰区域，在翻转后落入高光上方，与训练集视角保持一致。
      可视化将基于翻转后的图像进行展示。

    可视化输出：
      左上：翻转后的原图（标出检测到的高光中心线倾斜方向）
      右上：拉直后图像，红线为高光峰值 Y，绿色带为 ROI 条带
      左下：角度-峰值锐度曲线（角度搜索过程）
      右下：拉直后亮度的 1D 行投影
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取图片: {image_path}")
        return None

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
        return None

    best_angle = result["rotate_angle"]
    peak_y = result["peak_y"]
    y_start = result["y_start"]
    y_end = result["y_end"]
    offset_from_peak = result["offset_from_peak"]
    roi_width = result["roi_width"]
    straight_img = result["straight_img"]
    img = result["processed_img"]
    row_mean_smooth = result["row_mean_smooth"]
    angles = result["angles"]
    scores = result["scores"]

    # 5. 可视化
    if angles is not None:
        _, axes = plt.subplots(2, 2, figsize=(14, 10))
    else:
        _, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes = np.array([axes, [None, None]])  # 让索引兼容

    # 左上：翻转后的原图 + 倾斜方向示意
    ax = axes[0, 0]
    title_prefix = "Flipped + " if flip_vertical else ""
    ax.set_title(f"{title_prefix}Original (detected tilt: {best_angle:+.2f}°)")
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    h_img, w_img = img.shape[:2]
    cx, cy = w_img / 2, h_img / 2
    dx = w_img * 0.45
    dy = dx * np.tan(np.deg2rad(-best_angle))  # 反向：图像顺时针倾斜时检测到的角度是负的
    ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], 'r--', linewidth=1.5,
            label='Detected highlight direction')
    ax.legend(loc='lower left')

    # 右上：拉直后 + ROI
    ax = axes[0, 1]
    ax.set_title("Deskewed + Auto-Located ROI")
    ax.imshow(cv2.cvtColor(straight_img, cv2.COLOR_BGR2RGB))
    ax.axhline(y=peak_y, color='red', linestyle='--', label='Highlight Peak (Y)')
    ax.axhspan(ymin=y_start, ymax=y_end, color='green', alpha=0.3,
               label=f'Target ROI (Height: {y_end - y_start}px)')
    ax.legend(loc='lower left')

    # 下排两图仅在自动搜索时显示
    if angles is not None:
        ax = axes[1, 0]
        ax.set_title("Angle Search: Peak Sharpness vs Rotation Angle")
        ax.plot(angles, scores, color='purple')
        ax.axvline(x=best_angle, color='red', linestyle='--',
                   label=f'Best = {best_angle:+.2f}°')
        ax.set_xlabel("Rotation angle (degrees)")
        ax.set_ylabel("Peak sharpness score")
        ax.legend()

        ax = axes[1, 1]
        ax.set_title("1D Row Brightness Projection (after deskew)")
        ax.plot(row_mean_smooth, range(len(row_mean_smooth)), color='blue', label='Brightness')
        ax.axhline(y=peak_y, color='red', linestyle='--', label='Peak')
        ax.axhspan(ymin=y_start, ymax=y_end, color='green', alpha=0.3, label='ROI')
        ax.invert_yaxis()
        ax.set_xlabel("Avg brightness")
        ax.set_ylabel("Y coordinate (pixels)")
        ax.legend()

    plt.tight_layout()
    plt.show()

    return result


if __name__ == "__main__":
    # 待检测工件图路径
    TEST_IMAGE = "./all_frames/movie270_00000003.png"

    # 自动搜索模式（推荐）：rotate_angle 留空，让程序扫描出最佳角度
    find_highlight_for_test_image(
        TEST_IMAGE,
        rotate_angle=None,           # None = 自动搜索
        angle_offset=2.8,            # 正数 = 在自动结果基础上再顺时针转几度
        offset_from_peak=33,         # 与训练时保持一致
        roi_width=55,                # 与训练时保持一致
        angle_search_range=(-8.0, 8.0),
        angle_search_step=0.2,
        flip_vertical=True,          # 默认 True：翻转以对齐训练集（高光上方为清晰区）
    )

    # 如果想锁定一个角度手动复核，可以这样调：
    # find_highlight_for_test_image(
    #     TEST_IMAGE,
    #     rotate_angle=-2.0,
    #     angle_offset=0.0,
    #     offset_from_peak=33,
    #     roi_width=55,
    # )

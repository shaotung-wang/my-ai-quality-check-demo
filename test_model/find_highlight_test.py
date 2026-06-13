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
    h, w = gray.shape
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


def find_highlight_for_test_image(image_path,
                                  rotate_angle=None,
                                  offset_from_peak=33,
                                  roi_width=55,
                                  angle_search_range=(-8.0, 8.0),
                                  angle_search_step=0.2):
    """
    针对待检测工件图，自动搜索高光所在的旋转角，并定位高光上方的 ROI 条带。
    可视化输出：
      左上：原图（标出检测到的高光中心线倾斜方向）
      右上：拉直后图像，红线为高光峰值 Y，绿色带为 ROI 条带
      左下：角度-峰值锐度曲线（角度搜索过程）
      右下：拉直后亮度的 1D 行投影
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取图片: {image_path}")
        return None

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

    # 2. 用最佳角度拉直
    straight_img = manual_deskew_image(img, angle_deg=best_angle)

    # 3. 找高光峰值 Y
    row_mean_smooth = _row_brightness(straight_img)
    peak_y = int(np.argmax(row_mean_smooth))

    # 4. 计算 ROI 条带（高光上方）
    y_end = peak_y - offset_from_peak
    y_start = y_end - roi_width
    if y_start < 0:
        print("⚠️ 警告: ROI 已撞到图像顶部，自动截断到 y=0")
        y_start = 0
    if y_end <= y_start:
        y_end = y_start + 10

    print(f"✅ 高光峰值行: Y={peak_y}    ROI 条带: Y=[{y_start}:{y_end}] (高度 {y_end - y_start}px)")
    print("=" * 60)
    print("👉 把下面这组参数直接喂给 test_patch_builder.py:")
    print(f"     rotate_angle    = {best_angle:+.2f}")
    print(f"     offset_from_peak = {offset_from_peak}")
    print(f"     roi_width       = {roi_width}")
    print("=" * 60)

    # 5. 可视化
    if angles is not None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes = np.array([axes, [None, None]])  # 让索引兼容

    # 左上：原图 + 倾斜方向示意
    ax = axes[0, 0]
    ax.set_title(f"Original (detected tilt: {best_angle:+.2f}°)")
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

    return {
        "rotate_angle": best_angle,
        "peak_y": peak_y,
        "y_start": y_start,
        "y_end": y_end,
        "offset_from_peak": offset_from_peak,
        "roi_width": roi_width,
    }


if __name__ == "__main__":
    # 待检测工件图路径
    TEST_IMAGE = "./ng/ng_test.jpg"

    # 自动搜索模式（推荐）：rotate_angle 留空，让程序扫描出最佳角度
    find_highlight_for_test_image(
        TEST_IMAGE,
        rotate_angle=None,           # None = 自动搜索
        offset_from_peak=33,         # 与训练时保持一致
        roi_width=55,                # 与训练时保持一致
        angle_search_range=(-8.0, 8.0),
        angle_search_step=0.2,
    )

    # 如果想锁定一个角度手动复核，可以这样调：
    # find_highlight_for_test_image(
    #     TEST_IMAGE,
    #     rotate_angle=-2.0,
    #     offset_from_peak=33,
    #     roi_width=55,
    # )

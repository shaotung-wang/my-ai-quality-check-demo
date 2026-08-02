"""
共享核心库：高光定位、图像纠偏、ROI 提取。

本模块合并了此前分散在 model_train/find_highlight.py 和
test_model/find_highlight_test.py 中的重复逻辑，是唯一的高光搜索与 ROI 计算入口。

关键约定（由原始文件名决定，而不是由“训练/检测”阶段决定）：
- imageXXX：待采集清晰区域在高光上方 → flip_vertical=False
- movieXXX_XXX：待采集清晰区域在高光下方 → flip_vertical=True（翻转后高光上方即为原图下方）
- angle_offset：未翻转时正数 = 顺时针；翻转后正数 = 逆时针
"""

import os
import re

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


# 采集端的文件名同时标识了相机视角。把规则放在这里，确保训练、
# 离线评估和现场推理不会各自维护一套相互矛盾的 flip/角度逻辑。
_IMAGE_NAME = re.compile(r"^image\d+$", re.IGNORECASE)
_MOVIE_NAME = re.compile(r"^movie\d+_\d+$", re.IGNORECASE)

# 以下数值保留原流程已人工校准过的角度，只是把它们绑定到正确的采集
# 视角：imageXXX 原先在训练流程使用 -2°；movieXXX_XXX 原先在检测
# 流程使用自动角度 + 2.8° 校准。
IMAGE_ROTATE_ANGLE = -2.0
IMAGE_ANGLE_OFFSET = 0.0
MOVIE_ROTATE_ANGLE = None
MOVIE_ANGLE_OFFSET = 2.8


def roi_settings_for_path(image_path, *, unknown_flip_vertical=True):
    """按原始文件名返回正确的 ROI/旋转设置。

    ``imageXXX`` 的有效采集区位于高光上方，不翻转；
    ``movieXXX_XXX`` 的有效采集区位于高光下方，先上下翻转，使两者
    都以“高光上方”为统一坐标系裁剪。

    ``unknown_flip_vertical`` 仅用于没有这两种命名的历史/NG 文件。
    这些文件无法从名称判断来源，默认保持旧检测流程的下方采集规则。
    """
    stem = os.path.splitext(os.path.basename(os.fspath(image_path)))[0]
    if _IMAGE_NAME.fullmatch(stem):
        return {
            "source_type": "image",
            "flip_vertical": False,
            "rotate_angle": IMAGE_ROTATE_ANGLE,
            "angle_offset": IMAGE_ANGLE_OFFSET,
        }
    if _MOVIE_NAME.fullmatch(stem):
        return {
            "source_type": "movie",
            "flip_vertical": True,
            "rotate_angle": MOVIE_ROTATE_ANGLE,
            "angle_offset": MOVIE_ANGLE_OFFSET,
        }
    return {
        "source_type": "unknown",
        "flip_vertical": bool(unknown_flip_vertical),
        "rotate_angle": None,
        "angle_offset": MOVIE_ANGLE_OFFSET if unknown_flip_vertical else 0.0,
    }


# ============================================================
# 基础工具函数
# ============================================================

def manual_deskew_image(img, angle_deg):
    """固定角度纠偏。

    OpenCV 的 getRotationMatrix2D 使用正角度表示逆时针旋转。
    """
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def _row_brightness(img, sigma=5):
    """对一张图按行计算平滑后的亮度投影（中间 60% 宽度，避开暗角）。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, w = gray.shape
    crop = gray[:, int(w * 0.2):int(w * 0.8)]
    row_mean = np.mean(crop, axis=1)
    return gaussian_filter1d(row_mean, sigma=sigma)


def _peak_sharpness(row_mean_smooth):
    """评价高光峰值的锐度：峰值减背景均值，再除以背景标准差。

    高光越是被拉直成一条水平线，峰值越尖锐、得分越高。
    """
    peak = float(row_mean_smooth.max())
    bg_mean = float(row_mean_smooth.mean())
    bg_std = float(row_mean_smooth.std()) + 1e-6
    return (peak - bg_mean) / bg_std


# ============================================================
# 自动角度搜索
# ============================================================

def auto_search_rotate_angle(img, angle_range=(-8.0, 8.0), step=0.2):
    """在 [angle_range[0], angle_range[1]] 内以 step° 步长扫描，
    返回让水平高光峰值最锐的角度。

    Returns:
        best_angle, angles, scores
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


# ============================================================
# 核心：ROI 提取
# ============================================================

def extract_roi_from_image(img,
                           rotate_angle=None,
                           angle_offset=0.0,
                           offset_from_peak=33,
                           roi_width=55,
                           angle_search_range=(-8.0, 8.0),
                           angle_search_step=0.2,
                           flip_vertical=True,
                           verbose=True):
    """对输入图像自动搜索高光、拉直并截取 ROI 条带。

    参数：
        img: BGR 图像 (numpy array)
        rotate_angle: 手动指定拉直角度；None 则自动搜索。
        angle_offset: 人工校准偏移。
                      未翻转时正数 = 顺时针；翻转后正数 = 逆时针。
        offset_from_peak: ROI 起始行与高光峰值行的垂直距离（像素）。
        roi_width: ROI 条带高度（像素）。
        angle_search_range: 自动角度搜索范围。
        angle_search_step: 自动角度搜索步长。
        flip_vertical: 是否先上下翻转图像。
                       True  → 待检测图模式（原图 ROI 在高光下方，翻转后对齐训练集）。
                       False → 训练图模式（ROI 直接在高光上方）。
        verbose: 是否打印详细信息。

    返回：
        dict 包含 roi_strip 及定位元信息；若 ROI 无效则返回 None。
    """
    # 对待检测图做垂直翻转，对齐训练照片视角（高光上方为清晰区）
    if flip_vertical:
        img = cv2.flip(img, 0)
        if verbose:
            print("🔄 已对待检测图做上下翻转，以匹配训练集视角（高光上方为清晰区）")

    # 1. 自动搜索旋转角（除非用户已手动指定）
    if rotate_angle is None:
        best_angle, angles, scores = auto_search_rotate_angle(
            img, angle_range=angle_search_range, step=angle_search_step
        )
        if verbose:
            print(f"🔍 自动搜索得到最佳旋转角度: {best_angle:+.2f}°")
    else:
        best_angle = float(rotate_angle)
        angles, scores = None, None
        if verbose:
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
        if verbose:
            print(f"🔄 叠加校准偏移: {angle_offset:+.2f}°（{direction}），实际使用角度: {best_angle:+.2f}°")

    # 2. 用最佳角度拉直
    straight_img = manual_deskew_image(img, angle_deg=best_angle)

    # 3. 找高光峰值 Y
    row_mean_smooth = _row_brightness(straight_img)
    peak_y = int(np.argmax(row_mean_smooth))

    # 4. 计算 ROI 条带
    if flip_vertical:
        # 翻转后清晰区位于高光上方
        y_end = peak_y - offset_from_peak
        y_start = y_end - roi_width
    else:
        # 未翻转：清晰区位于高光上方（训练集）
        y_end = peak_y - offset_from_peak
        y_start = y_end - roi_width

    img_h = straight_img.shape[0]
    if y_start < 0:
        if verbose:
            print(f"⚠️ 警告: ROI 已撞到图像顶部，自动截断到 y=0")
        y_start = 0
    if y_end <= y_start:
        if verbose:
            print("❌ ROI 高度无效，无法定位有效区域")
        return None

    direction_label = "上方"
    if verbose:
        print(f"✅ 高光峰值行: Y={peak_y}    ROI 条带（高光{direction_label}）: Y=[{y_start}:{y_end}] (高度 {y_end - y_start}px)")
        print("=" * 60)
        print("👉 把下面这组参数直接喂给切片脚本:")
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


# ============================================================
# 可视化函数
# ============================================================

def find_highlight_for_test_image(image_path,
                                  rotate_angle=None,
                                  angle_offset=0.0,
                                  offset_from_peak=33,
                                  roi_width=55,
                                  angle_search_range=(-8.0, 8.0),
                                  angle_search_step=0.2,
                                  flip_vertical=True):
    """针对待检测工件图，自动搜索高光所在的旋转角，并定位 ROI 条带（带可视化）。

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
    straight_img = result["straight_img"]
    img = result["processed_img"]
    row_mean_smooth = result["row_mean_smooth"]
    angles = result["angles"]
    scores = result["scores"]

    # 可视化
    if angles is not None:
        _, axes = plt.subplots(2, 2, figsize=(14, 10))
    else:
        _, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes = np.array([axes, [None, None]])

    # 左上：翻转后的原图 + 倾斜方向示意
    ax = axes[0, 0]
    title_prefix = "Flipped + " if flip_vertical else ""
    ax.set_title(f"{title_prefix}Original (detected tilt: {best_angle:+.2f}°)")
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    h_img, w_img = img.shape[:2]
    cx, cy = w_img / 2, h_img / 2
    dx = w_img * 0.45
    dy = dx * np.tan(np.deg2rad(-best_angle))
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


def preview_highlight(image_path, rotate_angle=0.0, offset_from_peak=33, roi_width=55):
    """快速预览：寻找水平高光并定位 ROI 区域（仅绘图，不保存）。

    适用于训练图片（flip_vertical=False），直接在高光上方定位 ROI。
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取图片: {image_path}")
        return

    # 旋转纠偏
    straight_img = manual_deskew_image(img, angle_deg=rotate_angle)

    # 按行求平均，寻找水平高光
    gray = cv2.cvtColor(straight_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    crop_gray = gray[:, int(w * 0.2):int(w * 0.8)]
    row_mean = np.mean(crop_gray, axis=1)
    row_mean_smooth = gaussian_filter1d(row_mean, sigma=5)
    peak_y = int(np.argmax(row_mean_smooth))

    # 计算 ROI（高光上方）
    y_end = peak_y - offset_from_peak
    y_start = y_end - roi_width
    if y_start < 0:
        print(f"⚠️ 警告：设定的高度超出了图像顶边缘，已自动从 0 开始截断。")
        y_start = 0
    if y_end <= y_start:
        y_end = y_start + 10

    print(f"✅ 成功定位 ROI 区域: Y=[{y_start}:{y_end}]")

    # 绘图展示
    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    plt.title("Horizontal ROI Auto-Locked")
    plt.imshow(cv2.cvtColor(straight_img, cv2.COLOR_BGR2RGB))
    plt.axhline(y=peak_y, color='red', linestyle='--', label='Highlight Peak (Y)')
    plt.axhspan(ymin=y_start, ymax=y_end, color='green', alpha=0.3,
                label=f'Target ROI (Height: {y_end - y_start}px)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("1D Row Brightness Projection")
    plt.plot(row_mean_smooth, range(len(row_mean_smooth)), color='blue', label='Brightness')
    plt.axhline(y=peak_y, color='red', linestyle='--', label='Peak')
    plt.axhspan(ymin=y_start, ymax=y_end, color='green', alpha=0.3, label='ROI')
    plt.gca().invert_yaxis()
    plt.ylabel('Y Coordinate (Pixels)')
    plt.xlabel('Average Brightness')
    plt.legend()

    plt.tight_layout()
    plt.show()


# ============================================================
# 命令行自测
# ============================================================

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "data/test/images/movie270_00000004.png"

    # 自动搜索模式
    find_highlight_for_test_image(
        image_path,
        rotate_angle=None,
        angle_offset=2.8,
        offset_from_peak=33,
        roi_width=55,
        angle_search_range=(-8.0, 8.0),
        angle_search_step=0.2,
        flip_vertical=True,
    )

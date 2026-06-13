import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def manual_deskew_image(img, angle_deg):
    """手动固定角度纠偏"""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0))
    return rotated_img


def preview_metal_surface_horizontal(image_path, rotate_angle=0.0, offset_from_peak=60, roi_width=300):
    """
    仅预览：寻找水平高光，定位下方纹理区，直接输出绘图不保存图片。
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取图片: {image_path}")
        return

    # 1. 旋转纠偏
    straight_img = manual_deskew_image(img, angle_deg=rotate_angle)

    # 2. 按行求平均，寻找水平高光 (Peak Y)
    gray = cv2.cvtColor(straight_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 取图像中间的 60% 宽度计算，避开左右边缘可能的暗角干扰
    crop_gray = gray[:, int(w * 0.2):int(w * 0.8)]
    row_mean = np.mean(crop_gray, axis=1)

    row_mean_smooth = gaussian_filter1d(row_mean, sigma=5)

    # 找到最亮的行（水平高光带的中心 Y 坐标）
    peak_y = int(np.argmax(row_mean_smooth))

    # 3. 计算裁剪的上下边界 (提取光带上方的区域)
    y_end = peak_y - offset_from_peak
    y_start = y_end - roi_width

    # Y轴边界安全保护
    if y_start < 0:
        print(f"⚠️ 警告：设定的高度超出了图像顶边缘，已自动从 0 开始截断。")
        y_start = 0
    if y_end <= y_start:
        y_end = y_start + 10

    print(f"✅ 成功定位横向纹理带: Y=[{y_start}:{y_end}]")

    # 4. 绘图展示
    plt.figure(figsize=(14, 7))

    # 左图：原图与水平选框
    plt.subplot(1, 2, 1)
    plt.title("Horizontal ROI Auto-Locked")
    plt.imshow(cv2.cvtColor(straight_img, cv2.COLOR_BGR2RGB))
    plt.axhline(y=peak_y, color='red', linestyle='--', label='Highlight Peak (Y)')
    plt.axhspan(ymin=y_start, ymax=y_end, color='green', alpha=0.3,
                label=f'Target ROI (Height: {y_end - y_start}px)')
    plt.legend()

    # 右图：1D 波形图
    plt.subplot(1, 2, 2)
    plt.title("1D Row Brightness Projection")
    plt.plot(row_mean_smooth, range(len(row_mean_smooth)), color='blue', label='Brightness')
    plt.axhline(y=peak_y, color='red', linestyle='--', label='Peak')
    plt.axhspan(ymin=y_start, ymax=y_end, color='green', alpha=0.3, label='ROI')
    plt.gca().invert_yaxis()  # 让波形图的 Y 轴也是向下增长
    plt.ylabel('Y Coordinate (Pixels)')
    plt.xlabel('Average Brightness')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 请换成你那张新的、横向的工业相机测试图路径
    TEST_IMAGE = "./raw_images/test_frame.jpg"

    # ==========================================
    # 🛠️ 调试参数区
    # ==========================================
    preview_metal_surface_horizontal(
        TEST_IMAGE,
        rotate_angle=-2.0,  # 微调旋转角度
        offset_from_peak=33,  # 高光上方多少像素开始是纯净纹理
        roi_width=55  # 提取的表面宽度
    )
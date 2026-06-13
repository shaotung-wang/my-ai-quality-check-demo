import cv2
import numpy as np
import matplotlib.pyplot as plt


def find_highlight_curve(img,
                         x_sigma=15,
                         y_sigma=3,
                         search_y_band=(0.1, 0.9),
                         poly_degree=3,
                         brightness_threshold_ratio=0.3):
    """
    逐列跟踪高光的 Y 坐标，得到一条 peak_y(x) 曲线（而非单一 peak_y）。

    Args:
        img: BGR 原图
        x_sigma: 横向高斯平滑强度（让相邻列互相支援，抑制单列噪声）
        y_sigma: 纵向高斯平滑强度（抑制单像素亮点抖动）
        search_y_band: 只在图像高度的这一段 [top%, bottom%] 内搜索峰值，
                       避免顶部夹具/底部塑料块的干扰
        poly_degree: 对得到的离散峰值序列做多项式拟合的阶数（None = 不拟合，只做高斯平滑）
        brightness_threshold_ratio: 拟合时，只用"列亮度 ≥ 全图最大列亮度 × 该比例"的列，
                                    剔除背景暗列对拟合的拖拽

    Returns:
        peak_curve: 长度为 W 的一维 float 数组，每列对应的高光 Y 坐标（已平滑/拟合）
        raw_peak_y: 拟合前的原始离散峰值（调试用）
        col_brightness: 每列峰值亮度，供调试判断哪些列可信
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape

    # 1. 二维高斯平滑：横向让相邻列互相支援，纵向抹掉单像素抖动
    smooth = cv2.GaussianBlur(gray, ksize=(0, 0), sigmaX=x_sigma, sigmaY=y_sigma)

    # 2. 限制搜索范围：只在图像中间一段高度内找峰，避开顶/底部干扰
    y_top = int(h * search_y_band[0])
    y_bot = int(h * search_y_band[1])
    band = smooth[y_top:y_bot, :]

    # 3. 每列求亮度最大行（相对 band 的）
    raw_peak_y = np.argmax(band, axis=0).astype(np.float32) + y_top
    col_brightness = band.max(axis=0)

    # 4. 多项式拟合（推荐）：以列亮度的平方作权重，让真正属于高光的列主导拟合
    if poly_degree is not None and poly_degree > 0:
        max_b = float(col_brightness.max())
        threshold = max_b * brightness_threshold_ratio
        mask = col_brightness >= threshold

        xs = np.arange(w, dtype=np.float32)
        if mask.sum() >= poly_degree + 1:
            weights = (col_brightness[mask] / max_b) ** 2
            coeffs = np.polyfit(xs[mask], raw_peak_y[mask], poly_degree, w=weights)
            peak_curve = np.polyval(coeffs, xs)
        else:
            print(f"⚠️ 可信列不足，回退到单纯高斯平滑")
            peak_curve = cv2.GaussianBlur(raw_peak_y.reshape(1, -1),
                                          ksize=(0, 1), sigmaX=x_sigma).flatten()
    else:
        # 不做拟合，只在 X 方向再做一次平滑
        peak_curve = cv2.GaussianBlur(raw_peak_y.reshape(1, -1),
                                      ksize=(0, 1), sigmaX=x_sigma).flatten()

    return peak_curve, raw_peak_y, col_brightness


def extract_curved_roi_strip(img, peak_curve, offset_from_peak, roi_width):
    """
    沿着 peak_curve 提取高光上方的"弯曲条带"，并用 cv2.remap 重采样成
    一条完全水平的扁平条带（高度 = roi_width，宽度 = 原图宽度）。

    输出条带每一列 j 的像素都来自原图 (j, peak_curve[j] - offset_from_peak - roi_width + i)。
    经过这一步之后，下游所有"水平条带 → 滑窗切 patch"的逻辑就都还能直接用。
    """
    h, w = img.shape[:2]

    # 构造重采样坐标
    dy_grid, x_grid = np.meshgrid(
        np.arange(roi_width, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing='ij'
    )

    peak_2d = peak_curve.astype(np.float32)[np.newaxis, :]
    src_x = x_grid
    src_y = peak_2d - offset_from_peak - roi_width + dy_grid

    flat_strip = cv2.remap(img, src_x, src_y,
                           interpolation=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(0, 0, 0))

    return flat_strip


def preview_curved_highlight(image_path,
                             offset_from_peak=33,
                             roi_width=55,
                             x_sigma=15,
                             y_sigma=3,
                             poly_degree=3,
                             brightness_threshold_ratio=0.3,
                             search_y_band=(0.1, 0.9)):
    """
    完整可视化：检测高光曲线 → 在原图上叠加 → 展示重采样后的水平条带。
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取图片: {image_path}")
        return None

    peak_curve, raw_peak, col_bright = find_highlight_curve(
        img,
        x_sigma=x_sigma,
        y_sigma=y_sigma,
        search_y_band=search_y_band,
        poly_degree=poly_degree,
        brightness_threshold_ratio=brightness_threshold_ratio,
    )

    flat_strip = extract_curved_roi_strip(img, peak_curve, offset_from_peak, roi_width)

    h, w = img.shape[:2]
    xs = np.arange(w)
    y_top_curve = peak_curve - offset_from_peak - roi_width
    y_bot_curve = peak_curve - offset_from_peak

    out_of_bounds = (y_top_curve < 0).sum()
    if out_of_bounds > 0:
        print(f"⚠️ 有 {out_of_bounds} 列的 ROI 顶部超出图像上边缘（会被填黑）")

    print(f"✅ 高光曲线已锁定")
    print(f"   peak_y 范围: [{peak_curve.min():.1f}, {peak_curve.max():.1f}]  "
          f"幅度: {peak_curve.max() - peak_curve.min():.1f}px")
    print(f"   重采样后水平条带: {flat_strip.shape[1]} x {flat_strip.shape[0]} (W x H)")

    # 可视化
    fig = plt.figure(figsize=(14, 9))

    # 上：原图 + 曲线 + 弯曲 ROI 带
    ax1 = plt.subplot(3, 1, 1)
    ax1.set_title("Original + Detected Curved Highlight + Curved ROI Band")
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.plot(xs, peak_curve, color='red', linewidth=1.5, label='peak_y(x) (fitted)')
    ax1.scatter(xs[::20], raw_peak[::20], color='orange', s=4, alpha=0.5,
                label='raw per-column peaks')
    ax1.fill_between(xs, y_top_curve, y_bot_curve,
                     color='green', alpha=0.3, label=f'ROI band (width {roi_width}px)')
    ax1.set_xlim(0, w)
    ax1.set_ylim(h, 0)
    ax1.legend(loc='lower left')

    # 中：每列亮度，便于判断曲线可信度
    ax2 = plt.subplot(3, 1, 2)
    ax2.set_title("Per-Column Peak Brightness (used as fit weights)")
    ax2.plot(xs, col_bright, color='purple')
    threshold = col_bright.max() * brightness_threshold_ratio
    ax2.axhline(y=threshold, color='red', linestyle='--',
                label=f'fit threshold ({brightness_threshold_ratio:.0%} of max)')
    ax2.set_xlim(0, w)
    ax2.set_xlabel("X (pixels)")
    ax2.set_ylabel("Max brightness in column")
    ax2.legend()

    # 下：重采样后的水平条带（用于馈给切片器）
    ax3 = plt.subplot(3, 1, 3)
    ax3.set_title(f"Unwarped Flat Strip ({flat_strip.shape[1]}×{flat_strip.shape[0]}) — feeds patch slicer")
    ax3.imshow(cv2.cvtColor(flat_strip, cv2.COLOR_BGR2RGB), aspect='auto')
    ax3.set_xlabel("X (pixels)")
    ax3.set_ylabel("Y in strip")

    plt.tight_layout()
    plt.show()

    return {
        "peak_curve": peak_curve,
        "flat_strip": flat_strip,
    }


if __name__ == "__main__":
    TEST_IMAGE = "./ng/ng_test.jpg"

    preview_curved_highlight(
        TEST_IMAGE,
        offset_from_peak=33,
        roi_width=55,
        x_sigma=15,                 # 横向越大，曲线越平滑（应对噪声）
        y_sigma=3,                  # 纵向小一点，保留峰位锐度
        poly_degree=3,              # 3 阶足够拟合常见弧度；改成 None 关闭多项式拟合
        brightness_threshold_ratio=0.3,  # 30% 以下的暗列不参与拟合
        search_y_band=(0.1, 0.9),   # 只在图像中段 80% 高度内找峰
    )

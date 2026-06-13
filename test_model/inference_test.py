import os

# 授予本地模型加载的安全通行证 (防 pickle 拦截)
os.environ["TRUST_REMOTE_CODE"] = "1"

import cv2
import warnings
import matplotlib.pyplot as plt
from anomalib.deploy import TorchInferencer

# 屏蔽 Anomalib 的 Legacy 烦人警告
warnings.filterwarnings("ignore", category=UserWarning, module="anomalib")


def test_single_patch():
    print("🚀 正在加载 EfficientAD 模型 (CPU 极速模式)...")

    # .pt 模型权重路径
    MODEL_WEIGHT_PATH = "../results/weights/torch/model.pt"

    if not os.path.exists(MODEL_WEIGHT_PATH):
        print(f"❌ 找不到模型文件！请检查路径: {MODEL_WEIGHT_PATH}")
        return

    # 初始化推理器
    inferencer = TorchInferencer(
        path=MODEL_WEIGHT_PATH,
        device="cpu"  # 继续使用 CPU 极速模式
    )

    # 实际的测试图路径
    # TEST_IMAGE_PATH = "./My_Metal_Project/train/good/rod1_image111_patch_00.jpg"
    TEST_IMAGE_PATH = "../ng/ng_test.jpg"

    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"❌ 找不到测试图片！请检查路径: {TEST_IMAGE_PATH}")
        return

    print(f"🔍 正在检测图片: {TEST_IMAGE_PATH}")

    # 执行推理
    predictions = inferencer.predict(image=TEST_IMAGE_PATH)

    # 解析核心结果
    anomaly_score = predictions.pred_score
    is_defective = predictions.pred_label

    print("=" * 40)
    print(f"🎯 异常得分 (Anomaly Score): {anomaly_score.item():.4f}")
    print(f"🛠️ 模型最终判定: {'❌ 发现瑕疵 (NG)' if is_defective else '✅ 正常 (OK)'}")
    print("=" * 40)

    # 可视化热力图
    heatmap = predictions.anomaly_map.detach().cpu().numpy().squeeze()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Patch")
    img_rgb = cv2.cvtColor(cv2.imread(TEST_IMAGE_PATH), cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)

    plt.subplot(1, 2, 2)
    plt.title("EfficientAD Anomaly Heatmap")
    plt.imshow(heatmap, cmap='jet')
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    test_single_patch()
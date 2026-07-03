import os

# 授予本地模型加载的安全通行证 (防 pickle 拦截)
os.environ["TRUST_REMOTE_CODE"] = "1"

import cv2
import glob
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无 GUI 环境也能保存图片
import matplotlib.pyplot as plt
from anomalib.deploy import TorchInferencer

# 屏蔽 Anomalib 的 Legacy 烦人警告
warnings.filterwarnings("ignore", category=UserWarning, module="anomalib")


def load_inferencer(model_weight_path, device="cpu"):
    """加载 EfficientAD Torch 推理器"""
    if not os.path.exists(model_weight_path):
        raise FileNotFoundError(f"❌ 找不到模型文件！请检查路径: {model_weight_path}")

    print(f"🚀 正在加载 EfficientAD 模型 ({device} 模式)...")
    inferencer = TorchInferencer(
        path=model_weight_path,
        device=device,
    )
    return inferencer


def test_single_patch(inferencer, image_path):
    """对单张 patch 进行推理并返回结果字典"""
    if not os.path.exists(image_path):
        print(f"❌ 找不到测试图片: {image_path}")
        return None

    predictions = inferencer.predict(image=image_path)
    return {
        "image_path": image_path,
        "anomaly_score": float(predictions.pred_score),
        "is_defective": bool(predictions.pred_label),
        "predictions": predictions,
    }


def save_result_figure(image_path, predictions, save_path):
    """保存单张 patch 的原始图 + 异常热力图"""
    heatmap = predictions.anomaly_map.detach().cpu().numpy().squeeze()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    img_bgr = cv2.imread(image_path)
    if img_bgr is not None:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        axes[0].imshow(img_rgb)
    axes[0].set_title("Original Patch")
    axes[0].axis("off")

    im = axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("EfficientAD Anomaly Heatmap")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def batch_inference(patches_dir,
                    model_weight_path="../results/weights/torch/model.pt",
                    output_dir=None,
                    device="cpu",
                    score_threshold=None,
                    save_figures=True):
    """
    对一个 patches 目录下的所有图片进行批量推理。

    参数：
      patches_dir: 待检测 patch 文件夹路径
      model_weight_path: .pt 模型权重路径
      output_dir: 结果输出目录；None 则使用 patches_dir 同级的 _results 目录
      device: "cpu" 或 "cuda"
      score_threshold: 自定义阈值；None 则使用模型默认 pred_label
      save_figures: 是否保存每张 patch 的可视化结果
    """
    # 收集所有待检测图片
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(patches_dir, ext)))
    image_paths = sorted(image_paths)

    if not image_paths:
        print(f"❌ 目录 {patches_dir} 下没有可处理的图片")
        return []

    # 准备输出目录
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(patches_dir), "_results")
    os.makedirs(output_dir, exist_ok=True)
    if save_figures:
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

    # 加载模型
    inferencer = load_inferencer(model_weight_path, device=device)

    print(f"🔍 开始对 {len(image_paths)} 张 patch 进行推理...")
    print("=" * 50)

    results = []
    for img_path in image_paths:
        result = test_single_patch(inferencer, img_path)
        if result is None:
            continue

        # 使用自定义阈值或模型默认判定
        if score_threshold is not None:
            result["is_defective"] = result["anomaly_score"] >= score_threshold

        results.append(result)
        status = "❌ NG" if result["is_defective"] else "✅ OK"
        print(f"{status} | {os.path.basename(img_path):<30} | score={result['anomaly_score']:.4f}")

        # 保存可视化图
        if save_figures:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            save_path = os.path.join(vis_dir, f"{base_name}_result.jpg")
            save_result_figure(img_path, result["predictions"], save_path)

    print("=" * 50)

    # 汇总结果
    if results:
        scores = [r["anomaly_score"] for r in results]
        max_score = max(scores)
        mean_score = sum(scores) / len(scores)
        ng_count = sum(1 for r in results if r["is_defective"])
        overall_ng = ng_count > 0

        print(f"📊 总 patch 数: {len(results)}")
        print(f"📊 异常 patch 数: {ng_count}")
        print(f"📊 最大异常得分: {max_score:.4f}")
        print(f"📊 平均异常得分: {mean_score:.4f}")
        print(f"🛠️ 整体判定: {'❌ 发现瑕疵 (NG)' if overall_ng else '✅ 正常 (OK)'}")

        # 保存汇总文本
        summary_path = os.path.join(output_dir, "summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"model: {model_weight_path}\n")
            f.write(f"patches_dir: {patches_dir}\n")
            f.write(f"total_patches: {len(results)}\n")
            f.write(f"ng_patches: {ng_count}\n")
            f.write(f"max_score: {max_score:.6f}\n")
            f.write(f"mean_score: {mean_score:.6f}\n")
            f.write(f"overall: {'NG' if overall_ng else 'OK'}\n\n")
            for r in results:
                label = "NG" if r["is_defective"] else "OK"
                f.write(f"{label}\t{r['anomaly_score']:.6f}\t{r['image_path']}\n")
        print(f"📝 汇总结果已保存至: {summary_path}")
    else:
        print("⚠️ 没有有效的推理结果")

    return results


def test_single_patch_demo():
    """原始单张图片推理演示（兼容旧用法）"""
    MODEL_WEIGHT_PATH = "../results/weights/torch/model.pt"
    TEST_IMAGE_PATH = "../ng/ng_test.jpg"

    inferencer = load_inferencer(MODEL_WEIGHT_PATH, device="cpu")
    result = test_single_patch(inferencer, TEST_IMAGE_PATH)
    if result is None:
        return

    print("=" * 40)
    print(f"🎯 异常得分 (Anomaly Score): {result['anomaly_score']:.4f}")
    print(f"🛠️ 模型最终判定: {'❌ 发现瑕疵 (NG)' if result['is_defective'] else '✅ 正常 (OK)'}")
    print("=" * 40)

    # 可视化热力图
    heatmap = result["predictions"].anomaly_map.detach().cpu().numpy().squeeze()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Patch")
    img_rgb = cv2.cvtColor(cv2.imread(TEST_IMAGE_PATH), cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)

    plt.subplot(1, 2, 2)
    plt.title("EfficientAD Anomaly Heatmap")
    plt.imshow(heatmap, cmap="jet")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    # 以本脚本所在目录为基准，避免在不同工作目录下运行时路径错乱
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

    # ==========================================================
    # 模式 1：单张图片演示（与原始代码行为一致）
    # ==========================================================
    # test_single_patch_demo()

    # ==========================================================
    # 模式 2：批量推理 test_patch_builder.py 裁剪出的 patches 目录
    # ==========================================================
    PATCHES_DIR = os.path.join(PROJECT_ROOT, "test_patches", "movie270_00000003", "patches")
    MODEL_WEIGHT_PATH = os.path.join(PROJECT_ROOT, "results", "weights", "torch", "model.pt")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "test_patches", "movie270_00000003", "_results")

    batch_inference(
        patches_dir=PATCHES_DIR,
        model_weight_path=MODEL_WEIGHT_PATH,
        output_dir=OUTPUT_DIR,
        device="cpu",
        score_threshold=None,  # 使用模型默认阈值；想自定义可改为 0.5 等
        save_figures=True,
    )

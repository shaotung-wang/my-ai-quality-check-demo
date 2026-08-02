"""
PaDiM 推理测试脚本。

对已切好的 patch 目录进行批量推理，输出异常分数和可视化热力图。
支持自定义阈值校准。

用法：
    python scripts/test/run_inference.py

配置：
    修改 main 区块中的路径和阈值参数。
"""

import os
import sys
import glob
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import torch
from torchvision.transforms.v2.functional import to_dtype, to_image
from anomalib.deploy import TorchInferencer

warnings.filterwarnings("ignore", category=UserWarning, module="anomalib")

# ============================================================
# 异常分数阈值
# ============================================================
# 通过对正常训练图片跑推理统计得到（P99 分位数）
# 策略：P99 意味着约 1% 正常图片误判 NG，能检出大部分真实缺陷
DEFAULT_SCORE_THRESHOLD = 42.77


def calibrate_threshold(model_weight_path, train_good_dir, device="cpu", percentile=99):
    """对训练集正常图片跑推理，返回指定分位数的分数作为阈值。

    返回：(threshold_score, max_score, mean_score, scores_list)
    """
    inferencer = load_inferencer(model_weight_path, device=device)
    lightning_model = inferencer.model
    pre_processor = getattr(lightning_model, "pre_processor", None)

    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(train_good_dir, ext)))
    image_paths = sorted(image_paths)

    if not image_paths:
        raise FileNotFoundError(f"❌ 训练正常图片目录为空: {train_good_dir}")

    print(f"🔧 正在校准阈值，共 {len(image_paths)} 张正常图片...")
    scores = []
    for img_path in image_paths:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = to_dtype(to_image(img_rgb), torch.float32, scale=True)
        with torch.no_grad():
            proc_input = img_tensor.unsqueeze(0).to(inferencer.device)
            proc_output = pre_processor(proc_input) if pre_processor is not None else proc_input
            predictions = lightning_model.model(proc_output)
            scores.append(float(predictions.pred_score))

    threshold = float(np.percentile(scores, percentile))
    max_score = max(scores)
    mean_score = sum(scores) / len(scores)
    print(f"✅ 校准完成: P{percentile}={threshold:.6f}, max={max_score:.6f}, mean={mean_score:.6f}")
    return threshold, max_score, mean_score, scores


def load_inferencer(model_weight_path, device="cpu"):
    """加载 PaDiM Torch 推理器"""
    if not os.path.exists(model_weight_path):
        raise FileNotFoundError(f"❌ 找不到模型文件！请检查路径: {model_weight_path}")
    print(f"🚀 正在加载 PaDiM 模型 ({device} 模式)...")
    return TorchInferencer(path=model_weight_path, device=device)


def test_single_patch(inferencer, image_path):
    """对单张 patch 推理并返回原始马氏距离（绕过 PostProcessor 的 clamp）"""
    if not os.path.exists(image_path):
        print(f"❌ 找不到测试图片: {image_path}")
        return None

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"❌ 无法读取图片: {image_path}")
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = to_dtype(to_image(img_rgb), torch.float32, scale=True)

    lightning_model = inferencer.model
    pre_processor = getattr(lightning_model, "pre_processor", None)

    with torch.no_grad():
        proc_input = img_tensor.unsqueeze(0).to(inferencer.device)
        proc_output = pre_processor(proc_input) if pre_processor is not None else proc_input
        predictions = lightning_model.model(proc_output)

    return {
        "image_path": image_path,
        "anomaly_score": float(predictions.pred_score),
        "is_defective": False,
        "predictions": predictions,
    }


def save_result_figure(image_path, predictions, save_path, is_defective=False, score=0.0, threshold=None):
    """保存单张 patch 的原始图 + 异常热力图"""
    heatmap = predictions.anomaly_map.detach().cpu().numpy().squeeze()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    img_bgr = cv2.imread(image_path)
    if img_bgr is not None:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        axes[0].imshow(img_rgb)
    status_text = f"{'NG' if is_defective else 'OK'} (score={score:.2f}"
    if threshold is not None:
        status_text += f", thresh={threshold:.2f})"
    else:
        status_text += ")"
    axes[0].set_title(f"Original Patch\n{status_text}")
    axes[0].axis("off")

    im = axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("PaDiM Anomaly Heatmap")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def batch_inference(patches_dir,
                    model_weight_path,
                    output_dir=None,
                    device="cpu",
                    score_threshold=DEFAULT_SCORE_THRESHOLD,
                    save_figures=True):
    """对一个 patches 目录下的所有图片进行批量推理。

    参数：
      patches_dir: 待检测 patch 文件夹路径
      model_weight_path: .pt 模型权重路径
      output_dir: 结果输出目录；None 则使用 patches_dir 同级的 _results 目录
      device: "cpu" 或 "cuda"
      score_threshold: 原始马氏距离阈值；None 时不自动判定 NG
      save_figures: 是否保存每张 patch 的可视化结果
    """
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(patches_dir, ext)))
    image_paths = sorted(image_paths)

    if not image_paths:
        print(f"❌ 目录 {patches_dir} 下没有可处理的图片")
        return []

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(patches_dir), "_results")
    os.makedirs(output_dir, exist_ok=True)
    if save_figures:
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

    inferencer = load_inferencer(model_weight_path, device=device)

    if score_threshold is None:
        print("⚠️ 未指定 score_threshold，所有 patch 默认判为 OK")
    else:
        print(f"📐 当前阈值: score_threshold = {score_threshold}")

    print(f"🔍 开始对 {len(image_paths)} 张 patch 进行推理...")
    print("=" * 50)

    results = []
    for img_path in image_paths:
        result = test_single_patch(inferencer, img_path)
        if result is None:
            continue

        if score_threshold is not None:
            result["is_defective"] = result["anomaly_score"] >= score_threshold

        results.append(result)
        status = "❌ NG" if result["is_defective"] else "✅ OK"
        print(f"{status} | {os.path.basename(img_path):<30} | score={result['anomaly_score']:.4f}")

        if save_figures:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            save_path = os.path.join(vis_dir, f"{base_name}_result.jpg")
            save_result_figure(img_path, result["predictions"], save_path,
                             is_defective=result["is_defective"],
                             score=result["anomaly_score"],
                             threshold=score_threshold)

    print("=" * 50)

    if results:
        scores = [r["anomaly_score"] for r in results]
        max_score = max(scores)
        min_score = min(scores)
        mean_score = sum(scores) / len(scores)
        ng_count = sum(1 for r in results if r["is_defective"])
        ok_count = len(results) - ng_count
        overall_ng = ng_count > 0

        print(f"📊 总 patch 数: {len(results)}")
        print(f"📊 正常 (OK): {ok_count} | 异常 (NG): {ng_count}")
        print(f"📊 分数范围: {min_score:.4f} ~ {max_score:.4f}")
        print(f"📊 平均分数: {mean_score:.4f}")
        print(f"🛠️ 整体判定: {'❌ 发现瑕疵 (NG)' if overall_ng else '✅ 正常 (OK)'}")

        summary_path = os.path.join(output_dir, "summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"model: PaDiM ({model_weight_path})\n")
            f.write(f"patches_dir: {patches_dir}\n")
            f.write(f"score_threshold: {score_threshold}\n")
            f.write(f"total_patches: {len(results)}\n")
            f.write(f"ok_patches: {ok_count}\n")
            f.write(f"ng_patches: {ng_count}\n")
            f.write(f"min_score: {min_score:.6f}\n")
            f.write(f"max_score: {max_score:.6f}\n")
            f.write(f"mean_score: {mean_score:.6f}\n")
            f.write(f"overall: {'NG' if overall_ng else 'OK'}\n\n")
            for r in results:
                label = "NG" if r["is_defective"] else "OK"
                f.write(f"{label}\t{r['anomaly_score']:.6f}\t{r['image_path']}\n")
        print(f"📝 汇总结果已保存至: {summary_path}")

    return results


if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    MODEL_WEIGHT_PATH = os.path.join(PROJECT_ROOT, "outputs", "models", "padim", "torch", "model.pt")
    PATCHES_DIR = os.path.join(PROJECT_ROOT, "outputs", "patches",
                               "movie270_00000003", "patches")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "patches",
                              "movie270_00000003", "_results")

    batch_inference(
        patches_dir=PATCHES_DIR,
        model_weight_path=MODEL_WEIGHT_PATH,
        output_dir=OUTPUT_DIR,
        device="cpu",
        score_threshold=DEFAULT_SCORE_THRESHOLD,
        save_figures=True,
    )
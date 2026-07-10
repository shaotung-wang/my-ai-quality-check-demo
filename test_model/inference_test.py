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
import torch
from torchvision.transforms.v2.functional import to_dtype, to_image
from anomalib.deploy import TorchInferencer

# 屏蔽 Anomalib 的 Legacy 烦人警告
warnings.filterwarnings("ignore", category=UserWarning, module="anomalib")

# ============================================================
# 异常分数阈值（原始马氏距离）
# ------------------------------------------------------------
# 该值通过对 5124 张正常训练图片跑推理统计得到：
#   正常分数最小值 = 9.848657
#   正常分数 P1    = 12.183
#   正常分数 P5    = 14.002
#   正常分数中位数 = 21.414
#   正常分数 P95   = 33.565
#   正常分数 P99   = 42.769
#   正常分数最大值 = 89.343
#
# 策略：P99 分位数（实用平衡）
#   将阈值设在正常分数的 P99 处，意味着：
#   - 99% 的正常图片会判 OK，约 1% 正常图片误判 NG（可人工复核）
#   - 真实划痕分数通常显著高于 P99，能被有效检出
#   - 比取 min(正常分数) 实用得多（后者导致 100% 误报）
#   - 比取 max(正常分数) 安全得多（后者会漏掉大部分中等分数的缺陷）
#
#   注意：P99 无法保证严格 0 漏检。只有正常数据时无法确定划痕分数分布。
#   若需严格 0 漏检，请收集少量划痕样本，确认其分数均高于阈值后再定。
# ============================================================
DEFAULT_SCORE_THRESHOLD = 42.77


def calibrate_threshold(model_weight_path, train_good_dir, device="cpu", percentile=99):
    """
    对训练集正常图片跑推理，返回指定分位数的分数作为阈值。

    用法：当模型重新训练后，调用此函数重新校准阈值。
    参数：
      percentile: 取正常分数的第几百分位作为阈值（默认 99）
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
    print(f"   推荐阈值 (P{percentile}策略): {threshold:.6f}")
    return threshold, max_score, mean_score, scores


def load_inferencer(model_weight_path, device="cpu"):
    """加载 PaDiM Torch 推理器"""
    if not os.path.exists(model_weight_path):
        raise FileNotFoundError(f"❌ 找不到模型文件！请检查路径: {model_weight_path}")

    print(f"🚀 正在加载 PaDiM 模型 ({device} 模式)...")
    inferencer = TorchInferencer(
        path=model_weight_path,
        device=device,
    )
    return inferencer


def test_single_patch(inferencer, image_path):
    """对单张 patch 推理并返回原始马氏距离（绕过 PostProcessor 的 clamp）"""
    if not os.path.exists(image_path):
        print(f"❌ 找不到测试图片: {image_path}")
        return None

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"❌ 无法读取图片: {image_path}")
        return None

    # 手动读取并转成张量，与 TorchInferencer 内部对 numpy/PIL 的处理一致
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = to_dtype(to_image(img_rgb), torch.float32, scale=True)

    lightning_model = inferencer.model
    pre_processor = getattr(lightning_model, "pre_processor", None)

    with torch.no_grad():
        proc_input = img_tensor.unsqueeze(0).to(inferencer.device)
        proc_output = pre_processor(proc_input) if pre_processor is not None else proc_input
        # 直接调用 PadimModel，获取原始马氏距离，避开 PostProcessor 的 [0,1] 归一化
        predictions = lightning_model.model(proc_output)

    return {
        "image_path": image_path,
        "anomaly_score": float(predictions.pred_score),
        "is_defective": False,  # 默认不判 NG，由调用方根据 score_threshold 决定
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
                    model_weight_path="../results_padim/Padim/batch_dataset_padim/v0/weights/torch/model.pt",
                    output_dir=None,
                    device="cpu",
                    score_threshold=DEFAULT_SCORE_THRESHOLD,
                    save_figures=True):
    """
    对一个 patches 目录下的所有图片进行批量推理。

    参数：
      patches_dir: 待检测 patch 文件夹路径（可以是训练集之外新拍摄的图片）
      model_weight_path: .pt 模型权重路径
      output_dir: 结果输出目录；None 则使用 patches_dir 同级的 _results 目录
      device: "cpu" 或 "cuda"
      score_threshold: 原始马氏距离阈值；None 时不自动判定 NG（所有 patch 默认 OK）
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

    if score_threshold is None:
        print("⚠️ 未指定 score_threshold，所有 patch 默认判为 OK；请根据本次输出的 raw score 范围设定合适阈值")
    else:
        print(f"📐 当前阈值: score_threshold = {score_threshold}")
        print(f"   判定规则: score >= {score_threshold} → NG | score < {score_threshold} → OK")

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
            save_result_figure(img_path, result["predictions"], save_path,
                             is_defective=result["is_defective"],
                             score=result["anomaly_score"],
                             threshold=score_threshold)

    print("=" * 50)

    # 汇总结果
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
        if score_threshold is not None:
            print(f"📐 使用阈值: {score_threshold}")
        print(f"🛠️ 整体判定: {'❌ 发现瑕疵 (NG)' if overall_ng else '✅ 正常 (OK)'}")
        if score_threshold is None:
            print(f"💡 建议：观察到的 raw score 范围 {min_score:.4f} ~ {max_score:.4f}，可据此设定 score_threshold")

        # 保存汇总文本
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
    else:
        print("⚠️ 没有有效的推理结果")

    return results


if __name__ == "__main__":
    # 以本脚本所在目录为基准，避免在不同工作目录下运行时路径错乱
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

    # ==========================================================
    # 模型与数据路径
    # ==========================================================
    MODEL_WEIGHT_PATH = os.path.join(PROJECT_ROOT, "results_padim", "Padim", "batch_dataset_padim", "v0", "weights", "torch", "model.pt")
    PATCHES_DIR = os.path.join(PROJECT_ROOT, "test_patches", "movie270_00000003", "patches")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "test_patches", "movie270_00000003", "_results")

    # ==========================================================
    # 异常分数阈值
    # ------------------------------------------------------------
    # DEFAULT_SCORE_THRESHOLD = 42.77 是基于 5124 张正常训练图片统计得到的
    # P99 分位数（即 99% 的正常图片分数低于此值）
    #
    # 策略说明：
    #   - P99: 约 1% 正常图片误判 NG，能检出大部分真实缺陷（推荐）
    #   - max(89.34): 零误报但会漏检中等分数的缺陷，不推荐
    #   - min(9.84): 零漏检但 100% 误报，不实用
    #
    # 如果模型重新训练了，可以取消下面注释来重新校准阈值：
    #   TRAIN_GOOD_DIR = os.path.join(PROJECT_ROOT, "batch_dataset_padim", "train", "good")
    #   threshold, _, _, _ = calibrate_threshold(MODEL_WEIGHT_PATH, TRAIN_GOOD_DIR, percentile=99)
    #   SCORE_THRESHOLD = threshold  # P99 策略
    # ==========================================================
    SCORE_THRESHOLD = DEFAULT_SCORE_THRESHOLD  # 42.77 (P99)

    batch_inference(
        patches_dir=PATCHES_DIR,
        model_weight_path=MODEL_WEIGHT_PATH,
        output_dir=OUTPUT_DIR,
        device="cpu",
        score_threshold=SCORE_THRESHOLD,
        save_figures=True,
    )

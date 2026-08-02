"""
EfficientAd Lightning 检查点 (.ckpt) → anomalib Torch 推理权重 (.pt) 转换脚本。

将 results/EfficientAd/My_Metal_Project/v3 下的最新 Lightning checkpoint 转换为
TorchInferencer 可直接加载的 .pt 文件，并把 ckpt 与 pt 一起整理到
results/efficientad_2/ 目录下，与 outputs/models/efficientad_1 结构保持一致。

用法：
    python scripts/test/convert_ckpt_efficientad.py
    python scripts/test/convert_ckpt_efficientad.py --ckpt <path> --output <path>
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# macOS SSL 证书修复
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("SSL_CERT_DIR", os.path.dirname(certifi.where()))
except ImportError:
    pass

os.environ.setdefault("TRUST_REMOTE_CODE", "1")

import torch
from anomalib.deploy import TorchInferencer
from anomalib.models import EfficientAd


def find_latest_ckpt(model_root):
    """在指定目录下递归查找最新的 Lightning checkpoint。"""
    model_root = Path(model_root)
    ckpts = sorted(model_root.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts[0] if ckpts else None


def convert(ckpt_path, output_path, ckpt_copy_path=None, verify=True):
    """将 EfficientAd Lightning checkpoint 转换为 TorchInferencer 可用的 .pt 文件。

    参数：
        ckpt_path: 输入的 Lightning checkpoint 路径
        output_path: 输出的 .pt 文件路径
        ckpt_copy_path: 若提供，则将原始 ckpt 复制到该路径（保持目录结构）
        verify: 是否在导出后用 TorchInferencer 验证
    """
    ckpt_path = Path(ckpt_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()

    if not ckpt_path.exists():
        raise FileNotFoundError(f"❌ 找不到 checkpoint 文件: {ckpt_path}")

    if output_path.suffix.lower() not in {".pt", ".pth"}:
        output_path = output_path / "model.pt"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📥 正在加载 EfficientAd checkpoint: {ckpt_path}")
    model = EfficientAd.load_from_checkpoint(str(ckpt_path))
    print("✅ Checkpoint 恢复成功")

    # EfficientAd 的 to_torch 会把整个 Lightning 模型 pickle 到 .pt 文件中
    print(f"💾 正在导出 Torch 推理权重: {output_path}")
    exported_path = model.to_torch(output_path.parent)
    if Path(exported_path) != output_path:
        Path(exported_path).rename(output_path)
    print(f"🎉 导出完成: {output_path}")

    # 同时把 ckpt 也复制一份到 results/efficientad_2/weights/lightning/ 下，方便统一管理
    if ckpt_copy_path is not None:
        ckpt_copy_path = Path(ckpt_copy_path).expanduser().resolve()
        ckpt_copy_path.parent.mkdir(parents=True, exist_ok=True)
        if ckpt_copy_path.resolve() != ckpt_path.resolve():
            shutil.copy2(ckpt_path, ckpt_copy_path)
            print(f"📋 已复制原始 ckpt 到: {ckpt_copy_path}")

    if verify:
        print("🔍 正在用 TorchInferencer 验证导出的 .pt 文件...")
        inferencer = TorchInferencer(path=str(output_path), device="cpu")
        dummy_input = torch.zeros((1, 3, 256, 256))
        with torch.no_grad():
            pred = inferencer.model.model(dummy_input)
        score = float(pred.pred_score)
        print(f"✅ 验证通过，dummy pred_score={score:.6f}, "
              f"anomaly_map shape={tuple(pred.anomaly_map.shape)}")

    return output_path


def main():
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent.parent.parent

    # 默认查找 results/EfficientAd/ 下最新的 ckpt
    default_ckpt_root = project_root / "results" / "EfficientAd"
    latest_ckpt = find_latest_ckpt(default_ckpt_root) if default_ckpt_root.exists() else None
    default_ckpt = str(latest_ckpt) if latest_ckpt else \
        str(default_ckpt_root / "My_Metal_Project" / "v3" / "weights" / "lightning" / "model.ckpt")

    # 默认输出到 results/efficientad_2/torch/model.pt
    default_output = str(project_root / "results" / "efficientad_2" / "torch" / "model.pt")
    default_ckpt_copy = str(project_root / "results" / "efficientad_2" / "weights" / "lightning" / "model.ckpt")

    parser = argparse.ArgumentParser(
        description="将 EfficientAd Lightning checkpoint (.ckpt) 转换为 anomalib Torch 推理权重 (.pt)",
    )
    parser.add_argument(
        "--ckpt", type=str, default=default_ckpt,
        help="输入的 Lightning checkpoint 路径（默认自动查找 results/EfficientAd 下最新）",
    )
    parser.add_argument(
        "--output", type=str, default=default_output,
        help="输出的 .pt 文件路径",
    )
    parser.add_argument(
        "--ckpt-copy", type=str, default=default_ckpt_copy,
        help="若提供，则把原始 ckpt 复制到该路径；传空字符串可跳过复制",
    )
    parser.add_argument("--no-verify", action="store_true", help="跳过导出后的验证")

    args = parser.parse_args()

    try:
        convert(
            args.ckpt,
            args.output,
            ckpt_copy_path=(args.ckpt_copy if args.ckpt_copy else None),
            verify=not args.no_verify,
        )
    except Exception as exc:
        print(f"❌ 转换失败: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

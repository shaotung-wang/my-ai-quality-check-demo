"""
PaDiM Lightning 检查点 (.ckpt) → anomalib Torch 推理权重 (.pt) 转换脚本。

用法：
    python scripts/test/convert_ckpt.py
    python scripts/test/convert_ckpt.py --ckpt <path> --output <path>
"""

import argparse
import os
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
from anomalib.models import Padim


def convert(ckpt_path, output_path, verify=True):
    """将 PaDiM Lightning checkpoint 转换为 TorchInferencer 可用的 .pt 文件。"""
    ckpt_path = Path(ckpt_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()

    if not ckpt_path.exists():
        raise FileNotFoundError(f"❌ 找不到 checkpoint 文件: {ckpt_path}")

    if output_path.suffix.lower() not in {".pt", ".pth"}:
        output_path = output_path / "model.pt"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📥 正在加载 checkpoint: {ckpt_path}")
    model = Padim.load_from_checkpoint(str(ckpt_path))
    print("✅ Checkpoint 恢复成功")

    print(f"💾 正在导出 Torch 推理权重: {output_path}")
    exported_path = model.to_torch(output_path.parent)

    if output_path != exported_path:
        exported_path.rename(output_path)

    print(f"🎉 导出完成: {output_path}")

    if verify:
        print("🔍 正在用 TorchInferencer 验证导出的 .pt 文件...")
        inferencer = TorchInferencer(path=str(output_path), device="cpu")
        dummy_input = torch.zeros((1, 3, 256, 256))
        with torch.no_grad():
            _ = inferencer.model(dummy_input)
        print("✅ 验证通过，.pt 文件可被 TorchInferencer 正常加载与推理")

    return output_path


def main():
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent.parent

    parser = argparse.ArgumentParser(
        description="将 PaDiM Lightning checkpoint (.ckpt) 转换为 anomalib Torch 推理权重 (.pt)",
    )
    parser.add_argument(
        "--ckpt", type=str,
        default=str(project_root / "outputs" / "models" / "padim" / "lightning" / "model.ckpt"),
        help="输入的 Lightning checkpoint 路径",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(project_root / "outputs" / "models" / "padim" / "torch" / "model.pt"),
        help="输出的 .pt 文件路径",
    )
    parser.add_argument("--no-verify", action="store_true", help="跳过导出后的验证")

    args = parser.parse_args()

    try:
        convert(args.ckpt, args.output, verify=not args.no_verify)
    except Exception as exc:
        print(f"❌ 转换失败: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
"""
PaDiM Lightning 检查点 (.ckpt) → anomalib Torch 推理权重 (.pt) 转换脚本

说明：
    anomalib 的 TorchInferencer 期望加载一个包含 ``{"model": nn.Module}`` 的字典，
    其中 ``model`` 是完整的 AnomalyModule（包含预处理、PaDiM 模型、后处理阈值等）。
    本脚本读取训练好的 Lightning checkpoint，恢复 Padim 模型对象，并导出为兼容
    inference_test.py 的 .pt 文件。

用法示例：
    # 使用默认路径（从 test_model/ 目录执行）
    python convert_ckpt_to_pt.py

    # 自定义输入/输出路径
    python convert_ckpt_to_pt.py \
        --ckpt ../results_padim/Padim/batch_dataset_padim/v0/weights/lightning/model.ckpt \
        --output ../results_padim/Padim/batch_dataset_padim/v0/weights/torch/model.pt
"""

import argparse
import os
import sys
from pathlib import Path

# macOS 上 python.org 安装的 Python 3.13 缺 CA 根证书目录，
# anomalib 在下载预训练权重 / imagenette 时会 SSL 校验失败。
# 把 SSL_CERT_FILE 指到 venv 里的 certifi 即可。
try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("SSL_CERT_DIR", os.path.dirname(certifi.where()))
except ImportError:
    pass

# TorchInferencer 加载 .pt 依赖 pickle，需要显式声明信任远程代码
os.environ.setdefault("TRUST_REMOTE_CODE", "1")

import torch
from anomalib.deploy import TorchInferencer
from anomalib.models import Padim


def convert(
    ckpt_path: str | Path,
    output_path: str | Path,
    verify: bool = True,
) -> Path:
    """将 PaDiM Lightning checkpoint 转换为 TorchInferencer 可用的 .pt 文件。

    Args:
        ckpt_path: Lightning checkpoint (.ckpt) 路径。
        output_path: 导出的 .pt 文件路径；若指向目录，则在该目录下生成 model.pt。
        verify: 是否在导出后用 TorchInferencer 做一次加载验证。

    Returns:
        实际生成的 .pt 文件 Path。
    """
    ckpt_path = Path(ckpt_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()

    if not ckpt_path.exists():
        raise FileNotFoundError(f"❌ 找不到 checkpoint 文件: {ckpt_path}")

    # 如果 output_path 是目录，则按 anomalib 默认命名 model.pt
    if output_path.suffix.lower() not in {".pt", ".pth"}:
        output_path = output_path / "model.pt"

    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📥 正在加载 checkpoint: {ckpt_path}")
    # load_from_checkpoint 会自动读取 checkpoint 中保存的超参并恢复 Padim 模型
    model = Padim.load_from_checkpoint(str(ckpt_path))
    print("✅ Checkpoint 恢复成功")

    # to_torch 会把整个 AnomalyModule 以 {"model": self} 的形式 pickle 保存
    # 这正是 TorchInferencer.load_model 期望的格式
    print(f"💾 正在导出 Torch 推理权重: {output_path}")
    exported_path = model.to_torch(output_path.parent)

    if output_path != exported_path:
        # to_torch 返回的路径是 <parent>/weights/torch/model.pt；如果用户指定了别的名字，重命名一下
        exported_path.rename(output_path)

    print(f"🎉 导出完成: {output_path}")

    if verify:
        print("🔍 正在用 TorchInferencer 验证导出的 .pt 文件...")
        inferencer = TorchInferencer(path=str(output_path), device="cpu")
        # 用 dummy 张量跑一遍前向，确认模型可正常推理
        dummy_input = torch.zeros((1, 3, 256, 256))
        with torch.no_grad():
            _ = inferencer.model(dummy_input)
        print("✅ 验证通过，.pt 文件可被 TorchInferencer 正常加载与推理")

    return output_path


def main() -> int:
    # 以脚本所在目录为基准，避免在不同工作目录下运行时路径错乱
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description="将 PaDiM Lightning checkpoint (.ckpt) 转换为 anomalib Torch 推理权重 (.pt)",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=str(
            project_root
            / "results_padim"
            / "Padim"
            / "batch_dataset_padim"
            / "v0"
            / "weights"
            / "lightning"
            / "model.ckpt"
        ),
        help="输入的 Lightning checkpoint 路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(
            project_root
            / "results_padim"
            / "Padim"
            / "batch_dataset_padim"
            / "v0"
            / "weights"
            / "torch"
            / "model.pt"
        ),
        help="输出的 .pt 文件路径（或目录）",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="跳过导出后的 TorchInferencer 加载验证",
    )

    args = parser.parse_args()

    try:
        convert(args.ckpt, args.output, verify=not args.no_verify)
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 转换失败: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

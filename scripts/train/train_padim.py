"""
PaDiM 模型训练脚本。

使用 anomalib 框架对 MVTec AD 格式数据集进行特征建模（非神经网络训练）。
数据集需预先通过 build_dataset.py 构建。

用法：
    python scripts/train/train_padim.py
"""

import os
import sys
import certifi

# macOS SSL 证书修复（必须在 import anomalib 之前）
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("SSL_CERT_DIR", os.path.dirname(certifi.where()))

from anomalib.data import Folder
from anomalib.models import Padim
from anomalib.engine import Engine


def start_training(data_root="outputs/train_intermediate/padim", output_root="outputs/models/padim"):
    """启动 PaDiM 特征建模。

    参数：
        data_root: MVTec AD 格式数据集根目录（含 train/good/）
        output_root: 模型权重和日志输出目录
    """
    print("🚀 初始化工业视觉数据管道...")

    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"数据根目录未找到: {data_root}")

    datamodule = Folder(
        name="padim",
        root=data_root,
        normal_dir="train/good",
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=8,
        val_split_mode="from_train",
        val_split_ratio=0.1,
        test_split_mode="none",
    )

    print("🧠 构建 PaDiM 模型实例...")
    model = Padim(
        backbone="resnet18",
        pre_trained=True,
        layers=["layer1", "layer2", "layer3"],
        n_features=None,
    )

    print("⚙️ 配置硬件加速引擎（CPU：MPS 不支持 PaDiM memory bank 的大张量操作）...")
    engine = Engine(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        default_root_dir=output_root,
    )

    print("🔥 开始特征提取与高斯分布建模，请稍候...")
    engine.fit(datamodule=datamodule, model=model)

    print("✅ 训练流水线执行完毕！")
    print(f"📁 模型权重已自动保存在: {os.path.abspath(output_root)}")
    print("💡 PaDiM 模型在推理时：每个 patch 位置仅需一次马氏距离计算，")
    print("   无需内存库搜索，非常适合 M2 Mac mini 的 CPU 推理。")


if __name__ == "__main__":
    start_training()
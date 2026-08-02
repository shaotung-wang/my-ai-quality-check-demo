"""训练 EfficientAD，并复用已有 checkpoint 中的教师网络权重。

EfficientAD 的标准损失需要 ImageNette 作为异常曝光数据；训练集本身只应
放良品 patch。先运行 build_dataset.py 生成按采集视角统一后的数据集，再执行：

    .venv/bin/python scripts/train/train_efficientad.py \
        --data-root data/datasets/My_Metal_Project_roi_v2 \
        --imagenette-dir /private/tmp/imagenette2 \
        --output-root results/EfficientAd/My_Metal_Project_roi_v2
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import EfficientAd
from anomalib.utils.path import get_pretrained_weights_dir


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def seed_teacher_weights(checkpoint: Path) -> Path:
    """将可信旧 checkpoint 的教师权重放到 anomalib 的本地缓存。

    这样无需再次下载教师网络；只下载/指定标准的 ImageNette 辅助数据。
    """
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = payload["state_dict"]
    teacher = {
        key.removeprefix("model.teacher."): value
        for key, value in state_dict.items()
        if key.startswith("model.teacher.")
    }
    if not teacher:
        raise ValueError(f"checkpoint 中未找到 EfficientAD 教师权重: {checkpoint}")

    target_dir = get_pretrained_weights_dir() / "efficientad_pretrained_weights"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "pretrained_teacher_small.pth"
    torch.save(teacher, target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="训练按文件名 ROI 规则构建的 EfficientAD 模型")
    parser.add_argument("--data-root", type=Path,
                        default=PROJECT_ROOT / "data" / "datasets" / "My_Metal_Project_roi_v2")
    parser.add_argument("--imagenette-dir", type=Path, required=True,
                        help="解压后的 imagenette2 目录；其中应直接包含 10 个类别目录")
    parser.add_argument("--output-root", type=Path,
                        default=PROJECT_ROOT / "results" / "EfficientAd" / "My_Metal_Project_roi_v2")
    parser.add_argument("--init-checkpoint", type=Path,
                        default=PROJECT_ROOT / "results" / "efficientad_2" / "weights" / "lightning" / "model.ckpt")
    parser.add_argument("--max-steps", type=int, default=5000,
                        help="训练更新步数；正式训练建议至少 5000，默认 5000")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    if args.max_steps < 1:
        parser.error("--max-steps 必须大于等于 1")
    train_dir = args.data_root / "train" / "good"
    if not train_dir.is_dir() or not any(train_dir.iterdir()):
        raise FileNotFoundError(f"未找到训练良品 patch 目录: {train_dir}")
    if not args.imagenette_dir.is_dir() or not any(args.imagenette_dir.iterdir()):
        raise FileNotFoundError(f"未找到 ImageNette 目录: {args.imagenette_dir}")
    if not args.init_checkpoint.is_file():
        raise FileNotFoundError(f"未找到初始化 checkpoint: {args.init_checkpoint}")

    cache_path = seed_teacher_weights(args.init_checkpoint)
    print(f"✅ 已复用本地教师权重: {cache_path}")
    print(f"📁 良品训练 patch: {train_dir}")
    print(f"📁 ImageNette 辅助数据: {args.imagenette_dir}")

    datamodule = Folder(
        name="metal_roi_v2",
        root=args.data_root,
        normal_dir="train/good",
        train_batch_size=1,  # EfficientAD 算法要求 batch_size=1
        eval_batch_size=1,
        num_workers=args.num_workers,
        test_split_mode="none",
        val_split_mode="from_train",
        val_split_ratio=0.1,
        seed=42,
    )
    # 继续已有模型可保留已学到的 student/autoencoder、均值方差和分数归一化，
    # 只针对统一后的 ROI 分布继续拟合；教师仍会由上面的本地缓存可靠复用。
    model = EfficientAd.load_from_checkpoint(
        args.init_checkpoint,
        map_location="cpu",
        weights_only=False,
        imagenet_dir=args.imagenette_dir,
    )
    engine = Engine(
        max_steps=args.max_steps,
        max_epochs=-1,
        accelerator="cpu",
        devices=1,
        default_root_dir=args.output_root,
        logger=False,
        enable_checkpointing=True,
    )

    print(f"🚀 开始 CPU 训练（{args.max_steps} steps）...")
    engine.fit(model=model, datamodule=datamodule)
    print(f"✅ 训练完成，结果目录: {args.output_root.resolve()}")


if __name__ == "__main__":
    main()

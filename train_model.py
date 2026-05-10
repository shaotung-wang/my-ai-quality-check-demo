import os
import argparse
from ultralytics import YOLO

# 尝试导入 anomalib，如果不存在则提示用户安装
try:
    from anomalib.data import Folder
    from anomalib.models import EfficientAd
    from anomalib.engine import Engine
    from anomalib.deploy import ExportType
    HAS_ANOMALIB = True
except ImportError:
    HAS_ANOMALIB = False

def train_yolo(recall_priority=True):
    """
    启动 YOLO 模型训练

    参数:
        recall_priority: 如果为 True，优先保证高召回率（降低假负例）
                        这适合质检场景：宁可多检测缺陷，也不能漏检
    """
    print("--- 开始 YOLO 模型训练 ---")
    # 1. 加载预训练模型作为起点 (迁移学习)
    model = YOLO("yolov8n.pt")

    # 2. 确定数据集配置文件的绝对路径
    yaml_path = os.path.abspath("datasets/metal_defects/data.yaml")

    # 3. 准备训练参数
    train_params = {
        "data": yaml_path,
        "epochs": 100,              # 训练100轮
        "imgsz": 640,               # 图像输入尺寸
        "batch": 16,                # 每批次处理16张图
        "device": "mps",            # Apple Metal 加速
        "project": "industrial_runs",
        "name": "metal_v1_recall_optimized" if recall_priority else "metal_v1",
        "exist_ok": True,

        # === 关键参数：优化为高召回率 ===
        "loss": "focal" if recall_priority else "standard",
        "cls": 1.5 if recall_priority else 1.0,
        "box": 7.5,
        "dfl": 1.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 10,
        "translate": 0.1,
        "scale": 0.5,
        "flipud": 0.5,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.1,
        "warmup_epochs": 3,
        "patience": 15,
        "lr0": 0.01,
    }

    # 4. 开始训练
    print(f"启动训练... 模式: {'高召回率优先' if recall_priority else '标准'}")
    results = model.train(**train_params)
    print(f"YOLO 训练完成。模型保存在: {results.save_dir}")
    return results

def train_efficientad(dataset_path="datasets/metal_defects", export_path="models/efficientad"):
    """
    启动 EfficientAD 模型训练 (基于 Anomalib 实现)

    EfficientAD 适合无监督场景，只需要良品数据即可训练。
    """
    print("--- 开始 EfficientAD 模型训练 ---")
    if not HAS_ANOMALIB:
        print("[错误] 未检测到 anomalib 库。")
        print("请运行: pip install anomalib")
        return

    # 1. 准备数据
    # 假设数据集结构符合 Anomalib Folder 格式:
    # dataset_path/
    #   ├── train/
    #   │   └── good/ (只有良品图)
    #   ├── test/
    #   │   ├── good/
    #   │   └── bad/ (可选，用于验证)
    datamodule = Folder(
        name="metal_defects",
        root=dataset_path,
        normal_dir="train/good",
        test_split_mode="from_dir",
        train_batch_size=1,
        eval_batch_size=1,
        num_workers=4,
        image_size=(256, 256)
    )

    # 2. 初始化模型
    model = EfficientAd(
        teacher_out_channels=384,
        latent_channels=64
    )

    # 3. 初始化引擎
    engine = Engine(
        max_epochs=200,
        devices=1,
        accelerator="auto" # 在 Mac 上会自动尝试 mps
    )

    # 4. 开始训练
    print("启动 EfficientAD 训练 (无监督模式)...")
    engine.fit(model=model, datamodule=datamodule)

    # 5. 导出模型为 TorchScript (方便推理使用)
    os.makedirs(export_path, exist_ok=True)
    exported_model_path = engine.export(
        model=model,
        export_type=ExportType.TORCHSCRIPT,
        export_root=export_path
    )

    print(f"EfficientAD 训练与导出完成。模型位置: {exported_model_path}")
    print(f"请在 app/core/config.py 中配置 EFFICIENTAD_MODEL_PATH = '{exported_model_path}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="工业缺陷检测模型训练脚本")
    parser.add_argument("--type", type=str, default="yolo", choices=["yolo", "efficientad"], 
                        help="训练模型类型: yolo 或 efficientad (默认: yolo)")
    parser.add_argument("--recall", action="store_true", help="YOLO 模式下是否开启高召回率优先 (默认: False)")
    
    args = parser.parse_args()

    if args.type == "yolo":
        train_yolo(recall_priority=args.recall)
    elif args.type == "efficientad":
        train_efficientad()

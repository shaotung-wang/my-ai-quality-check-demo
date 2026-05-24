import os
from anomalib.data import MVTec
from anomalib.models import EfficientAd
from anomalib.engine import Engine


def start_training():
    print("🚀 初始化工业视觉数据管道...")

    # 1. 配置数据模块 (DataModule)
    # 自动读取指定类别下的 train/good 数据进行训练
    datamodule = MVTec(
        root="./",  # 数据集的根目录
        category="My_Metal_Project",  # 你的具体工件类别（文件夹名）
        train_batch_size=8,  # 批次大小，可根据内存情况调大到 16 或 32
        eval_batch_size=8,
        image_size=(256, 256),  # 强制统一输入尺寸
        num_workers=4  # 异步加载数据的线程数
    )

    print("🧠 构建 EfficientAD 模型实例...")
    # 2. 初始化 EfficientAD 模型
    # model_size="S" (Small) 拥有极少的参数量，能实现毫秒级推理
    model = EfficientAd(
        model_size="S",
        padding=False  # 对于纯净背景，无需开启额外 padding
    )

    print("⚙️ 配置硬件加速引擎...")
    # 3. 初始化训练引擎 (Engine)
    engine = Engine(
        max_epochs=100,  # 训练轮数。通常在 50-100 轮左右即可收敛
        accelerator="mps",  # 开启 Apple Silicon 底层 GPU 硬件加速
        devices=1,  # 使用单卡/单节点计算
        default_root_dir="./results",  # 模型权重、日志和可视化结果的保存路径
        check_val_every_n_epoch=5  # 每隔 5 轮评估一次模型状态
    )

    print("🔥 开始执行知识蒸馏与特征学习，请稍候...")
    # 4. 启动训练
    engine.fit(datamodule=datamodule, model=model)

    print("✅ 训练流水线执行完毕！")
    print("📁 模型权重 (Checkpoints) 已自动保存在 ./results 目录下。")


if __name__ == "__main__":
    start_training()
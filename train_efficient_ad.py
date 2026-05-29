import os

# macOS 上 python.org 安装的 Python 3.13 缺 CA 根证书目录，
# anomalib 在下载 EfficientAD 预训练权重 / imagenette 时会 SSL 校验失败。
# 把 SSL_CERT_FILE 指到 venv 里的 certifi 即可，必须在 import 任何会发起 HTTPS
# 请求的模块之前设置。
import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("SSL_CERT_DIR", os.path.dirname(certifi.where()))

from anomalib.data import Folder
from anomalib.models import EfficientAd
from anomalib.engine import Engine


def start_training():
    print("🚀 初始化工业视觉数据管道...")

    # 1. 配置数据模块 (DataModule)
    # 使用 Folder 适配你的自定义文件夹结构
    data_root = "./My_Metal_Project"
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"数据根目录未找到: {data_root}")
    # 当前只有 train/good，没有 test 异常样本：
    #   - 不传 abnormal_dir
    #   - test_split_mode="none"  跳过测试集构建
    #   - val_split_mode="from_train" 从训练集切一小部分做验证，
    #     EfficientAD 的分位数归一化阶段需要验证数据，否则会报空数据集
    datamodule = Folder(
        name="My_Metal_Project",
        root=data_root,
        normal_dir="train/good",       # 训练用的正常图像
        train_batch_size=1,            # EfficientAD 要求 batch_size=1
        eval_batch_size=8,
        num_workers=4,
        val_split_mode="from_train",
        val_split_ratio=0.1,
        test_split_mode="none",
    )

    print("🧠 构建 EfficientAD 模型实例...")
    # 2. 初始化 EfficientAD 模型
    # model_size 仅支持 "small" 或 "medium"
    model = EfficientAd(
        model_size="small",
        padding=False  # 对于纯净背景，无需开启额外 padding
    )

    print("⚙️ 配置硬件加速引擎...")
    # 3. 初始化训练引擎 (Engine)
    engine = Engine(
        max_epochs=100,  # 训练轮数。通常在 50-100 轮左右即可收敛
        accelerator="auto",  # 自动选择最佳设备（MPS/GPU/CPU）
        devices=1,  # 使用单卡/单节点计算
        default_root_dir="./results",  # 模型权重、日志和可视化结果的保存路径
        # 没有验证集时不要定期评估，否则会报错
    )

    print("🔥 开始执行知识蒸馏与特征学习，请稍候...")
    # 4. 启动训练
    engine.fit(datamodule=datamodule, model=model)

    print("✅ 训练流水线执行完毕！")
    print("📁 模型权重 (Checkpoints) 已自动保存在 ./results 目录下。")


if __name__ == "__main__":
    start_training()
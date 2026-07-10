import os

# macOS 上 python.org 安装的 Python 3.13 缺 CA 根证书目录，
# anomalib 在下载预训练权重 / imagenette 时会 SSL 校验失败。
# 把 SSL_CERT_FILE 指到 venv 里的 certifi 即可，必须在 import 任何会发起 HTTPS
# 请求的模块之前设置。
import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("SSL_CERT_DIR", os.path.dirname(certifi.where()))

from anomalib.data import Folder
from anomalib.models import Padim
from anomalib.engine import Engine


def start_training():
    print("🚀 初始化工业视觉数据管道...")

    # 1. 配置数据模块 (DataModule)
    # 使用 Folder 适配你的自定义文件夹结构（MVTec AD 格式）
    data_root = "../batch_dataset_padim"
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"数据根目录未找到: {data_root}")
    # 当前只有 train/good，没有 test 异常样本：
    #   - 不传 abnormal_dir
    #   - test_split_mode="none"  跳过测试集构建
    #   - val_split_mode="from_train" 从训练集切一小部分做验证
    datamodule = Folder(
        name="batch_dataset_padim",
        root=data_root,
        normal_dir="train/good",       # 训练用的正常图像
        train_batch_size=8,            # PaDiM 无 batch_size=1 限制，可适当增大
        eval_batch_size=8,
        num_workers=4,
        val_split_mode="from_train",
        val_split_ratio=0.1,
        test_split_mode="none",
    )

    print("🧠 构建 PaDiM 模型实例...")
    # 2. 初始化 PaDiM 模型
    #    - backbone: 特征提取骨干网络，resnet18 在速度与精度之间取得较好平衡
    #    - layers: 使用 layer1/layer2/layer3 三层特征拼接，同时保留低层纹理与高层语义
    #    - n_features: 随机降维后的特征维度，默认 None 为自动选择
    model = Padim(
        backbone="resnet18",
        pre_trained=True,
        layers=["layer1", "layer2", "layer3"],
        n_features=None,
    )

    print("⚙️ 配置硬件加速引擎...")
    # 3. 初始化训练引擎 (Engine)
    #    PaDiM 无需训练神经网络，Engine.fit() 实际执行的是：
    #      特征提取 → 逐位置高斯分布(μ, Σ)计算 → 保存模型参数
    #    因此 max_epochs 设为 1 即可完成特征建模
    engine = Engine(
        max_epochs=1,               # PaDiM 无需多轮训练，1 轮即可完成特征建模
        accelerator="auto",         # 自动选择最佳设备（MPS/GPU/CPU）
        devices=1,                  # 使用单卡/单节点计算
        default_root_dir="../results_padim",  # 模型权重、日志和可视化结果的保存路径
    )

    print("🔥 开始特征提取与高斯分布建模，请稍候...")
    # 4. 启动训练（实际为特征建模，非神经网络训练）
    engine.fit(datamodule=datamodule, model=model)

    print("✅ 训练流水线执行完毕！")
    print("📁 模型权重 (Checkpoints) 已自动保存在 ./results 目录下。")
    print("💡 PaDiM 模型在推理时：每个 patch 位置仅需一次马氏距离计算，")
    print("   无需内存库搜索，非常适合 M2 Mac mini 的 CPU 推理。")


if __name__ == "__main__":
    start_training()
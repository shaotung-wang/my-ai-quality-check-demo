from ultralytics import YOLO
import os

def start_training():
    # 1. 加载预训练模型作为起点 (迁移学习)
    # 使用 'n' (nano) 版本，因为它在边缘盒子上跑得最快
    model = YOLO("yolov8n.pt")

    # 2. 确定数据集配置文件的绝对路径
    yaml_path = os.path.abspath("datasets/metal_defects/data.yaml")

    # 3. 开始训练
    print("启动训练...")
    model.train(
        data=yaml_path,
        epochs=100,            # 训练100轮，工业模型通常需要这个量级
        imgsz=640,             # 图像输入尺寸
        batch=16,              # 每批次处理16张图
        device="mps",          # 强制使用 Apple Metal 加速
        project="industrial_runs", # 结果保存的主目录
        name="metal_v1",       # 本次训练的任务名
        exist_ok=True          # 如果文件夹存在则覆盖
    )

if __name__ == "__main__":
    start_training()
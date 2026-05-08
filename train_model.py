from ultralytics import YOLO
import os

def start_training(recall_priority=True):
    """
    启动模型训练
    
    参数:
        recall_priority: 如果为True，优先保证高召回率（降低假负例），可接受假正例
                        这适合质检场景：宁可多检测缺陷，也不能漏检
    """
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
        # 使用焦点损失，对难分样本增加权重
        "loss": "focal" if recall_priority else "standard",
        
        # 对缺陷类别增加权重（cls权重），降低背景误检
        # 这会让模型更倾向于检测缺陷，而不是漏检
        "cls": 1.5 if recall_priority else 1.0,
        
        # 增加目标框匹配的权重
        "box": 7.5 if recall_priority else 7.5,
        
        # IoU匹配权重
        "dfl": 1.5 if recall_priority else 1.5,
        
        # 其他重要参数
        "hsv_h": 0.015,             # HSV-色调调整
        "hsv_s": 0.7,               # HSV-饱和度调整
        "hsv_v": 0.4,               # HSV-值调整
        "degrees": 10,              # 数据增强：旋转
        "translate": 0.1,           # 数据增强：平移
        "scale": 0.5,               # 数据增强：缩放
        "flipud": 0.5,              # 数据增强：上下翻转
        "fliplr": 0.5,              # 数据增强：左右翻转
        "mosaic": 1.0,              # 马赛克增强：混合多张图
        "mixup": 0.1,               # Mixup数据增强
        "momentum": 0.937,          # SGD动量
        "weight_decay": 0.0005,     # L2正则化
        "warmup_epochs": 3,         # 预热轮数
        "patience": 15,             # 早停耐心值
        "lr0": 0.01,                # 初始学习率
        "lrf": 0.01,                # 最终学习率系数
    }
    
    # 4. 开始训练
    print("启动训练...")
    if recall_priority:
        print("✓ 高召回率优先模式：优化为最小化假负例")
        print("  - 使用焦点损失(Focal Loss)处理难分样本")
        print("  - 提高类别权重，增强缺陷检测能力")
        print("  - 可能增加假正例，但确保不漏检缺陷")
    
    results = model.train(**train_params)
    
    return results

if __name__ == "__main__":
    start_training()
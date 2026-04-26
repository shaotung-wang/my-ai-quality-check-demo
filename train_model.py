from ultralytics import YOLO

def main():
    # 1. 加载官方的基础预训练模型
    # n 代表 nano，是最轻量极快的版本，适合边缘计算盒子
    model = YOLO("yolov8n.pt")

    # 2. 开始训练
    print("开始训练专属模型...")
    results = model.train(
        data="datasets/data.yaml",  # 您的数据集配置文件路径
        epochs=100,                 # 训练轮数 (AI 看这批图片的次数，推荐先设 100)
        imgsz=640,                  # 图片缩放尺寸 (默认 640 即可)
        device="mps",               # 核心：调用苹果 M4 芯片的底层 GPU 加速
        batch=16,                   # 每次塞进内存的图片数量 (M4 Pro 内存大，16 毫无压力)
        project="my_inspection",    # 训练结果保存的文件夹名称
        name="defect_model_v1"      # 本次训练的版本号
    )
    print("训练完成！")

if __name__ == '__main__':
    main()
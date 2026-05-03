# config.py

# ================= 硬件与模型配置 =================
MODEL_PATH = "runs/detect/industrial_runs/metal_v1/weights/best.pt"  # 核心模型文件
DEVICE = "mps"             # Mac M4 Pro 硬件加速引擎
TARGET_CLASSES = None      # 识别目标：67 代表 COCO 数据集中的 cell phone
CAMERA_ID = 0              # 相机索引：0 通常代表 Mac 自带摄像头

# ================= 业务参数配置 =================
CONFIDENCE_THRESHOLD = 0.7 # 置信度阈值 (保留，后续换真实模型时有用)
QUEUE_MAX_LENGTH = 1       # 帧队列最大长度 (防止画面延迟)
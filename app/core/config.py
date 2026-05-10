# config for app.core
# 假设 6 台相机的系统索引
CAMERA_LIST = [0, 1, 2, 3, 4, 5]
CAMERA_ID = 0  # 默认摄像头ID

# 模型配置：支持 'efficientad'（首选）和 'yolo' 回退
MODEL_TYPE = "efficientad"  # 'efficientad' or 'yolo'
# 当使用 EfficientAD 时，期望为导出的模型文件（TorchScript/ONNX）或训练目录
EFFICIENTAD_MODEL_PATH = "models/efficientad/efficientad_export.pt"
# 当回退到 YOLO 时，使用现有的 ultralytics 权重
YOLO_MODEL_PATH = "runs/detect/industrial_runs/metal_v1_recall_optimized/weights/best.pt"

# 推理设备（在 M2 上建议使用 'mps'；可回退到 'cpu'）
DEVICE = "mps"

# 性能参数
IMG_SIZE = 512        # 推理输入尺寸，512 为性能/精度折中
FPS_LIMIT = 10        # 单相机每秒采样率（目标）
BATCH_SIZE = 8        # 推理时的 batch 大小（基于内存可调整）

# === 质量检查模式配置 ===
# 为了确保OK判定的产品100%没有瑕疵，我们采用不对称的阈值策略
QUALITY_CHECK_MODE = True  # 启用质量检查优先模式

# 基础置信度阈值（缺陷检测的主要阈值）
# 使用较低的阈值以捕捉更多潜在的缺陷，降低漏检率
CONF_THRESHOLD = 0.15  # 降低到0.15以提高召回率

# 高置信度检测阈值（仅当置信度非常高时才跳过）
# 只有当检测置信度非常高时，才认为这是真正可靠的
HIGH_CONF_THRESHOLD = 0.50  # 非常高的置信度阈值

# 缺陷框数量阈值
# 如果检测到缺陷框数量超过此值，肯定判定为NG
MIN_DEFECT_BOX_COUNT = 1  # 只要检测到1个缺陷框，就判定为NG

# 缺陷框面积阈值（相对于图像面积的比例）
# 防止极小的误检影响判定
MIN_DEFECT_AREA_RATIO = 0.001  # 最小缺陷面积为图像面积的0.1%

# 每帧的动态调查：连续帧中有多少帧检测到缺陷才确认
# 用于防止单帧噪声导致误判
DEFECT_CONFIRMATION_FRAMES = 1  # 1帧确认（可改为3以增加可靠性）

TARGET_CLASSES = None  # 目标类别，None表示所有类别
QUEUE_MAX_LENGTH = 32  # 队列最大长度（增加以支撑并��摄像头）

# 聚合策略（将多张图片的分数合成为轴级决策）
# strategy: 'max'|'topk_mean'|'quantile'
AGGREGATION_STRATEGY = 'max'
AGG_WINDOW_SIZE = 12   # 每根轴收集的帧数窗口（可根据旋转速度调整）
AGG_TOPK = 3           # 当 strategy='topk_mean' 时使用
AGG_QUANTILE = 0.9     # 当 strategy='quantile' 时使用（例如 0.9 表示 90% 分位数）

# 元数据字段名（如果输入流携带元数据）
META_CAM_FIELD = 'cam'
META_SHAFT_FIELD = 'shaft'

# === 调试和日志参数 ===
VERBOSE_MODE = True  # 启用详细日志
SAVE_HIGH_CONF_DETECTIONS = False  # 是否保存高置信度检测结果


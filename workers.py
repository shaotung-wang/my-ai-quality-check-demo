# workers.py
import cv2
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage
from ultralytics import YOLO
import config
from collections import deque
import numpy as np


def calculate_box_area(box):
    """计算检测框面积"""
    x1, y1, x2, y2 = box.xyxy[0]
    return float((x2 - x1) * (y2 - y1))


def is_valid_defect(box, image_height, image_width):
    """
    判断检测框是否有效

    参数:
        box: YOLO检测框对象
        image_height: 图像高度
        image_width: 图像宽度

    返回:
        True如果框有效，False否则
    """
    conf = float(box.conf[0])
    box_area = calculate_box_area(box)
    image_area = image_height * image_width
    area_ratio = box_area / image_area

    # 检查面积是否过小（可能是噪声）
    if area_ratio < config.MIN_DEFECT_AREA_RATIO:
        if config.VERBOSE_MODE:
            print(f"  ⚠ 检测框面积过小 ({area_ratio:.6f}), 置信度={conf:.3f} - 忽略")
        return False

    # 检查置信度
    if conf < config.CONF_THRESHOLD:
        if config.VERBOSE_MODE:
            print(f"  ⚠ 置信度过低 ({conf:.3f}) - 忽略")
        return False

    return True


def analyze_defects_in_frame(results, frame_height, frame_width):
    """
    分析单帧图像中的缺陷

    返回:
        (has_defect_bool, high_conf_detections_list, all_detections_list)
    """
    boxes = results[0].boxes

    if len(boxes) == 0:
        return False, [], []

    high_conf_detections = []
    all_detections = []

    for box in boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        box_area = calculate_box_area(box)

        all_detections.append({
            'conf': conf,
            'class': cls_id,
            'area': box_area
        })

        # 检查是否为有效的缺陷
        if is_valid_defect(box, frame_height, frame_width):
            high_conf_detections.append({
                'conf': conf,
                'class': cls_id,
                'area': box_area
            })

    has_defect = len(high_conf_detections) >= config.MIN_DEFECT_BOX_COUNT

    return has_defect, high_conf_detections, all_detections


class CameraWorker(QThread):
    """专门负责与硬件摄像头通信的子线程"""
    error_signal = Signal(str)

    def __init__(self, frame_queue):
        super().__init__()
        self.frame_queue = frame_queue
        self.running = True
        self.cap = None

    def run(self):
        self.cap = cv2.VideoCapture(config.CAMERA_ID)
        if not self.cap.isOpened():
            self.error_signal.emit("CAMERA ERROR: 无法连接硬件")
            return

        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            # 将最新帧塞入队列，队列满时自动丢弃最老的一帧
            self.frame_queue.append(frame)

        if self.cap:
            self.cap.release()

    def stop(self):
        self.running = False


class InferenceWorker(QThread):
    """专门负责压榨 M4 Pro 算力进行 AI 推理的子线程"""
    result_ready = Signal(QImage, bool)

    def __init__(self, frame_queue):
        super().__init__()
        self.frame_queue = frame_queue
        self.running = True

        print("正在加载 AI 模型...")
        self.model = YOLO(config.MODEL_PATH)
        print("模型加载完成！")

    def run(self):
        while self.running:
            try:
                # 1. 尝试获取最新帧
                if len(self.frame_queue) > 0:
                    frame = self.frame_queue.popleft()
                else:
                    self.msleep(10)  # 队列为空时休眠，降低 CPU 占用
                    continue

                # 2. 执行模型推理
                results = self.model(
                    frame,
                    device=config.DEVICE,
                    classes=config.TARGET_CLASSES,
                    conf=config.CONF_THRESHOLD,
                    verbose=False
                )

                # 3. 高级缺陷分析（质量检查优先模式）
                frame_height, frame_width = frame.shape[:2]
                has_defect, valid_detections, all_detections = analyze_defects_in_frame(
                    results, frame_height, frame_width
                )

                # 判定逻辑：
                # - 如果检测到有效的缺陷框 → NG（有瑕疵）
                # - 如果没检测到有效缺陷框 → OK（没瑕疵）
                is_ng = has_defect

                if config.VERBOSE_MODE and len(all_detections) > 0:
                    print(f"[推理结果] 总检测数={len(all_detections)}, 有效缺陷={len(valid_detections)}, 判定={'NG' if is_ng else 'OK'}")

                # 4. 图像后处理与画框
                res_frame = results[0].plot()
                h, w, ch = res_frame.shape

                if is_ng:
                    # NG：显示红色
                    label = f"NG (Defect Detected: {len(valid_detections)})"
                    color = (0, 0, 255)  # BGR: 红色
                    if config.VERBOSE_MODE:
                        print(f"  🔴 判定为NG：检测到{len(valid_detections)}个有效缺陷")
                else:
                    # OK：显示绿色
                    label = "OK (No Defect)"
                    color = (0, 255, 0)  # BGR: 绿色
                    if config.VERBOSE_MODE:
                        print(f"  🟢 判定为OK：未检测到有效缺陷")

                cv2.putText(res_frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

                # 5. 格式转换与信号发射
                bytes_per_line = ch * w
                qt_img = QImage(res_frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
                self.result_ready.emit(qt_img, is_ng)

            except Exception as e:
                print(f"推理引擎异常: {e}")

    def stop(self):
        self.running = False


class BatchInferenceWorker(QThread):
    # 发送：(相机ID列表, QImage列表, NG结果列表)
    batch_result_ready = Signal(list, list, list)

    def __init__(self, queues):
        super().__init__()
        self.queues = queues
        self.running = True
        self.model = YOLO(config.MODEL_PATH)

    def run(self):
        while self.running:
            batch_frames = []
            active_ids = []

            # 1. 尝试从队列中收割图像
            for i, q in enumerate(self.queues):
                if not q.empty():
                    batch_frames.append(q.get())
                    active_ids.append(i)

            if not batch_frames:
                self.msleep(5)
                continue

            # 2. 核心：Batch 推理，一次性处理多张
            results = self.model(batch_frames, device=config.DEVICE, conf=config.CONF_THRESHOLD, verbose=False)

            qt_images = []
            ng_results = []

            # 3. 后处理：对每一帧进行高级缺陷分析
            for idx, res in enumerate(results):
                # 获取原始帧以计算面积比
                frame = batch_frames[idx]
                frame_height, frame_width = frame.shape[:2]

                # 高级缺陷分析
                has_defect, valid_detections, all_detections = analyze_defects_in_frame(
                    [res], frame_height, frame_width
                )

                is_ng = has_defect
                ng_results.append(is_ng)

                # 绘制与转换
                plot_img = res.plot()
                h, w, ch = plot_img.shape

                # 添加判定标签
                label = f"NG ({len(valid_detections)} defects)" if is_ng else "OK"
                color = (0, 0, 255) if is_ng else (0, 255, 0)
                cv2.putText(plot_img, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

                qt_img = QImage(plot_img.data, w, h, ch * w, QImage.Format_BGR888)
                qt_images.append(qt_img)

            self.batch_result_ready.emit(active_ids, qt_images, ng_results)
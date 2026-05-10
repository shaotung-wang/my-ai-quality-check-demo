"""
app.core.workers

Copied from top-level workers.py but adjusted to import app.core.config and app.core.inference
"""
import cv2
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage
import numpy as np

from app.core import config as core_config
from app.core.inference import ModelInfer


def analyze_score(score: float):
    """基于模型 score 做简单缺陷判断（占位）。

    当前推理层返回单个浮点 score（越高越可疑）。如果 score >= CONF_THRESHOLD
    则认为存在缺陷（NG）。
    返回 (has_defect, details)
    """
    has_defect = float(score) >= core_config.CONF_THRESHOLD
    details = {'score': float(score)}
    return has_defect, details


class CameraWorker(QThread):
    """专门负责与硬件摄像头通信的子线程"""
    error_signal = Signal(str)

    def __init__(self, frame_queue):
        super().__init__()
        self.frame_queue = frame_queue
        self.running = True
        self.cap = None

    def run(self):
        self.cap = cv2.VideoCapture(core_config.CAMERA_ID)
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

        print("正在加载 AI 推理模块...")
        # 使用 inference.ModelInfer 作为推理后端（EfficientAD placeholder）
        self.infer = ModelInfer(device=core_config.DEVICE, batch_size=core_config.BATCH_SIZE)
        print("推理模块准备就绪（占位实现）")

    def run(self):
        while self.running:
            try:
                # 1. 尝试获取最新帧
                if len(self.frame_queue) > 0:
                    frame = self.frame_queue.popleft()
                else:
                    self.msleep(10)  # 队列为空时休眠，降低 CPU 占用
                    continue

                # 2. 执行模型推理（占位实现返回 score）
                scores, maps = self.infer.infer_batch([frame], img_size=core_config.IMG_SIZE)
                score = scores[0] if scores else 0.0

                # 3. 基于 score 的简单缺陷分析
                is_ng, details = analyze_score(score)

                if core_config.VERBOSE_MODE:
                    print(f"[推理结果] score={score:.4f}, 判定={'NG' if is_ng else 'OK'}")

                # 4. 绘制判定信息在原始帧上
                res_frame = frame.copy()
                h, w, ch = res_frame.shape

                if is_ng:
                    label = f"NG (score={score:.3f})"
                    color = (0, 0, 255)
                else:
                    label = f"OK (score={score:.3f})"
                    color = (0, 255, 0)

                cv2.putText(res_frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

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
        # 使用 inference 后端
        self.infer = ModelInfer(device=core_config.DEVICE, batch_size=core_config.BATCH_SIZE)

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

            # 2. 核心：Batch 推理（占位实现）
            scores, maps = self.infer.infer_batch(batch_frames, img_size=core_config.IMG_SIZE)

            qt_images = []
            ng_results = []

            # 3. 后处理：对每一帧基于 score 做判定
            for idx, score in enumerate(scores):
                frame = batch_frames[idx]
                is_ng, details = analyze_score(score)
                ng_results.append(is_ng)

                # 绘制与转换
                plot_img = frame.copy()
                h, w, ch = plot_img.shape
                label = f"NG (s={score:.3f})" if is_ng else f"OK (s={score:.3f})"
                color = (0, 0, 255) if is_ng else (0, 255, 0)
                cv2.putText(plot_img, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

                qt_img = QImage(plot_img.data, w, h, ch * w, QImage.Format_BGR888)
                qt_images.append(qt_img)

            self.batch_result_ready.emit(active_ids, qt_images, ng_results)


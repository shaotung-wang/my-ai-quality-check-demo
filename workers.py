# workers.py
import cv2
import time
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage
from ultralytics import YOLO
import config


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
                    conf=config.CONFIDENCE_THRESHOLD,
                    verbose=False
                )

                # 3. 结果判定
                boxes = results[0].boxes
                is_ng = len(boxes) > 0

                # 4. 图像后处理与画框
                res_frame = results[0].plot()
                h, w, ch = res_frame.shape

                if is_ng:
                    cv2.putText(res_frame, "NG (Defect)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                else:
                    cv2.putText(res_frame, "OK", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

                # 5. 格式转换与信号发射
                bytes_per_line = ch * w
                qt_img = QImage(res_frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
                self.result_ready.emit(qt_img, is_ng)

            except Exception as e:
                print(f"推理引擎异常: {e}")

    def stop(self):
        self.running = False
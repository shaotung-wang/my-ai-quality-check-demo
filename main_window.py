# main_window.py
from collections import deque
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QImage, QPixmap, QFont

import config
from workers import CameraWorker, InferenceWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("纯视觉 AI 质检站 Demo")
        self.resize(900, 700)
        self.setStyleSheet("background-color: #2b2b2b; color: white;")

        # 核心缓冲池：连接相机采集与 AI 推理的桥梁
        self.frame_queue = deque(maxlen=config.QUEUE_MAX_LENGTH)

        # 声明后台 Worker（暂时不启动）
        self.camera_worker = None
        self.inference_worker = None

        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 状态指示灯区域
        status_layout = QHBoxLayout()
        self.status_label = QLabel("SYSTEM IDLE")
        self.status_label.setFont(QFont("Arial", 24, QFont.Bold))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("background-color: gray; color: white; border-radius: 10px; padding: 10px;")
        status_layout.addWidget(self.status_label)
        main_layout.addLayout(status_layout)

        # 视频画面区域
        self.video_label = QLabel("点击下方按钮开启系统\n\n(将手机拿入画面模拟瑕疵报警)")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #1e1e1e; border: 2px dashed #555;")
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.video_label.setMinimumSize(640, 480)  # 给一个基础底线大小
        main_layout.addWidget(self.video_label, stretch=1)

        # 控制按钮
        self.btn_control = QPushButton("▶ 开启系统")
        self.btn_control.setFont(QFont("Arial", 14, QFont.Bold))
        self.btn_control.setStyleSheet("background-color: #007acc; padding: 15px; border-radius: 5px;")
        self.btn_control.clicked.connect(self.toggle_system)
        main_layout.addWidget(self.btn_control)

    def toggle_system(self):
        # 依据按钮文本状态来决定是“开启”还是“关闭”
        if "开启" in self.btn_control.text():
            self.start_workers()
        else:
            self.stop_workers()

    def start_workers(self):
        self.frame_queue.clear()

        # 1. 初始化并启动相机采集线程
        self.camera_worker = CameraWorker(self.frame_queue)
        self.camera_worker.error_signal.connect(self.handle_camera_error)
        self.camera_worker.start()

        # 2. 初始化并启动 AI 推理线程
        self.inference_worker = InferenceWorker(self.frame_queue)
        self.inference_worker.result_ready.connect(self.update_display)
        self.inference_worker.start()

        # 3. 更新 UI 状态
        self.btn_control.setText("■ 停止系统")
        self.btn_control.setStyleSheet("background-color: #cc4400; padding: 15px; border-radius: 5px;")

    def stop_workers(self):
        # 安全停止所有后台线程
        if self.inference_worker:
            self.inference_worker.stop()
            self.inference_worker.wait()

        if self.camera_worker:
            self.camera_worker.stop()
            self.camera_worker.wait()

        # 更新 UI 状态
        self.video_label.setText("系统已停止")
        self.video_label.setPixmap(QPixmap())
        self.status_label.setText("SYSTEM IDLE")
        self.status_label.setStyleSheet("background-color: gray; color: white; padding: 10px;")
        self.btn_control.setText("▶ 开启系统")
        self.btn_control.setStyleSheet("background-color: #007acc; padding: 15px; border-radius: 5px;")

    @Slot(str)
    def handle_camera_error(self, error_msg):
        """处理相机启动失败的情况"""
        self.status_label.setText(error_msg)
        self.status_label.setStyleSheet("background-color: red;")
        self.stop_workers()

    @Slot(QImage, bool)
    def update_display(self, qt_img, is_ng):
        """接收 AI 线程信号，更新主界面画面与警报灯"""
        self.video_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation))

        if is_ng:
            self.status_label.setText("⚠ NG - DEFECT DETECTED")
            self.status_label.setStyleSheet("background-color: red; color: white; border-radius: 10px; padding: 10px;")
        else:
            self.status_label.setText("✔ OK")
            self.status_label.setStyleSheet(
                "background-color: green; color: white; border-radius: 10px; padding: 10px;")
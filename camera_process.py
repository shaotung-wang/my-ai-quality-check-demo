# camera_process.py
import cv2
import time


def camera_capture_task(cam_id, queue):
    """运行在独立物理进程中，不受 GIL 限制"""
    cap = cv2.VideoCapture(cam_id)
    # 强制 MJPEG 节省总线带宽
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    last_time = time.time()
    while True:
        ret, frame = cap.read()
        if ret:
            # 严格限速，防止内存堆积
            current_time = time.time()
            if (current_time - last_time) < (1.0 / 10):
                continue

            # 队列只保留最新一帧
            if not queue.full():
                queue.put(frame)
            else:
                try:
                    queue.get_nowait()
                    queue.put(frame)
                except:
                    pass
            last_time = current_time
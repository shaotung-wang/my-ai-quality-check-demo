"""
app.core.inference

Copied from top-level inference.py: EfficientAD-centered inference placeholder.
"""
from typing import List, Tuple, Optional
import os
import numpy as np
from app.core import config as core_config


class ModelInfer:
    """Placeholder inference wrapper for EfficientAD.

    - If core_config.EFFICIENTAD_MODEL_PATH exists, you can implement loading logic here.
    - Otherwise returns zeros safely so the pipeline can run without YOLO.
    """
    def __init__(self, device: str = None, batch_size: int = None):
        self.device = device or core_config.DEVICE
        self.batch_size = batch_size or core_config.BATCH_SIZE
        self.model_type = core_config.MODEL_TYPE
        self.model = None
        self._load_model()

    def _load_model(self):
        path = core_config.EFFICIENTAD_MODEL_PATH
        if os.path.exists(path):
            # TODO: load TorchScript/ONNX EfficientAD model here
            print(f"[inference] Found EfficientAD model at {path}, loader not implemented. Using dummy infer.")
            self.model = None
        else:
            print(f"[inference] EfficientAD model not found at {path}. Using dummy infer (scores=0.0).")
            self.model = None

    def infer_batch(self, frames: List[np.ndarray], img_size: int = None) -> Tuple[List[float], List[Optional[np.ndarray]]]:
        """Infer a batch of BGR frames.

        Returns:
            scores: list of floats (anomaly score per image)
            maps: list of None or np.ndarray (anomaly maps if available)
        """
        n = len(frames)
        scores = [0.0 for _ in range(n)]
        maps = [None for _ in range(n)]
        return scores, maps


if __name__ == "__main__":
    import cv2

    mi = ModelInfer()
    img = cv2.imread('datasets/metal_defects/train/images/img_02_3402576500_00001_jpg.rf.ZKF2G2xS0VnLx6xI4pRI.jpg')
    if img is None:
        print('示例图片不存在，跳过自测')
    else:
        s, m = mi.infer_batch([img])
        print('scores=', s)


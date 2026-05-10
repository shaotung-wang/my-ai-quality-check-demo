"""app.workers - wrapper for workers module

Re-exports `CameraWorker`, `InferenceWorker`, `BatchInferenceWorker` from
the top-level `workers.py` module so GUI code can import from
`app.workers` instead of flat top-level modules.
"""
try:
    from app.core.workers import CameraWorker as _CameraWorker, InferenceWorker as _InferenceWorker, BatchInferenceWorker as _BatchInferenceWorker
except Exception:
    _CameraWorker = _InferenceWorker = _BatchInferenceWorker = None

CameraWorker = _CameraWorker
InferenceWorker = _InferenceWorker
BatchInferenceWorker = _BatchInferenceWorker

__all__ = ['CameraWorker', 'InferenceWorker', 'BatchInferenceWorker']


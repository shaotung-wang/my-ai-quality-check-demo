"""app.inference - wrapper exposing the inference modules under app namespace

This module re-exports the existing top-level `inference` and
`aggregator` modules so callers can use `app.inference` and
`app.inference.aggregator` consistently.
"""
try:
    from app.core.inference import ModelInfer as _ModelInfer
except Exception:
    _ModelInfer = None

try:
    from app.core.aggregator import Aggregator as _Aggregator
except Exception:
    _Aggregator = None

ModelInfer = _ModelInfer
Aggregator = _Aggregator

__all__ = ['ModelInfer', 'Aggregator']


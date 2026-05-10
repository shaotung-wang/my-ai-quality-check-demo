"""app.tools - wrapper for utility scripts like training and demos"""
try:
    from app.core.pipeline_demo import simulate_run as _simulate_run
except Exception:
    _simulate_run = None

try:
    # train_model placeholder moved under app.core if present
    from app.core import train_model as _train_mod
    _start_training = getattr(_train_mod, 'start_training_placeholder', None)
except Exception:
    _start_training = None

pipeline_demo = _simulate_run
start_training = _start_training

__all__ = ['pipeline_demo', 'start_training']


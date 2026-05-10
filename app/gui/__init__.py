"""app.gui - wrapper for GUI components

Exports `MainWindow` from the top-level `main_window.py` module.
"""
try:
    from app.core.main_window import MainWindow as _MainWindow
except Exception:
    _MainWindow = None

MainWindow = _MainWindow

__all__ = ['MainWindow']


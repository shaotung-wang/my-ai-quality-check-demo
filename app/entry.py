"""Application entrypoint placed under app package.

This module mirrors the previous top-level `main.py` behavior but lives
under the `app` package to provide a clearer project layout.
"""
import sys

from PySide6.QtWidgets import QApplication

from app.gui import MainWindow


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()


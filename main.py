import sys

print("1. 成功导入 sys")

try:
    from PySide6.QtWidgets import QApplication
    print("2. 成功导入 PySide6")

    from main_window import MainWindow
    print("3. 成功导入 main_window")
except Exception as e:
    print(f"导入阶段发生错误: {e}")
    sys.exit(1)

if __name__ == "__main__":
    print("4. 进入 main 代码块")
    app = QApplication(sys.argv)

    print("5. 准备实例化 MainWindow")
    window = MainWindow()

    print("6. 准备显示窗口")
    window.show()

    print("7. 进入事件主循环")
    sys.exit(app.exec())
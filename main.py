import os
import PyQt5
import cv2

os.environ["QT_QPA_PLATFORM"] = "xcb"  # Use XCB platform
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins")

import sys
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

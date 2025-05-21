# ui/image_view.py
import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QLabel, QScrollArea, QSizePolicy, QWidget, QVBoxLayout


class ImageView(QScrollArea):
    def __init__(self):
        super().__init__()

        self.container = QWidget()
        self.layout = QVBoxLayout(self.container)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Add title label
        self.title_label = QLabel()
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.layout.addWidget(self.title_label)

        # Image label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setScaledContents(False)
        self.layout.addWidget(self.image_label)

        self.setWidget(self.container)
        self.setWidgetResizable(True)
        self.setAlignment(Qt.AlignCenter)

        self.original_pixmap = None

        # Connect to resize event
        self.resizeEvent = self.on_resize

    def set_title(self, title):
        self.title_label.setText(f"<b>{title}</b>")

    def set_image(self, image):
        if image is None:
            self.image_label.clear()
            self.original_pixmap = None
            return

        # Convert OpenCV image (numpy array) to QImage
        height, width, channels = image.shape
        bytes_per_line = channels * width

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        q_image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.original_pixmap = QPixmap.fromImage(q_image)

        self.resize_pixmap()

    def resize_pixmap(self):
        if self.original_pixmap is None:
            return

        # Get the size of the viewing area
        view_size = self.viewport().size()

        # Scale pixmap to fit in the view while maintaining aspect ratio
        scaled_pixmap = self.original_pixmap.scaled(
            view_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.image_label.setPixmap(scaled_pixmap)

    def on_resize(self, event):
        # When the widget is resized, resize the pixmap too
        self.resize_pixmap()
        # Call the parent class resize event
        super().resizeEvent(event)

# ui/watermark_preview.py
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QSizePolicy, QGroupBox

from utils.image_utils import convert_to_binary_watermark


class WatermarkPreview(QGroupBox):
    def __init__(self):
        super().__init__("Current Watermark")

        layout = QVBoxLayout(self)

        # Image label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setScaledContents(False)
        layout.addWidget(self.image_label)

        self.setMaximumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)

        self.original_pixmap = None
        self.binary_watermark = None

    def set_watermark(self, image):
        if image is None:
            self.image_label.clear()
            self.original_pixmap = None
            self.binary_watermark = None
            return

        # Convert to binary watermark (black and white only)
        self.binary_watermark = convert_to_binary_watermark(image)

        # Create QImage from the binary watermark
        height, width = self.binary_watermark.shape
        q_image = QImage(self.binary_watermark.data, width, height, width, QImage.Format_Grayscale8)

        self.original_pixmap = QPixmap.fromImage(q_image)

        # Scale pixmap to fit in the preview while maintaining aspect ratio
        self.update_display()

    def update_display(self):
        if self.original_pixmap is None:
            return

        scaled_pixmap = self.original_pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.image_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        self.update_display()
        super().resizeEvent(event)

    def get_binary_watermark(self):
        """Return the binary watermark image"""
        return self.binary_watermark
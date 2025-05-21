# ui/split_image_view.py
from PyQt5.QtWidgets import QWidget, QHBoxLayout

from ui.image_view import ImageView


class SplitImageView(QWidget):
    def __init__(self):
        super().__init__()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Original image view
        self.original_view = ImageView()
        layout.addWidget(self.original_view)

        # Processed image view
        self.processed_view = ImageView()
        layout.addWidget(self.processed_view)

        # Add labels
        self.original_view.set_title("Original Image")
        self.processed_view.set_title("Processed Image")

    def set_original_image(self, image):
        self.original_view.set_image(image)

    def set_processed_image(self, image):
        self.processed_view.set_image(image)

    def set_processed_title(self, title):
        self.processed_view.set_title(title)
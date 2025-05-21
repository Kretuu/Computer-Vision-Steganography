# ui/loading_indicator.py
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar
from PyQt5.QtCore import Qt, QThread, pyqtSignal


class LoadingIndicator(QDialog):
    """
    A dialog showing a loading indicator with a message.
    """
    def __init__(self, parent=None, message="Processing..."):
        super().__init__(parent, Qt.WindowStaysOnTopHint | Qt.CustomizeWindowHint)
        self.setWindowTitle("Loading")

        layout = QVBoxLayout(self)

        # Message label
        self.message_label = QLabel(message)
        self.message_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.message_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(0)  # Indeterminate progress
        layout.addWidget(self.progress_bar)

        self.setFixedSize(300, 100)

    def set_message(self, message):
        """Update the loading message."""
        self.message_label.setText(message)



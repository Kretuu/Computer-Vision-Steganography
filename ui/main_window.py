# ui/main_window.py
import cv2
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QFileDialog, QMessageBox,
                             QGroupBox)

from core.watermark_controller import WatermarkController
from ui.loading_indicator import LoadingIndicator
from ui.split_image_view import SplitImageView
from ui.watermark_preview import WatermarkPreview
from utils.file_utils import load_image, save_image


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize controller
        self.controller = WatermarkController()
        self.connect_controller_signals()

        # Initialize UI components
        self.init_ui()

        # Initialize state variables
        self.current_image = None
        self.processed_image = None
        self.loading_indicator = None

        self.setWindowTitle("Watermark Processing Tool")
        self.resize(1200, 800)

    def connect_controller_signals(self):
        """Connect controller signals to UI methods"""
        # Watermark initialization signals
        self.controller.watermark_initialized.connect(self.on_watermark_initialized)
        self.controller.watermark_init_error.connect(self.on_error)

        # Watermark embedding signals
        self.controller.watermark_embedded.connect(self.on_watermark_embedded)
        self.controller.watermark_embed_error.connect(self.on_error)

        # Watermark recovery signals
        self.controller.watermark_recovered.connect(self.on_watermark_recovered)
        self.controller.watermark_recover_error.connect(self.on_error)

        # Tampering detection signals
        self.controller.tampering_detected.connect(self.on_tampering_detected)
        self.controller.tampering_error.connect(self.on_error)

    def init_ui(self):
        # Create central widget and main layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        # Top layout for image views and watermark preview
        top_layout = QHBoxLayout()

        # Create split image view for original and processed images
        self.split_image_view = SplitImageView()

        # Create watermark preview widget
        self.watermark_preview = WatermarkPreview()

        # Add widgets to top layout
        top_layout.addWidget(self.split_image_view, 4)  # Give the images more space
        top_layout.addWidget(self.watermark_preview, 1)  # Give watermark preview less space

        # Add top layout to main layout
        main_layout.addLayout(top_layout)

        # Create watermark actions group
        watermark_group = QGroupBox("Watermark Actions")
        watermark_layout = QHBoxLayout(watermark_group)

        # Import watermark button
        self.import_watermark_btn = QPushButton("Import Watermark")
        self.import_watermark_btn.clicked.connect(self.import_watermark)
        watermark_layout.addWidget(self.import_watermark_btn)

        # Embed watermark button
        self.embed_btn = QPushButton("Embed Watermark")
        self.embed_btn.clicked.connect(self.on_embed_watermark)
        watermark_layout.addWidget(self.embed_btn)

        # Recover watermark button
        self.recover_btn = QPushButton("Recover Watermark")
        self.recover_btn.clicked.connect(self.on_recover_watermark)
        watermark_layout.addWidget(self.recover_btn)

        # Save watermark button
        self.save_watermark_btn = QPushButton("Save Watermark")
        self.save_watermark_btn.clicked.connect(self.save_watermark)
        watermark_layout.addWidget(self.save_watermark_btn)

        # Create image actions group
        image_group = QGroupBox("Image Actions")
        image_layout = QHBoxLayout(image_group)

        # Import image button
        self.import_btn = QPushButton("Import Image")
        self.import_btn.clicked.connect(self.import_image)
        image_layout.addWidget(self.import_btn)

        # Tampering detection button
        self.detect_btn = QPushButton("Detect Tampering")
        self.detect_btn.clicked.connect(self.on_detect_tampering)
        image_layout.addWidget(self.detect_btn)

        # Save button
        self.save_btn = QPushButton("Save Processed Image")
        self.save_btn.clicked.connect(self.save_processed_image)
        image_layout.addWidget(self.save_btn)

        # Add button groups to main layout
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(image_group)
        buttons_layout.addWidget(watermark_group)
        main_layout.addLayout(buttons_layout)

        # Set central widget
        self.setCentralWidget(central_widget)

        # Disable processing buttons initially
        self._set_buttons_enabled(False)
        self.import_btn.setEnabled(True)
        self.import_watermark_btn.setEnabled(True)

    # UI Event Handlers
    def import_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_path:
            self.current_image = load_image(file_path)
            if self.current_image is not None:
                self.split_image_view.set_original_image(self.current_image)
                self.split_image_view.set_processed_image(None)  # Clear processed view

                # Disable buttons during processing
                self._set_buttons_enabled(False)

                # Show loading indicator
                self.show_loading_indicator("Initializing watermark processor...")

                # Initialize watermark object
                self.controller.set_image(self.current_image)

    def import_watermark(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Watermark Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_path:
            watermark_img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if watermark_img is not None:
                # Set watermark in controller
                binary_watermark = self.controller.set_watermark(watermark_img)

                # Update UI
                self.watermark_preview.set_watermark(binary_watermark)

                # Enable appropriate buttons
                self.embed_btn.setEnabled(self.current_image is not None)
                self.detect_btn.setEnabled(self.current_image is not None)

    def on_embed_watermark(self):
        self._set_buttons_enabled(False)
        self.show_loading_indicator("Embedding watermark...")
        self.controller.embed_watermark()

    def on_recover_watermark(self):
        self._set_buttons_enabled(False)
        self.show_loading_indicator("Recovering watermark...")
        self.controller.recover_watermark()

    def on_detect_tampering(self):
        self._set_buttons_enabled(False)
        self.show_loading_indicator("Detecting tampering...")
        self.controller.detect_tampering()

    def save_processed_image(self):
        if self.processed_image is not None:
            file_path, selected_filter = QFileDialog.getSaveFileName(
                self, "Save Image", "", "PNG Files (*.png);;JPEG Files (*.jpg)"
            )

            if file_path:
                if selected_filter == "PNG Files (*.png)" and not file_path.lower().endswith('.png'):
                    file_path += '.png'
                elif selected_filter == "JPEG Files (*.jpg)" and not file_path.lower().endswith(('.jpg', '.jpeg')):
                    file_path += '.jpg'

                success = save_image(self.processed_image, file_path)
                if success:
                    QMessageBox.information(self, "Success", "Image saved successfully!")
                else:
                    QMessageBox.critical(self, "Error", "Failed to save the image.")

    def save_watermark(self):
        """Save the current watermark to a file"""
        if self.controller.current_watermark is not None:
            file_path, selected_filter = QFileDialog.getSaveFileName(
                self, "Save Watermark", "", "PNG Files (*.png);;JPEG Files (*.jpg)"
            )

            if file_path:
                # Ensure file has the correct extension
                if selected_filter == "PNG Files (*.png)" and not file_path.lower().endswith('.png'):
                    file_path += '.png'
                elif selected_filter == "JPEG Files (*.jpg)" and not file_path.lower().endswith(('.jpg', '.jpeg')):
                    file_path += '.jpg'

                success = save_image(self.controller.current_watermark, file_path)
                if success:
                    QMessageBox.information(self, "Success", "Watermark saved successfully!")
                else:
                    QMessageBox.critical(self, "Error", "Failed to save the watermark.")

    # Controller Signal Handlers
    @pyqtSlot(object)
    def on_watermark_initialized(self, watermark_obj):
        """Called when Watermark object is successfully initialized"""
        self.hide_loading_indicator()

        # Enable appropriate buttons
        self.import_watermark_btn.setEnabled(True)
        self.import_btn.setEnabled(True)
        self.recover_btn.setEnabled(True)
        # Only enable these if we have a watermark
        has_watermark = self.controller.current_watermark is not None
        self.embed_btn.setEnabled(has_watermark)
        self.detect_btn.setEnabled(has_watermark)
        self.save_btn.setEnabled(False)

    @pyqtSlot(object)
    def on_watermark_embedded(self, embedded_image):
        """Called when watermark is successfully embedded"""
        self.processed_image = embedded_image
        self.split_image_view.set_processed_image(embedded_image)
        self.hide_loading_indicator()

        # Enable buttons
        self._set_buttons_enabled(True)
        self.split_image_view.set_processed_title("Embedded Image")

    @pyqtSlot(object)
    def on_watermark_recovered(self, recovered_watermark):
        """Called when watermark is successfully recovered"""
        self.watermark_preview.set_watermark(recovered_watermark)
        self.hide_loading_indicator()

        # Enable buttons
        self._set_buttons_enabled(True)

        QMessageBox.information(self, "Success", "Watermark recovered successfully!")

    @pyqtSlot(object, bool)
    def on_tampering_detected(self, result_image, is_tampered):
        """Called when tampering detection is complete"""
        if is_tampered:
            self.processed_image = result_image
            self.split_image_view.set_processed_image(result_image)

        self.hide_loading_indicator()

        # Enable buttons
        self._set_buttons_enabled(True)
        self.split_image_view.set_processed_title("Tampered points visualisation")

        # Show appropriate message
        if is_tampered:
            QMessageBox.warning(self, "Tampering Detected",
                                "The image appears to have been tampered with!")
        else:
            QMessageBox.information(self, "No Tampering Detected",
                                    "No evidence of tampering was found.")

    @pyqtSlot(str)
    def on_error(self, error_message):
        """General error handler for controller errors"""
        self.hide_loading_indicator()
        QMessageBox.critical(self, "Error", error_message)

        # Re-enable buttons
        self._set_buttons_enabled(True)

    # Helper methods
    def show_loading_indicator(self, message="Processing..."):
        """Show loading indicator with message"""
        if self.loading_indicator is None:
            self.loading_indicator = LoadingIndicator(self, message)
        else:
            self.loading_indicator.set_message(message)

        self.loading_indicator.show()

    def hide_loading_indicator(self):
        """Hide loading indicator"""
        if self.loading_indicator:
            self.loading_indicator.hide()

    def _set_buttons_enabled(self, enabled):
        """Enable or disable all action buttons"""
        self.import_btn.setEnabled(enabled)
        self.import_watermark_btn.setEnabled(enabled)
        self.embed_btn.setEnabled(enabled and self.current_image is not None and self.controller.current_watermark is not None)
        self.recover_btn.setEnabled(enabled and self.current_image is not None)
        self.detect_btn.setEnabled(enabled and self.current_image is not None and self.controller.current_watermark is not None)
        self.save_btn.setEnabled(enabled and self.processed_image is not None)
        self.save_watermark_btn.setEnabled(enabled and self.controller.current_watermark is not None)

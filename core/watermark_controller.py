# core/watermark_controller.py
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from core.background_worker import BackgroundWorker
from core.watermark import Watermark
from utils.image_utils import convert_to_binary_watermark

class WatermarkController(QObject):
    # Signals
    watermark_initialized = pyqtSignal(object)
    watermark_init_error = pyqtSignal(str)
    watermark_embedded = pyqtSignal(object)
    watermark_embed_error = pyqtSignal(str)
    watermark_recovered = pyqtSignal(object)
    watermark_recover_error = pyqtSignal(str)
    tampering_detected = pyqtSignal(object, bool)
    tampering_error = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.watermark_obj = None
        self.current_image = None
        self.current_watermark = None
        self.d_factor = 50

    def set_image(self, image):
        """Set current image and initialize watermark object"""
        self.current_image = image

        # Initialize watermark object in background thread
        self.init_worker = BackgroundWorker(
            self._initialize_watermark_obj,
            image,
            self.d_factor,
            self.current_watermark
        )
        self.init_worker.finished.connect(self._on_watermark_initialized)
        self.init_worker.error.connect(self._on_watermark_init_error)
        self.init_worker.start()

        return True

    def set_watermark(self, watermark_img):
        """Set current watermark"""
        self.current_watermark = convert_to_binary_watermark(watermark_img)

        # Update watermark in watermark object if it exists
        if self.watermark_obj:
            self.watermark_obj.set_watermark(self.current_watermark)

        return self.current_watermark

    def embed_watermark(self):
        """Embed watermark into current image"""
        if self.current_image is None or self.current_watermark is None:
            self.watermark_embed_error.emit("No image or watermark loaded")
            return False
        if self.watermark_obj is None:
            self.watermark_embed_error.emit("Watermark processor not initialized")
            return False

        # Embed watermark in background thread
        self.embed_worker = BackgroundWorker(self.watermark_obj.embed)
        self.embed_worker.finished.connect(self._on_watermark_embedded)
        self.embed_worker.error.connect(self._on_watermark_embed_error)
        self.embed_worker.start()

        return True

    def recover_watermark(self):
        """Recover watermark from current image"""
        if self.current_image is None:
            self.watermark_recover_error.emit("No image loaded")
            return False

        if self.watermark_obj is None:
            self.watermark_recover_error.emit("Watermark processor not initialized")
            return False

        # Recover watermark in background thread
        self.recover_worker = BackgroundWorker(
            self._recover_watermark_task
        )
        self.recover_worker.finished.connect(self._on_watermark_recovered)
        self.recover_worker.error.connect(self._on_watermark_recover_error)
        self.recover_worker.start()

        return True

    def detect_tampering(self):
        """Detect tampering in current image"""
        if self.current_image is None or self.current_watermark is None:
            self.tampering_error.emit("No image or watermark loaded")
            return False
        if self.watermark_obj is None:
            self.tampering_error.emit("Watermark processor not initialized")
            return False

        # Detect tampering in background thread
        self.tampering_worker = BackgroundWorker(
            self._detect_tampering_task
        )
        self.tampering_worker.finished.connect(self._on_tampering_detected)
        self.tampering_worker.error.connect(self._on_tampering_error)
        self.tampering_worker.start()

        return True

    # Private methods to initialize watermark object
    def _initialize_watermark_obj(self, image, d_factor, watermark_img=None):
        """Create Watermark object (runs in background thread)"""
        return Watermark(image, d_factor=d_factor, watermark_img=watermark_img)

    def _recover_watermark_task(self):
        """Recover watermark (runs in background thread)"""
        return self.watermark_obj.recover_watermark()

    def _detect_tampering_task(self):
        """Detect tampering (runs in background thread)"""
        visualisation_image, is_valid = self.watermark_obj.validate_watermark()
        return visualisation_image, not is_valid

    # Slots for worker thread signals
    @pyqtSlot(object)
    def _on_watermark_initialized(self, watermark_obj):
        """Called when Watermark object is successfully initialized"""
        self.watermark_obj = watermark_obj
        self.watermark_initialized.emit(watermark_obj)

    @pyqtSlot(str)
    def _on_watermark_init_error(self, error_message):
        """Called if there's an error initializing the Watermark object"""
        self.watermark_init_error.emit(error_message)

    @pyqtSlot(object)
    def _on_watermark_embedded(self, embedded_image):
        """Called when watermark is successfully embedded"""
        self.watermark_embedded.emit(embedded_image)

    @pyqtSlot(str)
    def _on_watermark_embed_error(self, error_message):
        """Called if there's an error embedding the watermark"""
        self.watermark_embed_error.emit(error_message)

    @pyqtSlot(object)
    def _on_watermark_recovered(self, recovered_watermark):
        """Called when watermark is successfully recovered"""
        self.current_watermark = recovered_watermark
        self.watermark_obj.set_watermark(recovered_watermark)
        self.watermark_recovered.emit(recovered_watermark)

    @pyqtSlot(str)
    def _on_watermark_recover_error(self, error_message):
        """Called if there's an error recovering the watermark"""
        self.watermark_recover_error.emit(error_message)

    @pyqtSlot(object)
    def _on_tampering_detected(self, result):
        """Called when tampering detection is complete"""
        result_image, is_tampered = result
        self.tampering_detected.emit(result_image, is_tampered)

    @pyqtSlot(str)
    def _on_tampering_error(self, error_message):
        """Called if there's an error detecting tampering"""
        self.tampering_error.emit(error_message)
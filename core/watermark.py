# core/watermark.py
import cv2
import numpy as np
import math

from core.embedding_point import EmbeddingPoint
from utils.image_utils import convert_to_binary_watermark
from core.sift import SIFT


class Watermark:
    def __init__(self, image, d_factor, watermark_img=None, header_size=16):
        if header_size % 2 != 0:
            raise ValueError(
                "Header size must be even, as it represents the number of bits to encode the watermark"
            )

        self.header_size = header_size
        self.watermark_img = watermark_img
        self.image = image
        self.d_factor = d_factor
        # Different patterns for encoding 0 and 1 bits
        self.patterns1 = [
            [1,0,1,0,1,0,1,0,1],
            [1,1,1,0,0,0,1,1,1],
            [1,1,1,0,0,0,1,0,1]
        ]
        self.patterns0 = [
            [1,0,1,1,1,1,1,0,1],
            [1,0,1,0,0,0,1,0,1],
            [0,1,0,1,1,1,0,1,0]
        ]
        self._init_image_data()

    def _init_image_data(self):
        """
        Initialize image data by converting color image to grayscale if needed,
        splitting color channels, and computing SIFT descriptors.
        The blue channel is used as the working image for watermark embedding.
        """
        if len(self.image.shape) == 3:
            self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            self.gray_image = self.image

        self.ch_b, self.ch_g, self.ch_r = cv2.split(self.image)
        self.working_image = self.ch_b.copy()
        self.descriptors = self._find_descriptors()

    def set_watermark(self, watermark_img):
        self.watermark_img = watermark_img

    def embed(self):
        """
        Embed a watermark into the image using SIFT keypoints and LSB steganography.
        
        The method works by:
        1. Converting watermark to a binary packet with header + payload
        2. Finding SIFT keypoints in image
        3. For each keypoint, encoding bits by modifying LSBs in a kernel pattern
        4. Preserving the first keypoints for header information
        
        Returns:
            numpy.ndarray: The watermarked image with embedded data
            
        Raises:
            ValueError: If a watermark image is not provided or too large to embed
        """
        if self.watermark_img is None:
            raise ValueError("Watermark image is not provided")

        self._reset_state(for_embedding=True)
        # Convert watermark to binary and add a dimension header
        packet = self._prepare_watermark_packet(self.watermark_img)
        bit_count = packet.size

        # Calculate how many bits should be encoded per keypoint
        bits_per_keypoint = math.ceil((bit_count - self.header_size) / (len(self.descriptors) - self.header_size))
        if bits_per_keypoint < 1:
            raise ValueError(
                f"Watermark size is too large for the image, cannot encode {bit_count} bits in {len(self.descriptors)} keypoints"
            )

        j = 0
        for i, (keypoint, descriptor) in enumerate(self.descriptors.items()):
            # Encode 1 bit per keypoint for header for easy header retrieval
            if i < self.header_size:
                stego_values = self._find_embedding_points(keypoint, descriptor, 1)
            else:
                stego_values = self._find_embedding_points(keypoint, descriptor, bits_per_keypoint)

            # Encode bits by modifying LSBs according to kernel patterns
            for stego_val in stego_values:
                current_bit = packet[j]
                kernel = self._next_pattern(current_bit)
                self._encode_bit(stego_val, kernel, current_bit)
                j += 1
                # Return merged channels once all bits are encoded
                if j >= bit_count:
                    return cv2.merge([self.working_image, self.ch_g, self.ch_r])

        # At this point there are no more keypoints to encode in, and there are bits left in the packet
        raise ValueError(f"The watermark is too large to embed")

    def recover_watermark(self):
        """
        Attempts to recover a binary watermark image that was previously embedded.
        
        The method works by:
        1. Setting pattern indices to initial state
        2. Recovering embedded bits from the image
        3. Converting bits back to a binary image
        
        Returns:
            numpy.ndarray: Recovered binary watermark image where white (255) represents 1s
            
        Raises:
            ValueError: If watermark bits cannot be properly recovered
        """
        # Reset pattern counters to initial state
        self._reset_state()

        # Recover the embedded bits from the image
        bits = self.recover_bits()

        # Check if any bits failed to be recovered
        if -1 in bits:
            raise ValueError("Failed to recover watermark")

        # Convert bits to binary image (0->0, 1->255)
        watermark_image = bits.astype(np.uint8) * 255
        return watermark_image

    def recover_bits(self):
        """
        Recovers watermark bits from the image using the embedded header and payload.
        
        The method works in two phases:
        1. Header recovery (first self.header_size bits):
           - Reads height from first half of header bits
           - Reads width from second half of header bits
           - Uses max_dimension if header bits are corrupted (-1)
           
        2. Payload recovery:
           - Adjusts bits per keypoint based on payload size
           - Decodes remaining bits to form watermark image
           
        Returns:
            numpy.ndarray: 2D array of recovered watermark bits reshaped to height x width
        """
        bits_per_keypoint = 1
        h, w = 0, 0
        # Maximum packet size possible for the header size
        max_dimension = 2 ** (self.header_size // 2) - 1
        bits_count = self.header_size + max_dimension ** 2
        bits = []

        j = 0  # Counter for total bits processed
        for i, (keypoint, descriptor) in enumerate(self.descriptors.items()):
            stego_values = self._find_embedding_points(keypoint, descriptor, bits_per_keypoint)

            for stego_val in stego_values:
                bits.append(self._decode_bit(stego_val))
                j += 1
                if j >= bits_count:
                    return np.array(bits[self.header_size:]).reshape(h, w)

            # Once header is recovered, decode dimensions and adjust payload parameters
            if len(bits) == self.header_size:
                # Get height from first half of header (use max_dimension if corrupted)
                if -1 in bits[:8]:
                    h = max_dimension
                else:
                    h = int("".join(map(str, bits[:8])), 2)

                # Get width from second half of header (use max_dimension if corrupted)
                if -1 in bits[8:]:
                    w = max_dimension
                else:
                    w = int("".join(map(str, bits[8:])), 2)

                # Adjust payload size based on the watermark dimensions encoded in the header
                payload_size = h * w
                bits_count = self.header_size + payload_size
                bits_per_keypoint = math.ceil(payload_size / (len(self.descriptors) - self.header_size))

        # Extract payload and pad with -1 if incomplete
        payload = bits[self.header_size:]
        expected_size = h*w
        if len(payload) < expected_size:
            payload += [-1] * (expected_size - len(payload))

        return np.array(payload).reshape(h, w)

    def validate_watermark(self):
        """
        Validate an embedded watermark against the original watermark image.
        
        This method works similarly to recover_watermark(), but instead of extracting
        the watermark, it validates each recovered bit against the original watermark.
        The header extraction is omitted since we already know the watermark dimensions.
        
        Returns:
            tuple: (error_points, is_valid) where:
                - visualisation image presenting image with points where tampering was detected
                - is_valid is True if no errors were found, False otherwise
                
        Raises:
            ValueError: If watermark image is not provided for validation
        """
        if self.watermark_img is None:
            raise ValueError("Watermark image is not provided")

        # Prepare watermark packet and sync pattern indices to match encoded header state
        packet = self._prepare_watermark_packet(self.watermark_img)
        watermark_bits = packet[self.header_size:]
        self._sync_pattern_indices(packet[:self.header_size])

        # Calculate bits per keypoint based on payload size
        bits_count = len(watermark_bits)
        bits_per_keypoint = math.ceil(bits_count / (len(self.descriptors) - self.header_size))

        # Track points where decoded bits don't match watermark
        error_points = []

        # Validate each bit, skipping header keypoints
        j = 0
        for (keypoint, descriptor) in list(self.descriptors.items())[self.header_size:]:
            stego_values = self._find_embedding_points(keypoint, descriptor, bits_per_keypoint)

            for stego_val in stego_values:
                decoded_bit = self._decode_bit(stego_val)
                if decoded_bit != watermark_bits[j]:
                    error_points.append(stego_val)
                j += 1
                if j >= bits_count:
                    return self._create_tamper_visualisation(error_points), not error_points

        return self._create_tamper_visualisation(error_points), False

    def _create_tamper_visualisation(self, error_points):
        """
        Create an image with visual markers showing where tampering was detected.

        Args:
            error_points: List of EmbeddingPoint objects representing tampered locations.
                         If None, uses the result of the last validate_watermark() call.

        Returns:
            numpy.ndarray: Copy of the original image with red circles marking tampered areas
        """
        if error_points is None:
            raise ValueError("error_points must be provided")

        # Create a copy to avoid modifying the original
        visualisation = self.image.copy()

        # Draw markers at each error point
        for point in error_points:
            # Draw a larger red circle at each error point
            cv2.circle(
                visualisation,
                (point.x, point.y),
                radius=15,  # Larger radius for better visibility
                color=(0, 0, 255),  # Red in BGR
                thickness=2
            )

            # Draw a filled circle at the center for emphasis
            cv2.circle(
                visualisation,
                (point.x, point.y),
                radius=5,  # Smaller center point
                color=(0, 0, 255),  # Red in BGR
                thickness=-1  # -1 means filled circle
            )

        return visualisation

    def _find_descriptors(self):
        """
        Detect SIFT keypoints and compute their descriptors for the image.
        
        Uses SIFT algorithm to find distinctive keypoints and computes descriptor 
        vectors for each keypoint.
        
        Returns:
            dict: Dictionary mapping keypoints to their descriptors
        """
        sift = SIFT(d_factor=self.d_factor)
        keypoints = sift.detect_and_filter_keypoints(self.gray_image)

        return sift.compute_descriptors(self.gray_image, keypoints)

    def _find_embedding_points(self, keypoint, descriptor, points=1, kernel_size=3):
        """Find suitable points around a SIFT keypoint for embedding watermark bits.
        
        For each SIFT keypoint, this method finds locations where kernel patterns can be
        applied to encode watermark bits. The points are selected based on the strongest
        orientation magnitudes and positioned to ensure:
        1. They fall outside the 16x16 descriptor block to preserve SIFT detection
        2. The kernel patterns don't overlap with each other
        3. The distance from keypoint is scaled by orientation magnitude
        
        Args:
            keypoint: SIFT keypoint object containing position information
            descriptor: SIFT descriptor for the keypoint
            points: Number of embedding points to find. Defaults to 1.
            kernel_size: Size of the kernel pattern (must be odd). Defaults to 3.
            
        Returns:
            List of EmbeddingPoint objects containing position and orientation data
        """
        blocks = descriptor.reshape(16, 8)  # Reshape descriptor into 16 orientation blocks
        pts = []
        x0, y0 = keypoint.pt
        kernel_half = kernel_size // 2
        min_separation = (2 * kernel_half + 1) ** 2  # Minimum distance between kernel centers
        h, w = self.gray_image.shape

        for bi, block in enumerate(blocks):
            # Process orientations in descending magnitude order
            for i in np.argsort(-block):
                theta = i * 45  # Convert orientation index to degrees
                magnitude = block[i] / 512  # Denormalize magnitude
                # Scale distance from keypoint based on magnitude
                dx = (16 + self.d_factor * magnitude) * math.cos(math.radians(theta))
                dy = (16 + self.d_factor * magnitude) * math.sin(math.radians(theta))
                # Calculate embedding point coordinates
                x_c, y_c = int(round(x0 + dx)), int(round(y0 + dy))

                # Check if the point is within the image boundaries
                if(x_c - kernel_half) < 0 or (x_c + kernel_half) >= w or (y_c - kernel_half) < 0 or (y_c + kernel_half) >= h:
                    continue

                # Check if the kernels will not overlap with existing points
                if all((x_c - e.x)**2 + (y_c - e.y)**2 > min_separation for e in pts):
                    pts.append(EmbeddingPoint(theta, magnitude, x_c, y_c, bi))
                    break

            # Only a defined number of points is required
            if len(pts) >= points:
                break

        return pts

    def _next_pattern(self, bit=0):
        """
        Increment pattern index and return the next encoding pattern for the specified bit value.
        
        Args:
            bit: The bit value (0 or 1) to get pattern for. Defaults to 0.
            
        Returns:
            list: Next kernel pattern for encoding the specified bit value after incrementing index
        """
        if bit == 0:
            self.pattern0_idx = (self.pattern0_idx + 1) % len(self.patterns0)
            return self.patterns0[self.pattern0_idx]
        else:
            self.pattern1_idx = (self.pattern1_idx + 1) % len(self.patterns1)
            return self.patterns1[self.pattern1_idx]

    def _seek_pattern(self, bit=0):
        """
        Get the current encoding pattern for the specified bit value without incrementing pattern index.
    
        Args:
            bit: The bit value (0 or 1) to get pattern for. Defaults to 0.
    
        Returns:
            list: Current kernel pattern for encoding the specified bit value
        """
        if bit == 0:
            return self.patterns0[self.pattern0_idx]
        else:
            return self.patterns1[self.pattern1_idx]

    def _encode_bit(self, point: EmbeddingPoint, mask, bit):
        """
        Apply a kernel mask to change LSBs at specific points

        Args:
            point: The center point of the kernel
            mask: A flat list or array representing the kernel mask
            bit: The bit value (0 or 1) to apply where mask is active
        """
        region, mask_2d = self._prepare_kernel_region(point, mask)

        if bit == 1:
            region[mask_2d] |= 1
        else:
            region[mask_2d] &= 0xFE

    def _decode_bit(self, point: EmbeddingPoint):
        """
        Decode a bit by checking LSBs in regions defined by mask0 and mask1

        Args:
            point: The center point of the kernel
        Returns:
            0 if all LSBs in mask0 are 0s
            1 if all LSBs in mask1 are 1s
            None if neither condition is satisfied
        """
        mask1 = self._seek_pattern(1)
        kernel_size = int(math.sqrt(len(mask1)))
        region, mask0_2d = self._prepare_kernel_region(point, self._seek_pattern(0))
        mask1_2d = np.array(mask1).reshape(kernel_size, kernel_size).astype(bool)
        lsbs = region & 1

        # Check if all LSBs in mask0 are 0s
        if np.all(lsbs[mask0_2d] == 0):
            self._next_pattern(0)
            return 0
        # Check if any LSBs in mask1 are not 1
        elif np.any(lsbs[mask1_2d] != 1):
            # Bit could not be retrieved, i.e. watermark is corrupted
            return -1
        else:
            self._next_pattern(1)
            return 1

    def _prepare_kernel_region(self, point, mask):
        """
        Helper method to prepare a kernel region for operations

        Args:
            point: The center point of the kernel
            mask: A flat list or array representing the kernel mask

        Returns:
            tuple: (region, mask_2d)
        """
        kernel_size = int(math.sqrt(len(mask)))
        half = kernel_size // 2

        x_start = int(point.x - half)
        y_start = int(point.y - half)

        mask_2d = np.array(mask).reshape(kernel_size, kernel_size).astype(bool)
        region = self.working_image[y_start:y_start + kernel_size, x_start:x_start + kernel_size]

        return region, mask_2d

    def _prepare_watermark_packet(self, watermark):
        """
        Prepare a watermark packet by combining header and payload bits.
        
        Creates an array of bits where first self.header_size bits encode the watermark 
        dimensions (height and width) and the rest contains the actual watermark data.
    
        Args:
            watermark: Input watermark image numpy array
    
        Returns:
            1D numpy array containing header bits followed by watermark payload bits
    
        Raises:
            ValueError: If watermark dimensions exceed maximum size, that can be encoded in the header
        """
        watermark_bits = self._get_watermark_bits(watermark)

        height, width = self.watermark_img.shape[:2]
        half_header = self.header_size // 2  # Split header bits equally between height and width
        max_dimension = 2**half_header - 1

        if height > max_dimension or width > max_dimension:
            raise ValueError(f"Watermark dimensions exceed {max_dimension} pixels, cannot encode in {half_header} bits")

        height_bits = np.array(list(np.binary_repr(height, width=half_header)), dtype=np.uint8)
        width_bits = np.array(list(np.binary_repr(width, width=half_header)), dtype=np.uint8)

        return np.concatenate((height_bits, width_bits, watermark_bits))

    def _sync_pattern_indices(self, header):
        """
        Synchronize pattern indices to match the state after header encoding.
        
        During validation, we skip decoding the header bits and directly compare with 
        the retrieved watermark. However, since pattern indices were incremented while 
        encoding header bits, we need to simulate those increments to ensure patterns 
        are properly aligned for payload validation.
    
        Args:
            header: Array of header bits that were used during encoding
        """
        self._reset_state()
        for bit in header:
            self._next_pattern(bit)

    
    @staticmethod
    def _get_watermark_bits(watermark):
        """
        Convert a watermark image to a flattened array of binary bits.
        
        For PNG images with transparency, transparent pixels are converted to white.
        The image is converted to grayscale and thresholded to create binary values.
        
        Args:
            watermark: Input watermark image numpy array
            
        Returns:
            1D numpy array of binary bits (0s and 1s)
        """
        binary_watermark = convert_to_binary_watermark(watermark)
        return np.clip(binary_watermark, 0, 1).astype(np.uint8).flatten()

    def _reset_state(self, for_embedding=False):
        """Reset pattern indices and working image to initial state."""
        if for_embedding:
            self.pattern0_idx = -1
            self.pattern1_idx = -1
        else:
            self.pattern0_idx = 0
            self.pattern1_idx = 0
        self.working_image = self.ch_b.copy()
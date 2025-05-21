# core/sift.py
import cv2
import numpy as np


class SIFT:
    """
    Class for SIFT (Scale-Invariant Feature Transform) operations.
    Provides methods for keypoint detection, descriptor computation,
    and feature matching.
    """

    def __init__(self, contrast_threshold=0.04, edge_threshold=10, sigma=1.6, d_factor = 50):
        """
        Initialize SIFT with customizable parameters.

        Args:
            contrast_threshold: Threshold for filtering low-contrast keypoints
            edge_threshold: Threshold for filtering edge-like keypoints
            sigma: Base sigma for Gaussian scale space
        """
        self.d_factor = d_factor
        self.sift = cv2.SIFT_create(
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
            sigma=sigma
        )

    def detect_and_filter_keypoints(self, gray_image):
        """
        Detect SIFT keypoints in an image and filter out close-by ones.

        Args:
            gray_image: Input grayscale image

        Returns:
            List of filtered keypoints distant from each other by 0.5 * d_factor + 17 * 2
        """

        # Detect keypoints
        keypoints = self.sift.detect(gray_image, None)

        # Sort keypoints by response (strength)
        keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)

        # Filter out keypoints that are too close to each other
        min_separation = (self.d_factor * 0.5 + 17) * 2
        filtered_keypoints = []
        for kp in keypoints:
            x, y = kp.pt
            if all(np.hypot(x - c.pt[0], y - c.pt[1]) >= min_separation for c in filtered_keypoints):
                filtered_keypoints.append(kp)

        return filtered_keypoints

    def compute_descriptors(self, gray_image, keypoints):
        """
        Compute SIFT descriptors for keypoints.

        Args:
            gray_image: Input grayscale image
            keypoints: List of keypoints

        Returns:
            Dictionary mapping keypoints to their descriptors
        """

        # Compute descriptors
        keypoints, descriptors = self.sift.compute(gray_image, keypoints)
        return {kp: desc for kp, desc in zip(keypoints, descriptors)}
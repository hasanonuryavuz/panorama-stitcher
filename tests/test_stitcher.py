"""Bilinen dönüşüm, gerçek siyah pikseller ve başarısız girdiler."""
import tempfile
import unittest
import cv2
import numpy as np
from demo import create_demo
from stitcher import stitch, StitchingError


class StitcherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as directory:
            cls.left, cls.right, cls.world, cls.expected = create_demo(directory)
        cv2.setRNGSeed(42)
        cls.result = stitch(cls.left, cls.right)

    def test_known_perspective_is_recovered(self):
        points = np.float32([[100, 100], [300, 200], [600, 350]]).reshape(-1, 1, 2)
        actual = cv2.perspectiveTransform(points, self.result.homography)
        expected = cv2.perspectiveTransform(points, self.expected)
        self.assertLess(float(np.linalg.norm(actual - expected, axis=2).max()), 2.0)
        self.assertGreater(self.result.metrics['inlier_ratio'], 0.8)
        self.assertGreaterEqual(self.result.panorama.shape[1], 1090)

    def test_black_pixels_remain_valid_image_content(self):
        # Bilinen siyah yamanın içi: dış çerçevedeki siyahlık bu testi geçiremez.
        # Demo tuvalinde üst öteleme ~15 px; bu alan yamanın merkezinde kalır.
        interior = self.result.panorama[340:350, 520:540]
        self.assertTrue(np.all(interior == 0))

    def test_reversed_inputs_recover_inverse_transform(self):
        result = stitch(self.right, self.left)
        points = np.float32([[450, 100], [550, 250], [650, 350]]).reshape(-1, 1, 2)
        expected = cv2.perspectiveTransform(points, np.linalg.inv(self.expected))
        actual = cv2.perspectiveTransform(points, result.homography)
        self.assertLess(float(np.linalg.norm(actual - expected, axis=2).max()), 2.0)

    def test_blank_images_are_rejected(self):
        blank = np.zeros((200, 300, 3), np.uint8)
        with self.assertRaisesRegex(StitchingError, 'özellik'):
            stitch(blank, blank)

    def test_unrelated_textures_are_rejected(self):
        rng = np.random.default_rng(14)
        a = rng.integers(0, 256, (300, 400, 3), dtype=np.uint8)
        b = rng.integers(0, 256, (300, 400, 3), dtype=np.uint8)
        with self.assertRaises(StitchingError):
            stitch(a, b)

    def test_canvas_limit_prevents_large_allocations(self):
        with self.assertRaisesRegex(StitchingError, 'Tuval'):
            stitch(self.left, self.right, max_canvas_pixels=100)

    def test_invalid_ratio_is_rejected(self):
        with self.assertRaises(StitchingError):
            stitch(self.left, self.right, ratio=1.5)


if __name__ == '__main__':
    unittest.main()

"""Tests for VisionModule."""

import unittest
import numpy as np

from src.env import TableTopEnv
from src.vision import VisionModule


class TestVisionModule(unittest.TestCase):

    def setUp(self):
        self.env = TableTopEnv(gui=False)
        self.env.reset()
        self.vision = VisionModule(self.env)

    def tearDown(self):
        self.env.close()

    def test_capture_image_keys(self):
        images = self.vision.capture_image()
        self.assertIn("rgb", images)
        self.assertIn("depth", images)
        self.assertIn("segmentation", images)

    def test_rgb_shape(self):
        images = self.vision.capture_image()
        self.assertEqual(images["rgb"].shape, (480, 640, 3))

    def test_rgb_dtype(self):
        images = self.vision.capture_image()
        self.assertEqual(images["rgb"].dtype, np.uint8)

    def test_depth_shape(self):
        images = self.vision.capture_image()
        self.assertEqual(images["depth"].shape, (480, 640))

    def test_segmentation_shape(self):
        images = self.vision.capture_image()
        self.assertEqual(images["segmentation"].shape, (480, 640))

    def test_detect_objects_returns_list(self):
        detections = self.vision.detect_objects()
        self.assertIsInstance(detections, list)

    def test_detect_objects_finds_some(self):
        detections = self.vision.detect_objects()
        self.assertGreater(len(detections), 0)

    def test_detection_has_name(self):
        detections = self.vision.detect_objects()
        if detections:
            self.assertIn("name", detections[0])
            self.assertIn("body_id", detections[0])
            self.assertIn("pixel_count", detections[0])

    def test_scene_description_from_vision(self):
        scene = self.vision.get_scene_description_from_vision()
        self.assertIn("robot_state", scene)
        self.assertIn("objects", scene)

    def test_scene_objects_have_visible_field(self):
        scene = self.vision.get_scene_description_from_vision()
        for obj in scene["objects"]:
            self.assertIn("visible", obj)
            self.assertIn("pixel_count", obj)

    def test_custom_resolution(self):
        v = VisionModule(self.env, width=320, height=240)
        images = v.capture_image()
        self.assertEqual(images["rgb"].shape, (240, 320, 3))


if __name__ == "__main__":
    unittest.main()

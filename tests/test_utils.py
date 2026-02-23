"""Tests for src.utils."""

import unittest
import numpy as np

from src.utils import normalize_direction, position_on_table, objects_within_distance, clamp


class TestNormalizeDirection(unittest.TestCase):

    def test_unit_vector_2d(self):
        result = normalize_direction([3.0, 4.0])
        np.testing.assert_array_almost_equal(result, [0.6, 0.8])

    def test_unit_vector_3d(self):
        result = normalize_direction([0.0, 0.0, 5.0])
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 1.0])

    def test_already_unit(self):
        result = normalize_direction([1.0, 0.0])
        np.testing.assert_array_almost_equal(result, [1.0, 0.0])

    def test_negative_direction(self):
        result = normalize_direction([-1.0, 0.0])
        np.testing.assert_array_almost_equal(result, [-1.0, 0.0])

    def test_zero_raises(self):
        with self.assertRaises(ValueError):
            normalize_direction([0.0, 0.0])

    def test_returns_ndarray(self):
        self.assertIsInstance(normalize_direction([1.0, 1.0]), np.ndarray)


class TestPositionOnTable(unittest.TestCase):

    def test_default_heights(self):
        pos = position_on_table(0.5, 0.1)
        self.assertEqual(pos, [0.5, 0.1, 0.07])

    def test_custom_heights(self):
        pos = position_on_table(0.3, -0.2, table_height=0.10, object_half_height=0.05)
        self.assertAlmostEqual(pos[2], 0.15)

    def test_returns_list(self):
        self.assertIsInstance(position_on_table(0.0, 0.0), list)

    def test_length(self):
        self.assertEqual(len(position_on_table(0.0, 0.0)), 3)


class TestObjectsWithinDistance(unittest.TestCase):

    def setUp(self):
        self.objects = {
            "a": np.array([0.0, 0.0, 0.0]),
            "b": np.array([1.0, 0.0, 0.0]),
            "c": np.array([0.5, 0.0, 0.0]),
        }

    def test_finds_nearby(self):
        result = objects_within_distance(self.objects, np.array([0.0, 0.0, 0.0]), 0.6)
        names = [name for name, _ in result]
        self.assertIn("a", names)
        self.assertIn("c", names)
        self.assertNotIn("b", names)

    def test_sorted_by_distance(self):
        result = objects_within_distance(self.objects, np.array([0.0, 0.0, 0.0]), 2.0)
        dists = [d for _, d in result]
        self.assertEqual(dists, sorted(dists))

    def test_empty_when_none_close(self):
        result = objects_within_distance(self.objects, np.array([10.0, 10.0, 10.0]), 0.1)
        self.assertEqual(result, [])

    def test_all_found_large_radius(self):
        result = objects_within_distance(self.objects, np.array([0.5, 0.0, 0.0]), 100.0)
        self.assertEqual(len(result), 3)

    def test_returns_tuples(self):
        result = objects_within_distance(self.objects, np.array([0.0, 0.0, 0.0]), 2.0)
        for item in result:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)


class TestClamp(unittest.TestCase):

    def test_within_range(self):
        self.assertEqual(clamp(5.0, 0.0, 10.0), 5.0)

    def test_below_low(self):
        self.assertEqual(clamp(-1.0, 0.0, 10.0), 0.0)

    def test_above_high(self):
        self.assertEqual(clamp(15.0, 0.0, 10.0), 10.0)

    def test_at_boundary_low(self):
        self.assertEqual(clamp(0.0, 0.0, 10.0), 0.0)

    def test_at_boundary_high(self):
        self.assertEqual(clamp(10.0, 0.0, 10.0), 10.0)


if __name__ == "__main__":
    unittest.main()

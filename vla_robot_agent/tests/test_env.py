"""Tests for TableTopEnv."""

import unittest
import io
import contextlib
import numpy as np

from src.env import TableTopEnv


class TestTableTopEnvInit(unittest.TestCase):
    """Test environment initialisation and teardown."""

    def setUp(self):
        self.env = TableTopEnv(gui=False)

    def tearDown(self):
        self.env.close()

    def test_physics_client_connected(self):
        import pybullet as p
        self.assertTrue(p.isConnected(self.env.physics_client))

    def test_plane_loaded(self):
        self.assertIsNotNone(self.env.plane_id)

    def test_table_loaded(self):
        self.assertIsNotNone(self.env.table_id)

    def test_robot_loaded(self):
        self.assertIsNotNone(self.env.robot_id)


class TestSceneDescription(unittest.TestCase):
    """Test the rich scene description format."""

    def setUp(self):
        self.env = TableTopEnv(gui=False)
        self.scene = self.env.reset()

    def tearDown(self):
        self.env.close()

    # -- Top-level structure --

    def test_has_robot_state(self):
        self.assertIn("robot_state", self.scene)

    def test_has_objects(self):
        self.assertIn("objects", self.scene)

    # -- robot_state fields --

    def test_robot_state_has_ee_position(self):
        rs = self.scene["robot_state"]
        self.assertIn("end_effector_position", rs)
        self.assertEqual(len(rs["end_effector_position"]), 3)

    def test_robot_state_has_gripper_width(self):
        rs = self.scene["robot_state"]
        self.assertIn("gripper_width", rs)
        self.assertIsInstance(rs["gripper_width"], float)

    def test_robot_state_has_joint_positions(self):
        rs = self.scene["robot_state"]
        self.assertIn("joint_positions", rs)
        self.assertEqual(len(rs["joint_positions"]), 7)

    def test_gripper_width_positive(self):
        """Gripper should be open after reset."""
        gw = self.scene["robot_state"]["gripper_width"]
        self.assertGreater(gw, 0.0)

    # -- objects list --

    def test_objects_count(self):
        self.assertEqual(len(self.scene["objects"]), 4)

    def test_object_names(self):
        names = {o["name"] for o in self.scene["objects"]}
        self.assertEqual(names, {"red_block", "green_block", "blue_block", "bowl"})

    def test_object_fields(self):
        required = {"name", "type", "color", "position", "orientation", "id"}
        for obj in self.scene["objects"]:
            self.assertTrue(required.issubset(obj.keys()),
                            f"Missing fields in {obj['name']}: {required - obj.keys()}")

    def test_block_type(self):
        for obj in self.scene["objects"]:
            if "block" in obj["name"]:
                self.assertEqual(obj["type"], "block")

    def test_bowl_type(self):
        bowl = [o for o in self.scene["objects"] if o["name"] == "bowl"][0]
        self.assertEqual(bowl["type"], "bowl")

    def test_block_colors(self):
        expected = {"red_block": "red", "green_block": "green", "blue_block": "blue"}
        for obj in self.scene["objects"]:
            if obj["name"] in expected:
                self.assertEqual(obj["color"], expected[obj["name"]])

    def test_position_length(self):
        for obj in self.scene["objects"]:
            self.assertEqual(len(obj["position"]), 3)

    def test_orientation_length(self):
        for obj in self.scene["objects"]:
            self.assertEqual(len(obj["orientation"]), 4)

    def test_id_is_int(self):
        for obj in self.scene["objects"]:
            self.assertIsInstance(obj["id"], int)

    def test_all_values_are_plain_python(self):
        """Positions and orientations should be plain Python lists, not numpy."""
        rs = self.scene["robot_state"]
        self.assertIsInstance(rs["end_effector_position"], list)
        self.assertIsInstance(rs["joint_positions"], list)
        for obj in self.scene["objects"]:
            self.assertIsInstance(obj["position"], list)
            self.assertIsInstance(obj["orientation"], list)


class TestObjectLookup(unittest.TestCase):
    """Test get_object_by_name and get_object_position."""

    def setUp(self):
        self.env = TableTopEnv(gui=False)
        self.env.reset()

    def tearDown(self):
        self.env.close()

    # -- get_object_by_name --

    def test_returns_int(self):
        body_id = self.env.get_object_by_name("red_block")
        self.assertIsInstance(body_id, int)

    def test_all_objects_found(self):
        for name in ("red_block", "green_block", "blue_block", "bowl"):
            body_id = self.env.get_object_by_name(name)
            self.assertEqual(body_id, self.env.objects[name])

    def test_unknown_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.env.get_object_by_name("purple_block")

    def test_error_message_lists_available(self):
        try:
            self.env.get_object_by_name("nope")
        except ValueError as exc:
            self.assertIn("red_block", str(exc))

    # -- get_object_position --

    def test_position_shape(self):
        pos = self.env.get_object_position("red_block")
        self.assertEqual(pos.shape, (3,))

    def test_position_on_table(self):
        for name in ("red_block", "green_block", "blue_block"):
            pos = self.env.get_object_position(name)
            # Block centre can sit slightly below TABLE_SIZE[2] due to settling
            self.assertGreater(pos[2], TableTopEnv.TABLE_SIZE[2] - 0.02)

    def test_position_within_table_bounds(self):
        cx, cy = TableTopEnv.TABLE_POSITION[0], TableTopEnv.TABLE_POSITION[1]
        hw = TableTopEnv.TABLE_SIZE[0] / 2
        hd = TableTopEnv.TABLE_SIZE[1] / 2
        for name in ("red_block", "green_block", "blue_block"):
            pos = self.env.get_object_position(name)
            self.assertGreater(pos[0], cx - hw)
            self.assertLess(pos[0], cx + hw)
            self.assertGreater(pos[1], cy - hd)
            self.assertLess(pos[1], cy + hd)

    def test_position_unknown_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.env.get_object_position("nonexistent")

    def test_blocks_not_overlapping(self):
        positions = [
            self.env.get_object_position(n)
            for n in ("red_block", "green_block", "blue_block")
        ]
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(positions[i][:2] - positions[j][:2])
                self.assertGreater(dist, 0.05)


class TestGripperWidth(unittest.TestCase):
    """Test get_gripper_width."""

    def setUp(self):
        self.env = TableTopEnv(gui=False)
        self.env.reset()

    def tearDown(self):
        self.env.close()

    def test_returns_float(self):
        self.assertIsInstance(self.env.get_gripper_width(), float)

    def test_open_after_reset(self):
        """After reset fingers are at 0.04 each → width ~ 0.08."""
        gw = self.env.get_gripper_width()
        self.assertAlmostEqual(gw, 0.08, places=3)


class TestRobotState(unittest.TestCase):
    """Test robot state queries."""

    def setUp(self):
        self.env = TableTopEnv(gui=False)
        self.env.reset()

    def tearDown(self):
        self.env.close()

    def test_ee_position_shape(self):
        self.assertEqual(self.env.get_ee_position().shape, (3,))

    def test_joint_positions_shape(self):
        self.assertEqual(self.env.get_joint_positions().shape, (7,))

    def test_home_joint_values(self):
        np.testing.assert_array_almost_equal(
            self.env.get_joint_positions(), TableTopEnv.HOME_JOINTS, decimal=2,
        )


class TestSimulation(unittest.TestCase):
    """Test simulation stepping."""

    def setUp(self):
        self.env = TableTopEnv(gui=False)
        self.env.reset()

    def tearDown(self):
        self.env.close()

    def test_step_single(self):
        self.env.step()  # default num_steps=1

    def test_step_multiple(self):
        self.env.step(num_steps=10)

    def test_step_seconds(self):
        positions_before = {
            name: self.env.get_object_position(name).copy()
            for name in self.env.objects
        }
        self.env.step_seconds(2.0)
        for name, before in positions_before.items():
            after = self.env.get_object_position(name)
            self.assertLess(np.linalg.norm(after - before), 0.1,
                            f"{name} drifted too far")

    def test_double_reset_cleans_old_objects(self):
        self.env.reset()
        self.assertEqual(len(self.env.objects), 4)


class TestPrintScene(unittest.TestCase):
    """Test print_scene produces readable output."""

    def setUp(self):
        self.env = TableTopEnv(gui=False)
        self.env.reset()

    def tearDown(self):
        self.env.close()

    def _capture(self) -> str:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.env.print_scene()
        return buf.getvalue()

    def test_does_not_crash(self):
        self.env.print_scene()

    def test_contains_scene_header(self):
        out = self._capture()
        self.assertIn("S C E N E", out)

    def test_contains_end_effector(self):
        out = self._capture()
        self.assertIn("End-Effector", out)

    def test_contains_gripper_width(self):
        out = self._capture()
        self.assertIn("Gripper Width", out)

    def test_contains_all_objects(self):
        out = self._capture()
        for name in ("red_block", "green_block", "blue_block", "bowl"):
            self.assertIn(name, out)

    def test_contains_coordinates(self):
        """Output should contain at least one coordinate like +0.500."""
        out = self._capture()
        self.assertRegex(out, r"[+-]\d+\.\d{3}")


class TestClose(unittest.TestCase):
    """Test clean shutdown."""

    def test_close_disconnects(self):
        import pybullet as p
        env = TableTopEnv(gui=False)
        client = env.physics_client
        env.close()
        self.assertFalse(p.isConnected(client))

    def test_double_close_safe(self):
        env = TableTopEnv(gui=False)
        env.close()
        env.close()  # should not raise


if __name__ == "__main__":
    unittest.main()

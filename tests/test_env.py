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


class TestSpawnObject(unittest.TestCase):
    """Test spawn_object with registries."""

    def setUp(self):
        self.env = TableTopEnv(gui=False)
        self.env.reset()

    def tearDown(self):
        self.env.close()

    def test_spawn_block(self):
        body_id = self.env.spawn_object("test_block", "block", "yellow", [0.5, 0.0, 0.07])
        self.assertIsInstance(body_id, int)
        self.assertIn("test_block", self.env.objects)

    def test_spawn_cylinder(self):
        body_id = self.env.spawn_object("cyl1", "cylinder", "orange", [0.5, 0.2, 0.08])
        self.assertIsInstance(body_id, int)

    def test_spawn_sphere(self):
        body_id = self.env.spawn_object("ball1", "sphere", "purple", [0.4, 0.1, 0.075])
        self.assertIsInstance(body_id, int)

    def test_spawn_obstacle(self):
        body_id = self.env.spawn_object("wall1", "obstacle", "black", [0.7, 0.0, 0.10])
        self.assertIsInstance(body_id, int)

    def test_meta_populated(self):
        self.env.spawn_object("test_obj", "block", "green", [0.5, 0.0, 0.07])
        self.assertIn("test_obj", self.env.OBJECT_META)
        self.assertEqual(self.env.OBJECT_META["test_obj"]["type"], "block")
        self.assertEqual(self.env.OBJECT_META["test_obj"]["color"], "green")

    def test_appears_in_scene(self):
        self.env.spawn_object("yellow_block", "block", "yellow", [0.5, 0.0, 0.07])
        scene = self.env.get_scene_description()
        names = {o["name"] for o in scene["objects"]}
        self.assertIn("yellow_block", names)

    def test_unknown_type_raises(self):
        with self.assertRaises(ValueError):
            self.env.spawn_object("x", "rocket", "red", [0.5, 0.0, 0.07])

    def test_unknown_color_raises(self):
        with self.assertRaises(ValueError):
            self.env.spawn_object("x", "block", "magenta", [0.5, 0.0, 0.07])

    def test_duplicate_name_raises(self):
        with self.assertRaises(ValueError):
            self.env.spawn_object("red_block", "block", "red", [0.5, 0.0, 0.07])

    def test_static_mass_zero(self):
        """Static objects (mass=0) should not move."""
        self.env.spawn_object("static_box", "block", "white", [0.5, 0.2, 0.07], mass=0)
        pos_before = self.env.get_object_position("static_box").copy()
        self.env.step(100)
        pos_after = self.env.get_object_position("static_box")
        np.testing.assert_array_almost_equal(pos_before, pos_after, decimal=3)


class TestDynamicObjects(unittest.TestCase):
    """Test add_object, remove_object, list_objects."""

    def setUp(self):
        self.env = TableTopEnv(gui=False)
        self.env.reset()

    def tearDown(self):
        self.env.close()

    def test_add_object_default_position(self):
        body_id = self.env.add_object("new_block", "block", "yellow")
        self.assertIn("new_block", self.env.objects)
        self.assertIsInstance(body_id, int)

    def test_add_object_custom_position(self):
        self.env.add_object("custom", "sphere", "orange", position=[0.6, 0.1, 0.1])
        pos = self.env.get_object_position("custom")
        self.assertAlmostEqual(pos[0], 0.6, places=1)

    def test_remove_object(self):
        self.env.remove_object("red_block")
        self.assertNotIn("red_block", self.env.objects)
        self.assertNotIn("red_block", self.env.OBJECT_META)

    def test_remove_nonexistent_raises(self):
        with self.assertRaises(ValueError):
            self.env.remove_object("nonexistent")

    def test_list_objects(self):
        objs = self.env.list_objects()
        self.assertEqual(len(objs), 4)
        names = {o["name"] for o in objs}
        self.assertEqual(names, {"red_block", "green_block", "blue_block", "bowl"})

    def test_list_objects_after_add(self):
        self.env.add_object("extra", "cylinder", "purple")
        objs = self.env.list_objects()
        self.assertEqual(len(objs), 5)

    def test_list_objects_after_remove(self):
        self.env.remove_object("bowl")
        objs = self.env.list_objects()
        self.assertEqual(len(objs), 3)


if __name__ == "__main__":
    unittest.main()

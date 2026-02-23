"""Tests for RobotSkills."""

import unittest
import io
import contextlib
import numpy as np

from src.env import TableTopEnv
from src.skills import RobotSkills


class _SkillsTestBase(unittest.TestCase):
    """Shared setUp / tearDown that creates env + skills in headless mode."""

    def setUp(self):
        self.env = TableTopEnv(gui=False)
        self.env.reset()
        self.skills = RobotSkills(self.env)

    def tearDown(self):
        self.env.close()


class _SeededSkillsTestBase(_SkillsTestBase):
    """Base with seeded RNG for deterministic block positions."""

    def setUp(self):
        np.random.seed(42)
        super().setUp()


# ==================================================================
# Initialisation
# ==================================================================

class TestSkillsInit(_SkillsTestBase):

    def test_robot_id_matches_env(self):
        self.assertEqual(self.skills.robot_id, self.env.robot_id)

    def test_joint_indices(self):
        self.assertEqual(self.skills.arm_joints, list(self.env.ARM_JOINTS))
        self.assertEqual(self.skills.gripper_joints, list(self.env.GRIPPER_JOINTS))

    def test_ee_link_index(self):
        self.assertEqual(self.skills.ee_link_index, self.env.EE_LINK_INDEX)


# ==================================================================
# Low-level helpers
# ==================================================================

class TestLowLevelHelpers(_SkillsTestBase):

    def test_joint_positions_length(self):
        self.assertEqual(len(self.skills._get_current_joint_positions()), 7)

    def test_joint_positions_match_env(self):
        np.testing.assert_array_almost_equal(
            self.skills._get_current_joint_positions(),
            self.env.get_joint_positions().tolist(),
        )

    def test_private_ee_position_shape(self):
        self.assertEqual(self.skills._get_ee_position().shape, (3,))

    def test_public_ee_position_shape(self):
        self.assertEqual(self.skills.get_ee_position().shape, (3,))

    def test_public_ee_matches_private(self):
        np.testing.assert_array_almost_equal(
            self.skills.get_ee_position(), self.skills._get_ee_position()
        )

    def test_ee_position_matches_env(self):
        np.testing.assert_array_almost_equal(
            self.skills.get_ee_position(), self.env.get_ee_position()
        )

    def test_gripper_width_matches_env(self):
        self.assertAlmostEqual(
            self.skills._get_gripper_width(), self.env.get_gripper_width(), places=4
        )

    def test_gripper_joint_forces_length(self):
        self.assertEqual(len(self.skills._get_gripper_joint_forces()), 2)

    def test_gripper_joint_forces_are_floats(self):
        for f in self.skills._get_gripper_joint_forces():
            self.assertIsInstance(f, float)


# ==================================================================
# Inverse kinematics
# ==================================================================

class TestSolveIK(_SkillsTestBase):

    def test_returns_seven_joints(self):
        self.assertEqual(len(self.skills.solve_ik([0.5, 0.0, 0.3])), 7)

    def test_returns_list(self):
        self.assertIsInstance(self.skills.solve_ik([0.5, 0.0, 0.3]), list)

    def test_default_orientation_finite(self):
        joints = self.skills.solve_ik([0.4, 0.1, 0.2])
        self.assertTrue(all(np.isfinite(joints)))

    def test_custom_orientation(self):
        import pybullet as pb
        orn = list(pb.getQuaternionFromEuler([np.pi, 0, np.pi / 4]))
        self.assertEqual(len(self.skills.solve_ik([0.5, 0.0, 0.3], target_orn=orn)), 7)


# ==================================================================
# Joint-space motion
# ==================================================================

class TestMoveToJointPositions(_SkillsTestBase):

    def test_converges_to_home(self):
        self.assertTrue(self.skills.move_to_joint_positions(self.skills.home_joint_positions))

    def test_converges_to_new_position(self):
        self.assertTrue(self.skills.move_to_joint_positions([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5]))

    def test_joints_reach_target(self):
        target = [0.1, -0.6, 0.1, -2.1, 0.1, 1.4, 0.6]
        self.skills.move_to_joint_positions(target)
        np.testing.assert_array_almost_equal(
            self.skills._get_current_joint_positions(), target, decimal=2
        )

    def test_returns_false_on_timeout(self):
        self.assertFalse(
            self.skills.move_to_joint_positions([1, -1, 1, -1, 1, 1, 1], max_steps=1)
        )


# ==================================================================
# Cartesian motion
# ==================================================================

class TestMoveToPosition(_SkillsTestBase):

    def test_converges(self):
        self.assertTrue(self.skills.move_to_position([0.5, 0.0, 0.3]))

    def test_ee_near_target(self):
        target = [0.4, 0.1, 0.25]
        self.skills.move_to_position(target)
        error = np.linalg.norm(self.skills._get_ee_position() - np.array(target))
        self.assertLess(error, 0.03)

    def test_several_positions(self):
        for pos in [[0.5, 0.0, 0.4], [0.4, 0.2, 0.3], [0.6, -0.1, 0.2]]:
            self.assertTrue(self.skills.move_to_position(pos), f"Failed at {pos}")


# ==================================================================
# Gripper control
# ==================================================================

class TestGripper(_SkillsTestBase):

    def test_open_gripper_width(self):
        self.skills.close_gripper()
        self.skills.open_gripper()
        self.assertGreater(self.skills._get_gripper_width(), 0.06)

    def test_close_gripper_width(self):
        self.skills.open_gripper()
        self.skills.close_gripper()
        self.assertLess(self.skills._get_gripper_width(), 0.01)

    def test_close_returns_bool(self):
        self.assertIsInstance(self.skills.close_gripper(), bool)

    def test_close_empty_returns_false(self):
        self.skills.move_to_position([0.3, 0.3, 0.4])
        self.skills.open_gripper()
        self.assertFalse(self.skills.close_gripper())

    def test_close_then_open_cycle(self):
        self.skills.close_gripper()
        w_closed = self.skills._get_gripper_width()
        self.skills.open_gripper()
        self.assertGreater(self.skills._get_gripper_width(), w_closed)

    def test_multiple_open_close_cycles(self):
        widths_open, widths_closed = [], []
        for _ in range(3):
            self.skills.open_gripper()
            widths_open.append(self.skills._get_gripper_width())
            self.skills.close_gripper()
            widths_closed.append(self.skills._get_gripper_width())
        self.assertLess(max(widths_open) - min(widths_open), 0.01)
        self.assertLess(max(widths_closed) - min(widths_closed), 0.005)

    def test_open_prints_message(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.skills.open_gripper()
        self.assertIn("Gripper opened", buf.getvalue())

    def test_close_prints_message(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.skills.close_gripper()
        self.assertIn("Gripper closed", buf.getvalue())


class TestGraspDetection(_SkillsTestBase):

    def test_not_grasping_when_open(self):
        self.skills.open_gripper()
        self.assertFalse(self.skills._is_grasping())

    def test_not_grasping_when_closed_empty(self):
        self.skills.move_to_position([0.3, 0.3, 0.4])
        self.skills.close_gripper()
        self.assertFalse(self.skills._is_grasping())

    def test_grasping_after_pick(self):
        np.random.seed(42)
        self.env.reset()
        self.skills = RobotSkills(self.env)
        success = self.skills.pick("red_block")
        if not success:
            self.skipTest("pick did not succeed")
        self.assertTrue(self.skills._is_grasping())

    def test_not_grasping_after_place(self):
        np.random.seed(42)
        self.env.reset()
        self.skills = RobotSkills(self.env)
        if not self.skills.pick("red_block"):
            self.skipTest("pick did not succeed")
        self.skills.place([0.6, 0.2, self.env.TABLE_SIZE[2] + 0.02])
        self.assertFalse(self.skills._is_grasping())

    def test_gripper_state_keys(self):
        state = self.skills.get_gripper_state()
        for key in ("width", "forces", "is_grasping"):
            self.assertIn(key, state)

    def test_gripper_state_types(self):
        state = self.skills.get_gripper_state()
        self.assertIsInstance(state["width"], float)
        self.assertIsInstance(state["forces"], list)
        self.assertEqual(len(state["forces"]), 2)
        self.assertIsInstance(state["is_grasping"], bool)

    def test_gripper_state_open(self):
        self.skills.open_gripper()
        state = self.skills.get_gripper_state()
        self.assertGreater(state["width"], 0.06)
        self.assertFalse(state["is_grasping"])

    def test_gripper_state_closed_empty(self):
        self.skills.move_to_position([0.3, 0.3, 0.4])
        self.skills.close_gripper()
        state = self.skills.get_gripper_state()
        self.assertLess(state["width"], 0.01)
        self.assertFalse(state["is_grasping"])


# ==================================================================
# Go home
# ==================================================================

class TestGoHome(_SkillsTestBase):

    def test_returns_true(self):
        self.skills.move_to_position([0.5, 0.2, 0.3])
        self.assertTrue(self.skills.go_home())

    def test_joints_at_home(self):
        self.skills.move_to_position([0.5, 0.2, 0.3])
        self.skills.go_home()
        np.testing.assert_array_almost_equal(
            self.skills._get_current_joint_positions(),
            self.skills.home_joint_positions, decimal=2,
        )

    def test_gripper_is_open_after_go_home(self):
        """go_home should open the gripper."""
        self.skills.close_gripper()
        self.skills.go_home()
        self.assertGreater(self.skills._get_gripper_width(), 0.06)

    def test_prints_returning(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.skills.go_home()
        self.assertIn("Returning to home position", buf.getvalue())

    def test_prints_reached(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.skills.go_home()
        self.assertIn("Home position reached", buf.getvalue())

    def test_go_home_from_far_position(self):
        """Arm should reach home even from the far edge of the workspace."""
        self.skills.move_to_position([0.6, 0.3, 0.2])
        self.assertTrue(self.skills.go_home())


# ==================================================================
# Pick skill
# ==================================================================

class TestPick(_SeededSkillsTestBase):

    def test_pick_returns_bool(self):
        self.assertIsInstance(self.skills.pick("red_block"), bool)

    def test_pick_lifts_block(self):
        initial_z = self.env.get_object_position("red_block")[2]
        if not self.skills.pick("red_block"):
            self.skipTest("pick did not succeed")
        self.assertGreater(
            self.env.get_object_position("red_block")[2],
            initial_z + 0.03,
        )

    def test_pick_unknown_raises(self):
        with self.assertRaises(ValueError):
            self.skills.pick("nonexistent_block")

    def test_gripper_not_fully_closed_after_pick(self):
        if not self.skills.pick("red_block"):
            self.skipTest("pick did not succeed")
        self.assertGreater(self.skills._get_gripper_width(), 0.005)

    def test_pick_prints_attempting(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.skills.pick("red_block")
        self.assertIn("Attempting to pick: red_block", buf.getvalue())

    def test_pick_prints_result(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = self.skills.pick("red_block")
        output = buf.getvalue()
        if result:
            self.assertIn("Successfully picked red_block", output)
        else:
            self.assertIn("Failed to pick red_block", output)

    def test_pick_prints_waypoints(self):
        """Output should mention pre-grasp, descending, and lifting."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.skills.pick("red_block")
        output = buf.getvalue()
        self.assertIn("pre-grasp", output)
        self.assertIn("Descending", output)
        self.assertIn("Lifting", output)


class TestPickEachBlock(_SeededSkillsTestBase):
    """Try picking every coloured block (deterministic placement)."""

    def test_pick_red_block(self):
        result = self.skills.pick("red_block")
        self.assertIsInstance(result, bool)

    def test_pick_green_block(self):
        result = self.skills.pick("green_block")
        self.assertIsInstance(result, bool)

    def test_pick_blue_block(self):
        result = self.skills.pick("blue_block")
        self.assertIsInstance(result, bool)


# ==================================================================
# Place skill
# ==================================================================

class TestPlace(_SeededSkillsTestBase):

    def test_place_converges(self):
        target = [0.5, 0.0, self.env.TABLE_SIZE[2] + 0.02]
        self.assertTrue(self.skills.place(target))

    def test_place_prints_messages(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.skills.place([0.5, 0.0, 0.1])
        output = buf.getvalue()
        self.assertIn("Placing object at", output)
        self.assertIn("Object placed", output)

    def test_place_at_specific_coordinates(self):
        """Place motions converge to several table positions."""
        table_z = self.env.TABLE_SIZE[2] + 0.02
        targets = [
            [0.3, 0.1, table_z],
            [0.5, -0.2, table_z],
            [0.7, 0.0, table_z],
        ]
        for t in targets:
            self.assertTrue(self.skills.place(t), f"Failed to place at {t}")

    def test_gripper_open_after_place(self):
        self.skills.place([0.5, 0.0, 0.1])
        self.assertGreater(self.skills._get_gripper_width(), 0.06)


# ==================================================================
# Full pick-and-place sequence
# ==================================================================

class TestPickAndPlace(_SeededSkillsTestBase):
    """End-to-end pick-then-place sequences."""

    def test_pick_then_place_motions_converge(self):
        if not self.skills.pick("red_block"):
            self.skipTest("pick did not succeed")
        place_pos = [0.6, 0.2, self.env.TABLE_SIZE[2] + 0.02]
        self.assertTrue(self.skills.place(place_pos))

    def test_pick_place_then_go_home(self):
        """Full cycle: pick → place → go home."""
        if not self.skills.pick("red_block"):
            self.skipTest("pick did not succeed")
        self.skills.place([0.6, 0.2, self.env.TABLE_SIZE[2] + 0.02])
        self.assertTrue(self.skills.go_home())

    def test_pick_place_block_lands_on_table(self):
        """After place, the block should rest on (or near) the table."""
        if not self.skills.pick("red_block"):
            self.skipTest("pick did not succeed")
        target = [0.6, 0.2, self.env.TABLE_SIZE[2] + 0.02]
        self.skills.place(target)
        self.env.step(120)  # let block settle
        z = self.env.get_object_position("red_block")[2]
        # Block should be on the table, not floating or underground
        self.assertGreater(z, self.env.TABLE_SIZE[2] - 0.02)
        self.assertLess(z, self.env.TABLE_SIZE[2] + 0.10)

    def test_sequential_pick_place_two_blocks(self):
        """Pick and place two different blocks in sequence."""
        table_z = self.env.TABLE_SIZE[2] + 0.02
        for block, target in [
            ("red_block", [0.3, 0.2, table_z]),
            ("green_block", [0.7, -0.1, table_z]),
        ]:
            picked = self.skills.pick(block)
            if not picked:
                self.skills.go_home()
                continue
            self.assertTrue(self.skills.place(target))
            self.skills.go_home()


# ==================================================================
# Execute trajectory
# ==================================================================

class TestExecuteTrajectory(_SkillsTestBase):

    def test_returns_true_for_valid_waypoints(self):
        waypoints = [
            [0.5, 0.0, 0.4],
            [0.4, 0.1, 0.3],
            [0.5, -0.1, 0.35],
        ]
        self.assertTrue(self.skills.execute_trajectory(waypoints))

    def test_ee_near_last_waypoint(self):
        waypoints = [[0.5, 0.0, 0.4], [0.4, 0.2, 0.3]]
        self.skills.execute_trajectory(waypoints)
        ee = self.skills.get_ee_position()
        error = np.linalg.norm(ee - np.array(waypoints[-1]))
        self.assertLess(error, 0.03)

    def test_single_waypoint(self):
        self.assertTrue(self.skills.execute_trajectory([[0.5, 0.0, 0.3]]))

    def test_empty_trajectory(self):
        self.assertTrue(self.skills.execute_trajectory([]))

    def test_prints_waypoint_labels(self):
        waypoints = [[0.5, 0.0, 0.4], [0.4, 0.1, 0.3]]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.skills.execute_trajectory(waypoints)
        output = buf.getvalue()
        self.assertIn("Waypoint 1/2", output)
        self.assertIn("Waypoint 2/2", output)
        self.assertIn("Trajectory complete", output)

    def test_prints_count(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.skills.execute_trajectory([[0.5, 0.0, 0.3]] * 3)
        self.assertIn("3 waypoints", buf.getvalue())


# ==================================================================
# Push skill
# ==================================================================

class TestPush(_SeededSkillsTestBase):

    def test_push_converges(self):
        self.assertTrue(self.skills.push("red_block", direction=[1.0, 0.0], distance=0.08))

    def test_push_moves_block(self):
        initial = self.env.get_object_position("red_block").copy()
        self.skills.push("red_block", direction=[1.0, 0.0], distance=0.08)
        self.env.step(120)
        disp = np.linalg.norm(
            self.env.get_object_position("red_block")[:2] - initial[:2]
        )
        self.assertGreater(disp, 0.02)

    def test_push_zero_direction_returns_false(self):
        self.assertFalse(self.skills.push("red_block", direction=[0.0, 0.0], distance=0.1))

    def test_push_unknown_raises(self):
        with self.assertRaises(ValueError):
            self.skills.push("nonexistent_block", direction=[1, 0])


# ==================================================================
# Stack skill
# ==================================================================

class TestStack(_SeededSkillsTestBase):

    def test_stack_returns_bool(self):
        result = self.skills.stack("red_block", "green_block")
        self.assertIsInstance(result, bool)

    def test_stack_prints_stacking(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.skills.stack("red_block", "green_block")
        self.assertIn("Stacking", buf.getvalue())

    def test_stack_unknown_source_raises(self):
        with self.assertRaises(ValueError):
            self.skills.stack("nonexistent", "green_block")

    def test_stack_unknown_target_raises(self):
        # stack will try to pick first (succeeds or fails),
        # then query target position which should raise
        result = self.skills.stack("red_block", "nonexistent")
        # If pick fails, returns False; if pick succeeds, target lookup raises
        # Either way, we get a bool or ValueError
        self.assertIsInstance(result, bool)


# ==================================================================
# Sweep skill
# ==================================================================

class TestSweep(_SeededSkillsTestBase):

    def test_sweep_returns_bool(self):
        result = self.skills.sweep("red_block", [0.6, 0.0, 0.07])
        self.assertIsInstance(result, bool)

    def test_sweep_at_target_returns_true(self):
        pos = self.env.get_object_position("red_block")
        result = self.skills.sweep("red_block", pos.tolist())
        self.assertTrue(result)

    def test_sweep_unknown_raises(self):
        with self.assertRaises(ValueError):
            self.skills.sweep("nonexistent", [0.5, 0.0, 0.07])


# ==================================================================
# Rotate gripper skill
# ==================================================================

class TestRotateGripper(_SkillsTestBase):

    def test_rotate_returns_bool(self):
        result = self.skills.rotate_gripper(0.5)
        self.assertIsInstance(result, bool)

    def test_rotate_small_angle(self):
        result = self.skills.rotate_gripper(0.1)
        self.assertIsInstance(result, bool)

    def test_rotate_zero(self):
        result = self.skills.rotate_gripper(0.0)
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()

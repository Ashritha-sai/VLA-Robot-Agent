"""
RobotSkills - High-level manipulation primitives for the Franka Panda.

Each skill (pick, place, push, go_home, ...) is a self-contained motion that
drives the arm through a sequence of waypoints using smooth joint-space
interpolation.  Skills are the building blocks that the VLA agent chains
together to accomplish language-described tasks.
"""

import pybullet as p
import numpy as np
import logging
from typing import List

from src.skills.base import RobotControlBase

logger = logging.getLogger(__name__)


class RobotSkills(RobotControlBase):
    """High-level manipulation skills for the Franka Panda robot."""

    # ------------------------------------------------------------------
    # High-level skills
    # ------------------------------------------------------------------

    def go_home(self, speed: float = 0.3) -> bool:
        """
        Return the arm to its safe home configuration.

        Sequence:
            1. Open the gripper so it doesn't collide on the way back.
            2. Drive all arm joints to ``home_joint_positions``.
            3. Let the robot settle for a few simulation steps.

        Args:
            speed: Position-control gain.

        Returns:
            ``True`` if the arm converged to the home pose.
        """
        print("Returning to home position...")
        logger.info("Going home")

        # 1. Open gripper for safety
        self.open_gripper()

        # 2. Move arm to home
        converged = self.move_to_joint_positions(
            self.home_joint_positions, speed=speed,
        )

        # 3. Settle
        self.env.step(10)

        if converged:
            print("Home position reached")
        else:
            print("Warning: home position not fully reached")
        return converged

    def pick(self, object_name: str) -> bool:
        """
        Pick up an object by name.

        Sequence:
            1. Query the object's position from the environment.
            2. Open the gripper.
            3. Move to a *pre-grasp* position above the object.
            4. Descend to the *grasp* position (slightly above object centre).
            5. Close the gripper and let the grasp stabilise.
            6. Lift the object.
            7. Verify the grasp held.

        Heights are controlled by class constants ``PRE_GRASP_HEIGHT``,
        ``GRASP_OFFSET``, and ``LIFT_HEIGHT`` -- all tuned for 0.04 m cubes
        on the standard table.

        Args:
            object_name: Semantic name (e.g. ``"red_block"``).

        Returns:
            ``True`` if all motions converged **and** the gripper is still
            holding the object after the lift.
        """
        print(f"Attempting to pick: {object_name}")

        # Step 1 -- object position
        object_pos = self.env.get_object_position(object_name)
        logger.info("pick('%s') at [%.3f, %.3f, %.3f]", object_name, *object_pos)

        # Step 2 -- open gripper
        self.open_gripper()

        # Step 3 -- pre-grasp (PRE_GRASP_HEIGHT above object)
        pre_grasp_pos = object_pos + np.array([0, 0, self.PRE_GRASP_HEIGHT])
        print(f"  Moving to pre-grasp: [{pre_grasp_pos[0]:.3f}, {pre_grasp_pos[1]:.3f}, {pre_grasp_pos[2]:.3f}]")
        if not self.move_to_position(pre_grasp_pos.tolist()):
            print(f"  Failed to reach pre-grasp position")
            return False

        # Step 4 -- descend to grasp height (GRASP_OFFSET above object centre)
        grasp_pos = object_pos + np.array([0, 0, self.GRASP_OFFSET])
        print(f"  Descending to grasp: [{grasp_pos[0]:.3f}, {grasp_pos[1]:.3f}, {grasp_pos[2]:.3f}]")
        if not self.move_to_position(grasp_pos.tolist()):
            print(f"  Failed to reach grasp position")
            return False

        # Step 5 -- close gripper and wait for stabilisation
        grasped = self.close_gripper()
        self.env.step(self.SETTLE_STEPS)

        # Step 6 -- lift object
        lift_pos = grasp_pos + np.array([0, 0, self.LIFT_HEIGHT])
        print(f"  Lifting to: [{lift_pos[0]:.3f}, {lift_pos[1]:.3f}, {lift_pos[2]:.3f}]")
        if not self.move_to_position(lift_pos.tolist()):
            print(f"  Failed to reach lift position")
            return False

        # Step 7 -- verify grasp survived the lift
        still_holding = self._is_grasping()
        success = grasped and still_holding

        if success:
            logger.info("pick('%s') succeeded (gripper_width=%.4f)",
                        object_name, self._get_gripper_width())
            print(f"  Successfully picked {object_name}")
        else:
            logger.warning("pick('%s') failed (grasped=%s, holding=%s, width=%.4f)",
                           object_name, grasped, still_holding, self._get_gripper_width())
            print(f"  Failed to pick {object_name}")
        return success

    def place(self, target_position: List[float]) -> bool:
        """
        Place the currently held object at *target_position*.

        Sequence:
            1. Move to a position above the target.
            2. Descend to the placement height.
            3. Open the gripper to release and let the object settle.
            4. Retreat upward.

        Heights are controlled by ``PLACE_APPROACH_HEIGHT`` and
        ``PLACE_OFFSET``.

        Args:
            target_position: ``[x, y, z]`` world-frame position to place
                             the object.

        Returns:
            ``True`` if all motions converged.
        """
        target = np.array(target_position, dtype=float)
        print(f"Placing object at: [{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}]")
        logger.info("place at [%.3f, %.3f, %.3f]", *target)

        # Step 1 -- above target
        above_target = target + np.array([0, 0, self.PLACE_APPROACH_HEIGHT])
        print(f"  Moving above target: [{above_target[0]:.3f}, {above_target[1]:.3f}, {above_target[2]:.3f}]")
        if not self.move_to_position(above_target.tolist()):
            print("  Failed to reach above-target position")
            return False

        # Step 2 -- descend to place height
        place_pos = target + np.array([0, 0, self.PLACE_OFFSET])
        print(f"  Descending to place: [{place_pos[0]:.3f}, {place_pos[1]:.3f}, {place_pos[2]:.3f}]")
        if not self.move_to_position(place_pos.tolist()):
            print("  Failed to reach place position")
            return False

        # Step 3 -- release
        self.open_gripper()
        self.env.step(self.SETTLE_STEPS)

        # Step 4 -- retreat
        print(f"  Retreating upward")
        if not self.move_to_position(above_target.tolist()):
            print("  Failed to retreat")
            return False

        logger.info("place completed")
        print("  Object placed")
        return True

    def push(
        self,
        object_name: str,
        direction: List[float],
        distance: float = 0.10,
    ) -> bool:
        """
        Push an object sideways along *direction* by *distance* metres.

        The gripper is closed, lowered to block height, then driven
        horizontally through the object.

        Args:
            object_name: Object to push.
            direction:   ``[dx, dy]`` push direction (normalised internally).
            distance:    How far to push (metres).

        Returns:
            ``True`` if all motions converged.
        """
        obj_pos = self.env.get_object_position(object_name)
        logger.info("push('%s') dir=[%.2f,%.2f] dist=%.3f",
                     object_name, direction[0], direction[1], distance)

        # Normalise direction to a unit vector
        d = np.array(direction[:2], dtype=float)
        norm = np.linalg.norm(d)
        if norm < 1e-6:
            logger.error("push direction is zero-length")
            return False
        d = d / norm

        # Close gripper so fingers don't catch on the block
        self.close_gripper()

        # Start position: slightly behind the object, at block height
        block_z = obj_pos[2]
        start_xy = obj_pos[:2] - d * 0.05  # 5 cm behind
        start = [float(start_xy[0]), float(start_xy[1]), block_z]

        # Approach from above
        above_start = [start[0], start[1], block_z + self.PRE_GRASP_HEIGHT]
        if not self.move_to_position(above_start):
            return False
        if not self.move_to_position(start):
            return False

        # Push through
        end_xy = obj_pos[:2] + d * distance
        end = [float(end_xy[0]), float(end_xy[1]), block_z]
        if not self.move_to_position(end, speed=0.15):
            return False

        # Retreat upward
        above_end = [end[0], end[1], block_z + self.PRE_GRASP_HEIGHT]
        if not self.move_to_position(above_end):
            return False

        logger.info("push completed")
        return True

    def execute_trajectory(self, waypoints: List[List[float]]) -> bool:
        """
        Execute a sequence of Cartesian waypoints.

        Each waypoint is reached via IK + joint-space motion before the
        next is attempted.  If any waypoint fails to converge, the
        trajectory is aborted.

        Args:
            waypoints: List of ``[x, y, z]`` positions.

        Returns:
            ``True`` if every waypoint was reached, ``False`` if any
            failed (partial trajectory may have been executed).
        """
        total = len(waypoints)
        print(f"Executing trajectory ({total} waypoints)")
        logger.info("execute_trajectory: %d waypoints", total)

        for i, wp in enumerate(waypoints):
            label = f"Waypoint {i + 1}/{total}"
            print(f"  {label}: [{wp[0]:.3f}, {wp[1]:.3f}, {wp[2]:.3f}]")
            if not self.move_to_position(wp):
                print(f"  {label} FAILED")
                logger.warning("execute_trajectory aborted at waypoint %d", i + 1)
                return False

        print("Trajectory complete")
        logger.info("execute_trajectory finished successfully")
        return True

    def stack(self, object_name: str, target_object_name: str) -> bool:
        """
        Stack *object_name* on top of *target_object_name*.

        Sequence:
            1. Pick the source object.
            2. Query the target object's current position.
            3. Place the source on top of the target (offset by block height).

        Args:
            object_name: Object to pick up.
            target_object_name: Object to stack on top of.

        Returns:
            ``True`` if pick and place both succeeded.
        """
        print(f"Stacking {object_name} on {target_object_name}")
        logger.info("stack('%s', '%s')", object_name, target_object_name)

        if not self.pick(object_name):
            print(f"  Failed to pick {object_name}")
            return False

        target_pos = self.env.get_object_position(target_object_name)
        # Place above the target object (one block height up)
        place_pos = [
            float(target_pos[0]),
            float(target_pos[1]),
            float(target_pos[2]) + self.BLOCK_HALF * 2,
        ]

        return self.place(place_pos)

    # Block half-extent used for stacking offset
    BLOCK_HALF = 0.02

    def sweep(self, object_name: str, target_position: List[float]) -> bool:
        """
        Sweep (push) an object toward a target position.

        Computes the direction from the object to the target and pushes
        for the appropriate distance.

        Args:
            object_name: Object to sweep.
            target_position: ``[x, y, z]`` or ``[x, y]`` destination.

        Returns:
            ``True`` if the push motion converged.
        """
        obj_pos = self.env.get_object_position(object_name)
        target = np.array(target_position[:2], dtype=float)
        direction = target - obj_pos[:2]
        distance = float(np.linalg.norm(direction))

        if distance < 0.01:
            print(f"  {object_name} is already at target")
            return True

        logger.info("sweep('%s') toward [%.3f, %.3f], dist=%.3f",
                     object_name, target[0], target[1], distance)
        print(f"Sweeping {object_name} toward [{target[0]:.3f}, {target[1]:.3f}]")

        return self.push(object_name, direction.tolist(), distance)

    def rotate_gripper(self, angle: float) -> bool:
        """
        Rotate the gripper about the Z-axis while keeping the EE position fixed.

        Args:
            angle: Rotation angle in radians (positive = counter-clockwise
                   when viewed from above).

        Returns:
            ``True`` if the motion converged.
        """
        # Get current EE pose
        ee_pos = self._get_ee_position().tolist()
        state = p.getLinkState(self.robot_id, self.ee_link_index)
        current_orn = state[1]
        current_euler = list(p.getEulerFromQuaternion(current_orn))

        # Rotate about Z
        current_euler[2] += angle
        new_orn = list(p.getQuaternionFromEuler(current_euler))

        logger.info("rotate_gripper by %.3f rad", angle)
        print(f"Rotating gripper by {angle:.3f} rad")

        return self.move_to_position(ee_pos, target_orn=new_orn)

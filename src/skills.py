"""
RobotSkills - High-level manipulation primitives for the Franka Panda.

Each skill (pick, place, push, go_home, …) is a self-contained motion that
drives the arm through a sequence of waypoints using smooth joint-space
interpolation.  Skills are the building blocks that the VLA agent chains
together to accomplish language-described tasks.
"""

import pybullet as p
import numpy as np
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class RobotSkills:
    """High-level manipulation skills for the Franka Panda robot."""

    # ---- Franka Panda joint limits (radians) ----
    LOWER_LIMITS = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
    UPPER_LIMITS = [ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973]
    JOINT_RANGES = [ul - ll for ul, ll in zip(UPPER_LIMITS, LOWER_LIMITS)]

    # ---- Motion defaults ----
    POSITION_GAIN = 0.3          # kp for position-control motors
    MAX_VELOCITY = 1.0           # rad/s cap sent to motor controller
    GRIPPER_FORCE = 100.0        # Newton grip force
    CONVERGENCE_TOL = 0.01       # rad – per-joint convergence tolerance
    MAX_MOVE_STEPS = 720         # safety cap (3 s at 240 Hz)

    # ---- Pick / place heights (tuned for 0.04 m cubes on a 0.05 m table) ----
    PRE_GRASP_HEIGHT = 0.10      # metres above object for approach
    GRASP_OFFSET = 0.02          # metres above object centre for grasp
    LIFT_HEIGHT = 0.15           # metres above grasp point after picking
    PLACE_APPROACH_HEIGHT = 0.15 # metres above target for place approach
    PLACE_OFFSET = 0.05          # metres above target z for release
    SETTLE_STEPS = 20            # sim steps to let grasp / release stabilise

    def __init__(self, env):
        """
        Initialise skills bound to a specific environment.

        Args:
            env: A ``TableTopEnv`` instance (must already be connected).
        """
        self.env = env
        self.robot_id = env.robot_id

        # Pull constants from the environment so there is a single source
        self.arm_joints = list(env.ARM_JOINTS)
        self.gripper_joints = list(env.GRIPPER_JOINTS)
        self.ee_link_index = env.EE_LINK_INDEX
        self.home_joint_positions = list(env.HOME_JOINTS)

        # Gripper state: tracks whether the last command was "close"
        self._gripper_closed_command = False

        # Increase friction on gripper finger links for more reliable grasps
        for joint_idx in self.gripper_joints:
            p.changeDynamics(self.robot_id, joint_idx, lateralFriction=1.5)

        # Constraint-based grasp: body id and constraint id of the
        # object currently attached to the end-effector.
        self._grasp_constraint_id: Optional[int] = None
        self._grasped_body_id: Optional[int] = None

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _get_current_joint_positions(self) -> List[float]:
        """Return current arm joint angles as a plain Python list."""
        return [p.getJointState(self.robot_id, j)[0] for j in self.arm_joints]

    def _get_ee_position(self) -> np.ndarray:
        """Return end-effector world position as (3,) array."""
        state = p.getLinkState(self.robot_id, self.ee_link_index)
        return np.array(state[0])

    def get_ee_position(self) -> np.ndarray:
        """
        Get current end-effector position in world frame.

        Uses ``p.getLinkState`` on the EE link to read its world-frame
        translation.

        Returns:
            ``(3,)`` numpy array ``[x, y, z]``.
        """
        return self._get_ee_position()

    def _get_gripper_width(self) -> float:
        """Sum of both finger joint positions (total opening)."""
        return sum(p.getJointState(self.robot_id, j)[0] for j in self.gripper_joints)

    def _get_gripper_joint_forces(self) -> List[float]:
        """
        Read the motor torques currently applied to each gripper finger.

        ``p.getJointState`` index 3 is ``appliedJointMotorTorque``.

        Returns:
            List of two floats (one per finger).
        """
        return [p.getJointState(self.robot_id, j)[3] for j in self.gripper_joints]

    # ------------------------------------------------------------------
    # Inverse kinematics
    # ------------------------------------------------------------------

    def solve_ik(
        self,
        target_pos: List[float],
        target_orn: Optional[List[float]] = None,
    ) -> List[float]:
        """
        Compute arm joint angles that place the end-effector at the target.

        Args:
            target_pos: ``[x, y, z]`` desired end-effector position.
            target_orn: ``[qx, qy, qz, qw]`` desired orientation quaternion.
                        Defaults to gripper pointing straight down.

        Returns:
            List of 7 joint angles.
        """
        # Default: gripper pointing straight down (Z-axis of EE faces -Z world)
        if target_orn is None:
            target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])

        joint_positions = p.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self.ee_link_index,
            targetPosition=target_pos,
            targetOrientation=target_orn,
            lowerLimits=self.LOWER_LIMITS,
            upperLimits=self.UPPER_LIMITS,
            jointRanges=self.JOINT_RANGES,
            restPoses=self.home_joint_positions,
            maxNumIterations=100,
            residualThreshold=1e-5,
        )

        # p.calculateInverseKinematics returns values for all movable joints;
        # we only need the first 7 (arm joints).
        return list(joint_positions[: len(self.arm_joints)])

    # ------------------------------------------------------------------
    # Joint-space motion
    # ------------------------------------------------------------------

    def move_to_joint_positions(
        self,
        target_joints: List[float],
        speed: float = 0.3,
        max_steps: Optional[int] = None,
    ) -> bool:
        """
        Drive the arm to *target_joints* with smooth position control.

        Uses ``p.setJointMotorControlArray`` in ``POSITION_CONTROL`` mode
        at every simulation step.  Returns when every joint is within
        ``CONVERGENCE_TOL`` of its target or after *max_steps*.

        Args:
            target_joints: 7 target joint angles (rad).
            speed:         Position-control gain (higher → faster).
            max_steps:     Safety cap on simulation steps.

        Returns:
            ``True`` if converged within tolerance, ``False`` if the step
            limit was reached.
        """
        if max_steps is None:
            max_steps = self.MAX_MOVE_STEPS

        target = np.array(target_joints, dtype=float)

        for step_i in range(max_steps):
            # Command position controllers for all arm joints at once.
            p.setJointMotorControlArray(
                bodyUniqueId=self.robot_id,
                jointIndices=self.arm_joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=target.tolist(),
                positionGains=[speed] * len(self.arm_joints),
                velocityGains=[1.0] * len(self.arm_joints),
            )

            self.env.step()

            # Check convergence
            current = np.array(self._get_current_joint_positions())
            if np.all(np.abs(current - target) < self.CONVERGENCE_TOL):
                logger.debug("move_to_joint_positions converged in %d steps", step_i + 1)
                return True

        logger.warning(
            "move_to_joint_positions did not converge after %d steps "
            "(max error=%.4f rad)",
            max_steps,
            float(np.max(np.abs(np.array(self._get_current_joint_positions()) - target))),
        )
        return False

    # ------------------------------------------------------------------
    # Cartesian motion
    # ------------------------------------------------------------------

    def move_to_position(
        self,
        target_pos: List[float],
        target_orn: Optional[List[float]] = None,
        speed: float = 0.3,
    ) -> bool:
        """
        Move the end-effector to a Cartesian position via IK.

        Args:
            target_pos: ``[x, y, z]`` world-frame position.
            target_orn: Optional orientation quaternion.
            speed:      Position-control gain.

        Returns:
            ``True`` if the joint motion converged.
        """
        joints = self.solve_ik(target_pos, target_orn)
        logger.info("Moving EE to [%.3f, %.3f, %.3f]", *target_pos)
        return self.move_to_joint_positions(joints, speed=speed)

    # ------------------------------------------------------------------
    # Gripper control
    # ------------------------------------------------------------------

    # Width thresholds for grasp detection
    GRASP_WIDTH_THRESHOLD = 0.005   # m – below this the gripper is "empty-closed"
    GRIPPER_OPEN_WIDTH = 0.04       # m – per-finger target when fully open
    GRIPPER_SETTLE_STEPS = 60       # sim steps to let fingers reach target

    def _attach_object(self, body_id: int) -> None:
        """Create a fixed constraint attaching *body_id* to the end-effector.

        This prevents the object from slipping out of the gripper during
        transport — a standard workaround for PyBullet friction limits.
        Uses JOINT_POINT2POINT (position-only) to avoid rotational
        energy buildup that JOINT_FIXED can cause.
        """
        self._detach_object()  # release any previous grasp

        ee_state = p.getLinkState(self.robot_id, self.ee_link_index)
        ee_pos = ee_state[0]
        ee_orn = ee_state[1]

        obj_pos, _ = p.getBasePositionAndOrientation(body_id)

        # Compute object position in EE-local frame
        inv_ee_pos, inv_ee_orn = p.invertTransform(ee_pos, ee_orn)
        rel_pos, _ = p.multiplyTransforms(inv_ee_pos, inv_ee_orn,
                                          obj_pos, [0, 0, 0, 1])

        self._grasp_constraint_id = p.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=self.ee_link_index,
            childBodyUniqueId=body_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=rel_pos,
            childFramePosition=[0, 0, 0],
        )
        p.changeConstraint(self._grasp_constraint_id, maxForce=50)
        self._grasped_body_id = body_id
        logger.debug("Attached body %d to EE with constraint %d",
                      body_id, self._grasp_constraint_id)

    def _detach_object(self) -> None:
        """Remove the grasp constraint, releasing the held object.

        Gradually reduces constraint force, then removes the constraint
        and zeroes the object velocity to prevent flinging.
        """
        if self._grasp_constraint_id is not None:
            body = self._grasped_body_id
            # Ramp down the constraint force to avoid energy release
            for force in [20, 5, 0]:
                p.changeConstraint(self._grasp_constraint_id, maxForce=force)
                for _ in range(5):
                    p.stepSimulation()
            # Zero velocity before and after removal
            p.resetBaseVelocity(body, [0, 0, 0], [0, 0, 0])
            p.removeConstraint(self._grasp_constraint_id)
            p.resetBaseVelocity(body, [0, 0, 0], [0, 0, 0])
            # Let physics settle
            for _ in range(10):
                p.stepSimulation()
            p.resetBaseVelocity(body, [0, 0, 0], [0, 0, 0])
            logger.debug("Removed grasp constraint %d (body %d)",
                          self._grasp_constraint_id, body)
            self._grasp_constraint_id = None
            self._grasped_body_id = None

    def _find_grasped_body(self) -> Optional[int]:
        """Identify which object body is between the gripper fingers.

        Checks contact points on both finger links and returns the body
        that *both* fingers are touching (i.e. the grasped object).
        Returns *None* if no common contact is found.
        """
        sets: List[set] = []
        for joint_idx in self.gripper_joints:
            contacts = p.getContactPoints(bodyA=self.robot_id,
                                          linkIndexA=joint_idx)
            sets.append({c[2] for c in contacts if c[2] != self.robot_id})
        if len(sets) == 2:
            common = sets[0] & sets[1]
            if common:
                return common.pop()
        return None

    def open_gripper(self) -> None:
        """
        Open the gripper to its maximum width (≈ 0.08 m total).

        Each finger is driven to ``GRIPPER_OPEN_WIDTH`` (0.04 m) using
        ``p.setJointMotorControl2`` in ``POSITION_CONTROL`` mode.
        The simulation is stepped ``GRIPPER_SETTLE_STEPS`` times so the
        fingers have time to move before the caller continues.
        """
        self._detach_object()
        self._drive_gripper(self.GRIPPER_OPEN_WIDTH)
        self._gripper_closed_command = False
        gw = self._get_gripper_width()
        logger.info("Gripper opened (width=%.4f m)", gw)
        print(f"Gripper opened (width={gw:.4f} m)")

    def close_gripper(self) -> bool:
        """
        Close the gripper to grasp an object.

        Fingers are commanded to 0.0 m with ``GRIPPER_FORCE``.  If an
        object is between the fingers the force controller will stall,
        leaving a measurable residual width.

        After closing, grasp detection runs automatically—see
        ``_is_grasping``.

        Returns:
            ``True`` if the gripper appears to be holding an object.
        """
        self._drive_gripper(0.0)
        self._gripper_closed_command = True
        gw = self._get_gripper_width()
        grasping = self._is_grasping()

        # If a grasp is detected, attach the object with a fixed constraint
        # so it cannot slip during transport.
        if grasping:
            body = self._find_grasped_body()
            if body is not None:
                self._attach_object(body)

        status = "grasping object" if grasping else "empty"
        logger.info("Gripper closed (width=%.4f m, %s)", gw, status)
        print(f"Gripper closed (width={gw:.4f} m, {status})")
        return grasping

    def _drive_gripper(self, target_width_per_finger: float) -> None:
        """
        Command both gripper fingers to *target_width_per_finger*.

        Uses ``p.setJointMotorControl2`` per finger (gives explicit
        per-joint force control) and runs ``GRIPPER_SETTLE_STEPS``
        simulation steps so the fingers reach the target or stall on an
        object.
        """
        for _ in range(self.GRIPPER_SETTLE_STEPS):
            for joint_idx in self.gripper_joints:
                p.setJointMotorControl2(
                    bodyUniqueId=self.robot_id,
                    jointIndex=joint_idx,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target_width_per_finger,
                    force=self.GRIPPER_FORCE,
                )
            self.env.step()

    def _is_grasping(self) -> bool:
        """
        Check whether the gripper is currently holding an object.

        The check is only meaningful after a ``close_gripper`` command.
        If the gripper was last commanded to open, this returns ``False``
        immediately.

        Two complementary heuristics are used when the gripper is closed:

        1. **Width check** – a residual opening wider than
           ``GRASP_WIDTH_THRESHOLD`` means something is caught between
           the fingers (they could not close fully).
        2. **Contact-point check** – ``p.getContactPoints`` is queried
           for each finger link.  If *both* fingers report contact with
           the same non-robot body, a grasp is confirmed.

        Either heuristic being true is sufficient (contacts can be
        transient; width can be ambiguous for very small objects).

        Returns:
            ``True`` if the gripper appears to be holding an object.
        """
        # If the last command was "open", the gripper is not grasping.
        if not self._gripper_closed_command:
            return False

        # --- Heuristic 1: residual width ---
        gw = self._get_gripper_width()
        width_indicates_grasp = gw > self.GRASP_WIDTH_THRESHOLD

        # --- Heuristic 2: contact points on both fingers ---
        bodies_per_finger: List[set] = []
        for joint_idx in self.gripper_joints:
            # p.getContactPoints returns contacts involving (bodyA, linkA).
            # We query contacts where our robot is bodyA and the finger
            # link is linkIndexA.
            contacts = p.getContactPoints(bodyA=self.robot_id, linkIndexA=joint_idx)
            # Collect the *other* body IDs that are in contact.
            touched = {c[2] for c in contacts if c[2] != self.robot_id}
            bodies_per_finger.append(touched)

        # An object is grasped if both fingers touch the same non-robot body.
        if len(bodies_per_finger) == 2:
            common = bodies_per_finger[0] & bodies_per_finger[1]
            contact_indicates_grasp = len(common) > 0
        else:
            contact_indicates_grasp = False

        is_grasping = width_indicates_grasp or contact_indicates_grasp

        logger.debug(
            "_is_grasping: width=%.4f (%s), contacts=%s (%s) -> %s",
            gw,
            width_indicates_grasp,
            [sorted(s) for s in bodies_per_finger],
            contact_indicates_grasp,
            is_grasping,
        )
        return is_grasping

    def get_gripper_state(self) -> dict:
        """
        Return a snapshot of the gripper's current state.

        Useful for debugging and logging.

        Returns:
            Dictionary with keys ``width``, ``forces``, ``is_grasping``.
        """
        return {
            "width": self._get_gripper_width(),
            "forces": self._get_gripper_joint_forces(),
            "is_grasping": self._is_grasping(),
        }

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
        ``GRASP_OFFSET``, and ``LIFT_HEIGHT`` — all tuned for 0.04 m cubes
        on the standard table.

        Args:
            object_name: Semantic name (e.g. ``"red_block"``).

        Returns:
            ``True`` if all motions converged **and** the gripper is still
            holding the object after the lift.
        """
        print(f"Attempting to pick: {object_name}")

        # Step 1 — object position
        object_pos = self.env.get_object_position(object_name)
        logger.info("pick('%s') at [%.3f, %.3f, %.3f]", object_name, *object_pos)

        # Step 2 — open gripper
        self.open_gripper()

        # Step 3 — pre-grasp (PRE_GRASP_HEIGHT above object)
        pre_grasp_pos = object_pos + np.array([0, 0, self.PRE_GRASP_HEIGHT])
        print(f"  Moving to pre-grasp: [{pre_grasp_pos[0]:.3f}, {pre_grasp_pos[1]:.3f}, {pre_grasp_pos[2]:.3f}]")
        if not self.move_to_position(pre_grasp_pos.tolist()):
            print(f"  Failed to reach pre-grasp position")
            return False

        # Step 4 — descend to grasp height (GRASP_OFFSET above object centre)
        grasp_pos = object_pos + np.array([0, 0, self.GRASP_OFFSET])
        print(f"  Descending to grasp: [{grasp_pos[0]:.3f}, {grasp_pos[1]:.3f}, {grasp_pos[2]:.3f}]")
        if not self.move_to_position(grasp_pos.tolist()):
            print(f"  Failed to reach grasp position")
            return False

        # Step 5 — close gripper and wait for stabilisation
        grasped = self.close_gripper()
        self.env.step(self.SETTLE_STEPS)

        # Step 6 — lift object
        lift_pos = grasp_pos + np.array([0, 0, self.LIFT_HEIGHT])
        print(f"  Lifting to: [{lift_pos[0]:.3f}, {lift_pos[1]:.3f}, {lift_pos[2]:.3f}]")
        if not self.move_to_position(lift_pos.tolist()):
            print(f"  Failed to reach lift position")
            return False

        # Step 7 — verify grasp survived the lift
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

        # Step 1 — above target
        above_target = target + np.array([0, 0, self.PLACE_APPROACH_HEIGHT])
        print(f"  Moving above target: [{above_target[0]:.3f}, {above_target[1]:.3f}, {above_target[2]:.3f}]")
        if not self.move_to_position(above_target.tolist()):
            print("  Failed to reach above-target position")
            return False

        # Step 2 — descend to place height
        place_pos = target + np.array([0, 0, self.PLACE_OFFSET])
        print(f"  Descending to place: [{place_pos[0]:.3f}, {place_pos[1]:.3f}, {place_pos[2]:.3f}]")
        if not self.move_to_position(place_pos.tolist()):
            print("  Failed to reach place position")
            return False

        # Step 3 — release
        self.open_gripper()
        self.env.step(self.SETTLE_STEPS)

        # Step 4 — retreat
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

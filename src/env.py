"""
TableTopEnv - PyBullet simulation environment for tabletop manipulation.

Provides a Franka Panda robot arm on a table with colored blocks and a bowl.
Used as the physical simulation backend for the VLA Robot Agent.
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class TableTopEnv:
    """PyBullet simulation environment for tabletop manipulation."""

    # Franka Panda home joint positions (radians).
    # These place the arm in a safe, upright posture above the table.
    HOME_JOINTS = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]

    # Joint indices on the Franka Panda URDF
    ARM_JOINTS = [0, 1, 2, 3, 4, 5, 6]
    GRIPPER_JOINTS = [9, 10]
    EE_LINK_INDEX = 11

    # Table geometry
    TABLE_SIZE = [1.0, 0.8, 0.05]          # half-extents used below
    TABLE_POSITION = [0.5, 0.0, 0.0]       # centre of table top surface
    TABLE_HEIGHT = TABLE_SIZE[2]            # top surface z (half-height placed at z=0 → surface at 0.05)

    # Object geometry
    BLOCK_HALF = 0.02                       # 0.04 m cube → half-extent 0.02
    BOWL_RADIUS = 0.1
    BOWL_HEIGHT = 0.05

    # Block colours (RGBA)
    BLOCK_COLORS: Dict[str, List[float]] = {
        "red_block":   [1.0, 0.0, 0.0, 1.0],
        "green_block": [0.0, 1.0, 0.0, 1.0],
        "blue_block":  [0.0, 0.0, 1.0, 1.0],
    }

    # Metadata used by get_scene_description to produce semantic labels
    OBJECT_META: Dict[str, Dict[str, str]] = {
        "red_block":   {"type": "block", "color": "red"},
        "green_block": {"type": "block", "color": "green"},
        "blue_block":  {"type": "block", "color": "blue"},
        "bowl":        {"type": "bowl",  "color": "gray"},
    }

    def __init__(self, gui: bool = True, time_step: float = 1.0 / 240.0):
        """
        Initialise the PyBullet simulation.

        Args:
            gui: If True, open a graphical window. If False, run headless (DIRECT).
            time_step: Physics time-step in seconds (default 240 Hz).
        """
        self.gui = gui
        self.time_step = time_step

        # --- 1. Connect to physics server ---
        if gui:
            self.physics_client = p.connect(p.GUI)
            logger.info("Connected to PyBullet GUI (client %d)", self.physics_client)
        else:
            self.physics_client = p.connect(p.DIRECT)
            logger.info("Connected to PyBullet DIRECT (client %d)", self.physics_client)

        # --- 2. Add pybullet_data path so built-in URDFs can be found ---
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # --- 3. Configure physics ---
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(time_step)

        # --- 4. Load ground plane ---
        # plane.urdf is shipped with pybullet_data
        self.plane_id = p.loadURDF("plane.urdf")
        logger.info("Loaded ground plane (id %d)", self.plane_id)

        # --- 5. Create table ---
        self.table_id = self._create_table()

        # --- 6. Load Franka Panda robot ---
        self.robot_id = self._load_robot()

        # --- 7. Camera (only meaningful in GUI mode) ---
        if gui:
            # Position camera to look at centre of table from a comfortable angle
            p.resetDebugVisualizerCamera(
                cameraDistance=1.2,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0.5, 0.0, 0.2],
            )

        # --- 8. Object tracking ---
        self.objects: Dict[str, int] = {}

        logger.info("TableTopEnv initialised (time_step=%.6f s)", time_step)

    # ------------------------------------------------------------------
    # Scene construction helpers
    # ------------------------------------------------------------------

    def _create_table(self) -> int:
        """
        Create a table using primitive collision / visual shapes.

        The table is a flat box centred at TABLE_POSITION, sitting on the
        ground plane.  Its top surface is at z = TABLE_SIZE[2] (0.05 m).

        Returns:
            PyBullet body id of the table.
        """
        half_extents = [
            self.TABLE_SIZE[0] / 2.0,   # 0.5
            self.TABLE_SIZE[1] / 2.0,   # 0.4
            self.TABLE_SIZE[2] / 2.0,   # 0.025
        ]

        col_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
        )
        vis_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[0.6, 0.4, 0.2, 1.0],  # brown / wood colour
        )

        # Place so the top surface is at z = TABLE_SIZE[2]
        # Centre of box is at z = half_extents[2]
        table_z = half_extents[2]
        table_id = p.createMultiBody(
            baseMass=0,  # static (infinite mass)
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=[
                self.TABLE_POSITION[0],
                self.TABLE_POSITION[1],
                table_z,
            ],
        )
        logger.info("Created table (id %d) at z=%.3f", table_id, table_z)
        return table_id

    def _load_robot(self) -> int:
        """
        Load the Franka Panda robot from pybullet_data.

        The robot base is placed at the origin so its workspace covers the
        table area.

        Returns:
            PyBullet body id of the robot.
        """
        robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0, 0, 0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,  # robot base is bolted down
        )
        logger.info("Loaded Franka Panda (id %d)", robot_id)

        # Assign before calling _reset_robot so it can reference self.robot_id
        self.robot_id = robot_id
        self._reset_robot()
        return robot_id

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> Dict[str, Dict]:
        """
        Reset the environment to a fresh episode.

        1. Remove any previously spawned objects.
        2. Reset robot joints to the home configuration.
        3. Spawn blocks and bowl.
        4. Let physics settle.
        5. Return a scene description dict.

        Returns:
            Dictionary mapping object names to their poses, e.g.::

                {
                    "red_block": {"position": [x, y, z], "orientation": [x, y, z, w]},
                    ...
                }
        """
        # --- Remove old objects ---
        for name, body_id in self.objects.items():
            p.removeBody(body_id)
            logger.debug("Removed object '%s' (id %d)", name, body_id)
        self.objects.clear()

        # --- Reset robot ---
        self._reset_robot()

        # --- Spawn new objects ---
        self._spawn_blocks()
        self._spawn_bowl()

        # --- Let objects settle under gravity ---
        for _ in range(50):
            p.stepSimulation()

        logger.info("Environment reset complete (%d objects)", len(self.objects))
        return self.get_scene_description()

    # ------------------------------------------------------------------
    # Object spawning
    # ------------------------------------------------------------------

    def _spawn_blocks(self) -> None:
        """
        Spawn three coloured cubes at random positions on the table.

        Each block is a 0.04 m cube.  Positions are randomised within the
        table bounds with a margin so blocks don't fall off the edge, and a
        minimum separation so they don't overlap.
        """
        half = self.BLOCK_HALF  # 0.02

        # Collision shape shared by all blocks (same geometry)
        col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half, half, half])

        # Margin from table edge and minimum spacing between block centres
        margin = 0.1
        min_spacing = 0.10  # metres

        # Table bounds for random placement
        x_lo = self.TABLE_POSITION[0] - self.TABLE_SIZE[0] / 2 + margin
        x_hi = self.TABLE_POSITION[0] + self.TABLE_SIZE[0] / 2 - margin
        y_lo = self.TABLE_POSITION[1] - self.TABLE_SIZE[1] / 2 + margin
        y_hi = self.TABLE_POSITION[1] + self.TABLE_SIZE[1] / 2 - margin

        # z so the block sits on top of the table surface
        block_z = self.TABLE_SIZE[2] + half

        placed_positions: List[np.ndarray] = []

        for name, rgba in self.BLOCK_COLORS.items():
            vis_shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[half, half, half],
                rgbaColor=rgba,
            )

            # Keep generating random (x, y) until we find one that doesn't
            # collide with already-placed blocks.
            for _ in range(100):
                x = np.random.uniform(x_lo, x_hi)
                y = np.random.uniform(y_lo, y_hi)
                pos = np.array([x, y])
                if all(np.linalg.norm(pos - prev) >= min_spacing for prev in placed_positions):
                    break
            else:
                logger.warning("Could not find non-overlapping position for '%s'", name)

            body_id = p.createMultiBody(
                baseMass=0.1,  # 100 g
                baseCollisionShapeIndex=col_shape,
                baseVisualShapeIndex=vis_shape,
                basePosition=[x, y, block_z],
            )

            # Increase lateral friction so the gripper can hold the block
            p.changeDynamics(body_id, -1, lateralFriction=1.5)

            self.objects[name] = body_id
            placed_positions.append(pos)
            logger.debug("Spawned '%s' (id %d) at [%.3f, %.3f, %.3f]", name, body_id, x, y, block_z)

    def _spawn_bowl(self) -> None:
        """
        Spawn a bowl (approximated as a cylinder) at a fixed position on the table.

        The bowl sits at the centre of the table.
        """
        col_shape = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=self.BOWL_RADIUS,
            height=self.BOWL_HEIGHT,
        )
        vis_shape = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=self.BOWL_RADIUS,
            length=self.BOWL_HEIGHT,
            rgbaColor=[0.8, 0.8, 0.8, 1.0],  # light grey
        )

        bowl_z = self.TABLE_SIZE[2] + self.BOWL_HEIGHT / 2.0

        body_id = p.createMultiBody(
            baseMass=0,  # static
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=[
                self.TABLE_POSITION[0],
                self.TABLE_POSITION[1],
                bowl_z,
            ],
        )

        self.objects["bowl"] = body_id
        logger.debug("Spawned bowl (id %d) at [%.3f, %.3f, %.3f]",
                      body_id, self.TABLE_POSITION[0], self.TABLE_POSITION[1], bowl_z)

    # ------------------------------------------------------------------
    # Robot helpers
    # ------------------------------------------------------------------

    def _reset_robot(self) -> None:
        """
        Reset the Franka Panda to its home configuration.

        Uses ``p.resetJointState`` (instantaneous, no physics step needed)
        to place each arm joint and open the gripper fingers.
        """
        # Arm joints → home position
        for joint_idx, angle in zip(self.ARM_JOINTS, self.HOME_JOINTS):
            p.resetJointState(self.robot_id, joint_idx, angle)

        # Gripper fingers → fully open (each finger at 0.04 m)
        for joint_idx in self.GRIPPER_JOINTS:
            p.resetJointState(self.robot_id, joint_idx, 0.04)

        logger.debug("Robot reset to home configuration")

    # ------------------------------------------------------------------
    # Observation / state queries
    # ------------------------------------------------------------------

    def get_scene_description(self) -> Dict:
        """
        Build a rich semantic description of the current scene.

        Returns:
            Dictionary with two top-level keys::

                {
                    "robot_state": {
                        "end_effector_position": [x, y, z],
                        "gripper_width": float,
                        "joint_positions": [j1, ..., j7]
                    },
                    "objects": [
                        {
                            "name": "red_block",
                            "type": "block",
                            "color": "red",
                            "position": [x, y, z],
                            "orientation": [qx, qy, qz, qw],
                            "id": int
                        },
                        ...
                    ]
                }
        """
        # --- Robot state ---
        ee_pos = self.get_ee_position()
        gripper_w = self.get_gripper_width()
        joint_pos = self.get_joint_positions()

        robot_state = {
            "end_effector_position": ee_pos.tolist(),
            "gripper_width": gripper_w,
            "joint_positions": joint_pos.tolist(),
        }

        # --- Object list ---
        objects_list: List[Dict] = []
        for name, body_id in self.objects.items():
            pos, orn = p.getBasePositionAndOrientation(body_id)
            meta = self.OBJECT_META.get(name, {"type": "unknown", "color": "unknown"})
            objects_list.append({
                "name": name,
                "type": meta["type"],
                "color": meta["color"],
                "position": list(pos),
                "orientation": list(orn),
                "id": body_id,
            })

        return {"robot_state": robot_state, "objects": objects_list}

    def get_object_by_name(self, name: str) -> int:
        """
        Look up a PyBullet body ID by semantic object name.

        Args:
            name: Object name (e.g. ``"red_block"``, ``"bowl"``).

        Returns:
            PyBullet body ID.

        Raises:
            ValueError: If *name* is not a tracked object.
        """
        if name not in self.objects:
            raise ValueError(
                f"Object '{name}' not found. Available: {list(self.objects.keys())}"
            )
        return self.objects[name]

    def get_object_position(self, name: str) -> np.ndarray:
        """
        Get the world-frame position of a tracked object.

        Args:
            name: Object name (e.g. ``"red_block"``).

        Returns:
            (3,) numpy array ``[x, y, z]``.

        Raises:
            ValueError: If the object name is not tracked.
        """
        body_id = self.get_object_by_name(name)
        pos, _ = p.getBasePositionAndOrientation(body_id)
        return np.array(pos)

    def get_ee_position(self) -> np.ndarray:
        """
        Get the current position of the robot's end-effector link.

        Returns:
            (3,) numpy array ``[x, y, z]``.
        """
        state = p.getLinkState(self.robot_id, self.EE_LINK_INDEX)
        return np.array(state[0])  # worldLinkFramePosition

    def get_gripper_width(self) -> float:
        """
        Get the current distance between the two gripper fingers.

        Each finger joint reports its displacement from closed.  The total
        opening width is the sum of both finger positions.

        Returns:
            Gripper opening width in metres.
        """
        left = p.getJointState(self.robot_id, self.GRIPPER_JOINTS[0])[0]
        right = p.getJointState(self.robot_id, self.GRIPPER_JOINTS[1])[0]
        return left + right

    def get_joint_positions(self) -> np.ndarray:
        """
        Get the current positions of all 7 arm joints.

        Returns:
            (7,) numpy array of joint angles in radians.
        """
        return np.array([p.getJointState(self.robot_id, j)[0] for j in self.ARM_JOINTS])

    # ------------------------------------------------------------------
    # Simulation control
    # ------------------------------------------------------------------

    def step(self, num_steps: int = 1) -> None:
        """
        Advance the physics simulation.

        Args:
            num_steps: Number of simulation steps to execute.

        When running in GUI mode a small ``time.sleep`` is inserted per
        step so the visualisation runs at approximately real-time.
        """
        sleep_dt = self.time_step if self.gui else 0.0
        for _ in range(num_steps):
            p.stepSimulation()
            if sleep_dt:
                time.sleep(sleep_dt)

    def step_seconds(self, seconds: float) -> None:
        """
        Advance the simulation by the given duration.

        Args:
            seconds: Number of simulated seconds to advance.
        """
        n_steps = int(seconds / self.time_step)
        self.step(n_steps)

    def close(self) -> None:
        """Disconnect from the PyBullet physics server and release resources."""
        if p.isConnected(self.physics_client):
            p.disconnect(self.physics_client)
            logger.info("Disconnected from PyBullet (client %d)", self.physics_client)

    # ------------------------------------------------------------------
    # Pretty-printing
    # ------------------------------------------------------------------

    # Icons used by print_scene (maps object name → emoji/label)
    _PRINT_ICONS: Dict[str, str] = {
        "red_block":   "[RED]",
        "green_block": "[GRN]",
        "blue_block":  "[BLU]",
        "bowl":        "[BWL]",
    }

    def print_scene(self) -> None:
        """
        Print a human-readable, colour-coded scene summary to the console.

        Uses *colorama* for cross-platform terminal colours.  Falls back to
        plain text if colorama is not installed.
        """
        try:
            from colorama import Fore, Style, init as colorama_init
            colorama_init()
        except ImportError:
            # Provide no-op colour stubs so the rest of the method works.
            class _NoColor:
                def __getattr__(self, _: str) -> str:
                    return ""
            Fore = Style = _NoColor()  # type: ignore[assignment]

        scene = self.get_scene_description()
        robot = scene["robot_state"]

        ee = robot["end_effector_position"]
        gw = robot["gripper_width"]
        joints = robot["joint_positions"]

        print(f"\n{Style.BRIGHT}{'=' * 40}")
        print(f"           S C E N E")
        print(f"{'=' * 40}{Style.RESET_ALL}")

        print(f"\n  {Style.BRIGHT}Robot{Style.RESET_ALL}")
        print(f"    End-Effector : [{ee[0]:+.3f}, {ee[1]:+.3f}, {ee[2]:+.3f}]")
        print(f"    Gripper Width: {gw:.4f} m")
        print(f"    Joints       : [{', '.join(f'{j:+.3f}' for j in joints)}]")

        # Colour map for block names
        color_map = {
            "red":   Fore.RED,
            "green": Fore.GREEN,
            "blue":  Fore.BLUE,
            "gray":  Fore.WHITE,
        }

        print(f"\n  {Style.BRIGHT}Objects{Style.RESET_ALL}")
        for obj in scene["objects"]:
            c = color_map.get(obj["color"], "")
            icon = self._PRINT_ICONS.get(obj["name"], "[???]")
            pos = obj["position"]
            print(
                f"    {c}{icon} {obj['name']:<14}{Style.RESET_ALL}"
                f"  at [{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}]"
            )

        print()

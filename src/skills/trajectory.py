"""
TrajectoryRecorder - Record and replay end-effector trajectories.

Captures time-stamped end-effector positions during live control,
saves/loads them as JSON, and replays through RobotSkills.
"""

import json
import time
import logging
import threading
import numpy as np
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class TrajectoryRecorder:
    """Record, save, load, and replay end-effector trajectories."""

    def __init__(self, env):
        """
        Args:
            env: A TableTopEnv instance.
        """
        self.env = env
        self._frames: List[Dict] = []
        self._recording = False
        self._capture_thread: Optional[threading.Thread] = None
        self._capture_interval = 0.05  # seconds between captures

    @property
    def frame_count(self) -> int:
        """Number of recorded frames."""
        return len(self._frames)

    @property
    def frames(self) -> List[Dict]:
        """Return a copy of the recorded frames."""
        return list(self._frames)

    def start(self) -> None:
        """Start recording end-effector positions in a background thread."""
        if self._recording:
            logger.warning("Recording already in progress")
            return
        self._frames.clear()
        self._recording = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        logger.info("Trajectory recording started")

    def stop(self) -> None:
        """Stop recording."""
        self._recording = False
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None
        logger.info("Trajectory recording stopped (%d frames)", len(self._frames))

    def capture(self) -> Dict:
        """Capture a single frame (EE position + gripper width + timestamp)."""
        import pybullet as p
        state = p.getLinkState(self.env.robot_id, self.env.EE_LINK_INDEX)
        ee_pos = list(state[0])
        gripper_width = self.env.get_gripper_width()
        frame = {
            "timestamp": time.time(),
            "ee_position": ee_pos,
            "gripper_width": gripper_width,
        }
        self._frames.append(frame)
        return frame

    def _capture_loop(self) -> None:
        """Background thread that captures frames at fixed intervals."""
        while self._recording:
            self.capture()
            time.sleep(self._capture_interval)

    def save(self, path: str) -> None:
        """Save recorded frames to a JSON file.

        Args:
            path: File path to write.
        """
        data = {
            "frame_count": len(self._frames),
            "frames": self._frames,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved %d frames to %s", len(self._frames), path)

    def load(self, path: str) -> None:
        """Load frames from a JSON file.

        Args:
            path: File path to read.
        """
        with open(path, "r") as f:
            data = json.load(f)
        self._frames = data.get("frames", [])
        logger.info("Loaded %d frames from %s", len(self._frames), path)

    def replay(self, skills, speed: float = 0.3) -> bool:
        """
        Replay recorded trajectory through RobotSkills.

        Args:
            skills: A RobotSkills instance.
            speed: Motion speed for move_to_position.

        Returns:
            True if all waypoints were reached.
        """
        if not self._frames:
            logger.warning("No frames to replay")
            return True

        waypoints = [f["ee_position"] for f in self._frames]
        # Subsample if too many frames for efficiency
        max_waypoints = 50
        if len(waypoints) > max_waypoints:
            indices = np.linspace(0, len(waypoints) - 1, max_waypoints, dtype=int)
            waypoints = [waypoints[i] for i in indices]

        return skills.execute_trajectory(waypoints)

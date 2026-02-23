"""
Skills package -- re-exports RobotSkills for backwards compatibility.

Usage::

    from src.skills import RobotSkills
    from src.skills.trajectory import TrajectoryRecorder
"""

from src.skills.manipulation import RobotSkills
from src.skills.trajectory import TrajectoryRecorder

__all__ = ["RobotSkills", "TrajectoryRecorder"]

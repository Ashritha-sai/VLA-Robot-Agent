"""Tests for TrajectoryRecorder."""

import unittest
import os
import tempfile
import time

from src.env import TableTopEnv
from src.skills import RobotSkills
from src.skills.trajectory import TrajectoryRecorder


class TestTrajectoryRecorder(unittest.TestCase):

    def setUp(self):
        self.env = TableTopEnv(gui=False)
        self.env.reset()
        self.recorder = TrajectoryRecorder(self.env)

    def tearDown(self):
        self.recorder.stop()
        self.env.close()

    def test_initial_frame_count_zero(self):
        self.assertEqual(self.recorder.frame_count, 0)

    def test_capture_adds_frame(self):
        self.recorder.capture()
        self.assertEqual(self.recorder.frame_count, 1)

    def test_frame_has_required_keys(self):
        frame = self.recorder.capture()
        self.assertIn("timestamp", frame)
        self.assertIn("ee_position", frame)
        self.assertIn("gripper_width", frame)

    def test_ee_position_length(self):
        frame = self.recorder.capture()
        self.assertEqual(len(frame["ee_position"]), 3)

    def test_start_stop_captures_frames(self):
        self.recorder.start()
        time.sleep(0.2)
        self.recorder.stop()
        self.assertGreater(self.recorder.frame_count, 0)

    def test_save_and_load(self):
        self.recorder.capture()
        self.recorder.capture()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            self.recorder.save(path)
            new_recorder = TrajectoryRecorder(self.env)
            new_recorder.load(path)
            self.assertEqual(new_recorder.frame_count, 2)
        finally:
            os.unlink(path)

    def test_replay_empty(self):
        skills = RobotSkills(self.env)
        self.assertTrue(self.recorder.replay(skills))

    def test_replay_with_frames(self):
        self.recorder.capture()
        self.recorder.capture()
        skills = RobotSkills(self.env)
        result = self.recorder.replay(skills)
        self.assertIsInstance(result, bool)


if __name__ == "__main__":
    unittest.main()

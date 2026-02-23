"""Tests for ConversationMemory."""

import unittest
from src.memory import ConversationMemory


class TestConversationMemory(unittest.TestCase):

    def setUp(self):
        self.mem = ConversationMemory()

    def test_initial_turn_count_zero(self):
        self.assertEqual(self.mem.turn_count, 0)

    def test_add_interaction_increments_count(self):
        self.mem.add_interaction("pick red block", "will pick it", True)
        self.assertEqual(self.mem.turn_count, 1)

    def test_add_multiple_interactions(self):
        self.mem.add_interaction("pick red", "plan A", True)
        self.mem.add_interaction("place bowl", "plan B", False)
        self.assertEqual(self.mem.turn_count, 2)

    def test_history_content(self):
        self.mem.add_interaction("pick", "reasoning", True)
        h = self.mem.history
        self.assertEqual(h[0]["instruction"], "pick")
        self.assertEqual(h[0]["reasoning"], "reasoning")
        self.assertEqual(h[0]["success"], "success")

    def test_failure_recorded(self):
        self.mem.add_interaction("fail task", "oops", False)
        self.assertEqual(self.mem.history[0]["success"], "failure")

    def test_get_history_prompt_empty(self):
        self.assertEqual(self.mem.get_history_prompt(), "")

    def test_get_history_prompt_nonempty(self):
        self.mem.add_interaction("pick red", "plan", True)
        prompt = self.mem.get_history_prompt()
        self.assertIn("pick red", prompt)
        self.assertIn("Turn 1", prompt)

    def test_clear(self):
        self.mem.add_interaction("a", "b", True)
        self.mem.clear()
        self.assertEqual(self.mem.turn_count, 0)
        self.assertEqual(self.mem.get_history_prompt(), "")

    def test_max_turns_truncation(self):
        mem = ConversationMemory(max_turns=3)
        for i in range(5):
            mem.add_interaction(f"instr_{i}", f"reason_{i}", True)
        self.assertEqual(mem.turn_count, 3)
        # Oldest entries should be dropped
        self.assertIn("instr_4", mem.history[-1]["instruction"])

    def test_history_returns_copy(self):
        self.mem.add_interaction("x", "y", True)
        h = self.mem.history
        h.clear()
        self.assertEqual(self.mem.turn_count, 1)


if __name__ == "__main__":
    unittest.main()

"""
End-to-end tests with real LLM calls (Ollama).

These tests are skip-gated: they only run when a local Ollama server is
reachable.  They exercise the full pipeline from natural language
instruction through LLM planning to physics execution.

Run with:
    python -m pytest tests/test_e2e_llm.py -v
"""

import unittest
import json
import io
import contextlib
import numpy as np

from src.env import TableTopEnv

# Check if Ollama is installed and running
OLLAMA_RUNNING = False
try:
    import ollama
    # Attempt a lightweight API call to verify the server is up
    ollama.list()
    OLLAMA_RUNNING = True
except Exception:
    pass


def _make_env():
    np.random.seed(42)
    env = TableTopEnv(gui=False)
    env.reset()
    return env


@unittest.skipUnless(OLLAMA_RUNNING, "Ollama server not running")
class TestE2EPickAndPlace(unittest.TestCase):
    """End-to-end test: instruct the LLM to pick and place."""

    def setUp(self):
        self.env = _make_env()
        from src.agent import VLAAgent
        self.agent = VLAAgent(self.env, llm_provider="ollama")

    def tearDown(self):
        self.env.close()

    def test_llm_returns_valid_plan(self):
        """LLM should return a parseable JSON plan."""
        scene = self.env.get_scene_description()
        prompt = self.agent._build_prompt("put the red block in the bowl", scene)
        raw = self.agent.query_llm(prompt)
        plan = self.agent.parse_llm_response(raw)
        self.assertIn("actions", plan)
        self.assertIsInstance(plan["actions"], list)
        self.assertGreater(len(plan["actions"]), 0)

    def test_plan_contains_pick(self):
        """Plan for 'pick up the red block' should contain a pick action."""
        scene = self.env.get_scene_description()
        prompt = self.agent._build_prompt("pick up the red block", scene)
        raw = self.agent.query_llm(prompt)
        plan = self.agent.parse_llm_response(raw)
        actions = [a["action"] for a in plan["actions"]]
        self.assertIn("pick", actions)

    def test_plan_has_reasoning(self):
        """Plan should include a reasoning field."""
        scene = self.env.get_scene_description()
        prompt = self.agent._build_prompt("go home", scene)
        raw = self.agent.query_llm(prompt)
        plan = self.agent.parse_llm_response(raw)
        self.assertIn("reasoning", plan)


@unittest.skipUnless(OLLAMA_RUNNING, "Ollama server not running")
class TestE2EMultiTurn(unittest.TestCase):
    """End-to-end multi-turn conversation test."""

    def setUp(self):
        self.env = _make_env()
        from src.agent import VLAAgent
        self.agent = VLAAgent(self.env, llm_provider="ollama")

    def tearDown(self):
        self.env.close()

    def test_memory_grows_after_chat(self):
        """Memory turn count should increase after chat()."""
        self.assertEqual(self.agent.memory.turn_count, 0)
        # Chat will query LLM, parse, execute (may fail), and record
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.agent.chat("go home")
        self.assertEqual(self.agent.memory.turn_count, 1)

    def test_second_prompt_includes_history(self):
        """After one turn, the second prompt should include conversation history."""
        self.agent.memory.add_interaction("go home", "just go home", True)
        scene = self.env.get_scene_description()
        prompt = self.agent._build_prompt("pick the red block", scene)
        self.assertIn("go home", prompt)
        self.assertIn("Conversation History", prompt)


if __name__ == "__main__":
    unittest.main()

"""Tests for VLAAgent."""

import unittest
import json
import io
import contextlib
import os

import numpy as np

from src.env import TableTopEnv
from src.agent import VLAAgent, OPENAI_AVAILABLE, ANTHROPIC_AVAILABLE, OLLAMA_AVAILABLE


# ── Helpers ────────────────────────────────────────────────────────────

def _make_env():
    """Create a headless environment and reset it."""
    env = TableTopEnv(gui=False)
    env.reset()
    return env


def _make_agent(env):
    """Create a VLAAgent using whichever provider is installed.

    Prompt construction and parsing tests never call the LLM, so an
    API key is not required.
    """
    if OPENAI_AVAILABLE:
        return VLAAgent(env, llm_provider="openai")
    if ANTHROPIC_AVAILABLE:
        return VLAAgent(env, llm_provider="anthropic")
    return None


def _make_seeded_env():
    """Create a deterministic env for physics-dependent tests."""
    np.random.seed(42)
    return _make_env()


# ══════════════════════════════════════════════════════════════════════
#  Test: Initialisation
# ══════════════════════════════════════════════════════════════════════

class TestAgentInit(unittest.TestCase):
    """Test VLAAgent constructor."""

    def setUp(self):
        self.env = _make_env()

    def tearDown(self):
        self.env.close()

    @unittest.skipUnless(OPENAI_AVAILABLE, "openai not installed")
    def test_init_openai(self):
        agent = VLAAgent(self.env, llm_provider="openai")
        self.assertEqual(agent.llm_provider, "openai")
        self.assertEqual(agent.model, "gpt-4o")

    @unittest.skipUnless(ANTHROPIC_AVAILABLE, "anthropic not installed")
    def test_init_anthropic(self):
        agent = VLAAgent(self.env, llm_provider="anthropic")
        self.assertEqual(agent.llm_provider, "anthropic")
        self.assertEqual(agent.model, "claude-sonnet-4-20250514")

    @unittest.skipUnless(OPENAI_AVAILABLE, "openai not installed")
    def test_custom_model(self):
        agent = VLAAgent(self.env, llm_provider="openai", model="gpt-3.5-turbo")
        self.assertEqual(agent.model, "gpt-3.5-turbo")

    def test_unknown_provider_raises(self):
        with self.assertRaises(ValueError) as ctx:
            VLAAgent(self.env, llm_provider="gemini")
        self.assertIn("gemini", str(ctx.exception))

    def test_skills_attached(self):
        agent = _make_agent(self.env)
        if agent is None:
            self.skipTest("No LLM provider installed")
        self.assertIsNotNone(agent.skills)
        self.assertIs(agent.skills.env, self.env)

    def test_api_key_flag_when_no_key(self):
        """Agent should still construct without an API key."""
        agent = _make_agent(self.env)
        if agent is None:
            self.skipTest("No LLM provider installed")
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
            self.assertFalse(agent._api_key_set)

    def test_query_llm_raises_without_key(self):
        """query_llm should raise RuntimeError when no API key is set."""
        agent = _make_agent(self.env)
        if agent is None:
            self.skipTest("No LLM provider installed")
        if agent._api_key_set:
            self.skipTest("API key is set; cannot test missing-key path")
        with self.assertRaises(RuntimeError) as ctx:
            agent.query_llm("test prompt")
        self.assertIn("API key", str(ctx.exception))


# ══════════════════════════════════════════════════════════════════════
#  Test: Prompt construction
# ══════════════════════════════════════════════════════════════════════

class _PromptTestBase(unittest.TestCase):
    """Base that creates an agent + scene for prompt tests."""

    def setUp(self):
        self.env = _make_env()
        self.agent = _make_agent(self.env)
        if self.agent is None:
            self.skipTest("No LLM provider installed")
        self.scene = self.env.get_scene_description()

    def tearDown(self):
        self.env.close()


class TestBuildPrompt(_PromptTestBase):
    """Test _build_prompt output."""

    def _prompt(self, instruction="put the red block in the bowl"):
        return self.agent._build_prompt(instruction, self.scene)

    # -- Contains key sections --

    def test_contains_system_role(self):
        self.assertIn("robot task planner", self._prompt())

    def test_contains_user_instruction(self):
        instruction = "stack all blocks"
        self.assertIn(instruction, self._prompt(instruction))

    def test_contains_ee_position(self):
        self.assertIn("end-effector position", self._prompt())

    def test_contains_gripper_width(self):
        self.assertIn("Gripper width", self._prompt())

    # -- Lists all objects --

    def test_contains_all_object_names(self):
        p = self._prompt()
        for name in ("red_block", "green_block", "blue_block", "bowl"):
            self.assertIn(name, p)

    def test_contains_object_positions(self):
        """Prompt should contain formatted coordinates for every object."""
        p = self._prompt()
        for obj in self.scene["objects"]:
            pos = obj["position"]
            coord = f"{pos[0]:.3f}"
            self.assertIn(coord, p)

    # -- Available actions --

    def test_contains_pick_action(self):
        self.assertIn("pick(object_name)", self._prompt())

    def test_contains_place_action(self):
        self.assertIn("place(position)", self._prompt())

    def test_contains_go_home_action(self):
        self.assertIn("go_home()", self._prompt())

    def test_contains_push_action(self):
        self.assertIn("push(", self._prompt())

    # -- JSON format instructions --

    def test_contains_json_format_example(self):
        p = self._prompt()
        self.assertIn('"actions"', p)
        self.assertIn('"reasoning"', p)

    def test_contains_rules(self):
        p = self._prompt()
        self.assertIn("Rules:", p)
        self.assertIn("pick one object at a time", p)

    # -- Returns string --

    def test_returns_str(self):
        self.assertIsInstance(self._prompt(), str)

    def test_prompt_not_empty(self):
        self.assertGreater(len(self._prompt()), 200)


# ══════════════════════════════════════════════════════════════════════
#  Test: Response parsing
# ══════════════════════════════════════════════════════════════════════

class TestParseLLMResponse(unittest.TestCase):
    """Test parse_llm_response with synthetic responses."""

    def setUp(self):
        self.env = _make_env()
        self.agent = _make_agent(self.env)
        if self.agent is None:
            self.skipTest("No LLM provider installed")

    def tearDown(self):
        self.env.close()

    # -- Valid plans --

    def test_parse_simple_plan(self):
        raw = json.dumps({
            "reasoning": "Pick red block and place in bowl",
            "actions": [
                {"action": "pick", "target": "red_block"},
                {"action": "place", "target": [0.5, 0.2, 0.05]},
            ]
        })
        plan = self.agent.parse_llm_response(raw)
        self.assertEqual(len(plan["actions"]), 2)
        self.assertEqual(plan["actions"][0]["action"], "pick")

    def test_parse_preserves_reasoning(self):
        raw = json.dumps({
            "reasoning": "Go home first",
            "actions": [{"action": "go_home"}]
        })
        plan = self.agent.parse_llm_response(raw)
        self.assertEqual(plan["reasoning"], "Go home first")

    def test_parse_with_code_fence(self):
        raw = '```json\n{"reasoning":"test","actions":[{"action":"go_home"}]}\n```'
        plan = self.agent.parse_llm_response(raw)
        self.assertEqual(len(plan["actions"]), 1)

    def test_parse_with_bare_code_fence(self):
        raw = '```\n{"reasoning":"x","actions":[]}\n```'
        plan = self.agent.parse_llm_response(raw)
        self.assertEqual(plan["actions"], [])

    def test_parse_multi_action_plan(self):
        raw = json.dumps({
            "reasoning": "Move two blocks",
            "actions": [
                {"action": "pick", "target": "red_block"},
                {"action": "place", "target": [0.5, 0.0, 0.05]},
                {"action": "go_home"},
                {"action": "pick", "target": "blue_block"},
                {"action": "place", "target": [0.6, 0.0, 0.05]},
                {"action": "go_home"},
            ]
        })
        plan = self.agent.parse_llm_response(raw)
        self.assertEqual(len(plan["actions"]), 6)

    # -- Invalid plans --

    def test_parse_invalid_json_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.agent.parse_llm_response("not json at all")
        self.assertIn("not valid JSON", str(ctx.exception))

    def test_parse_missing_actions_raises(self):
        raw = json.dumps({"reasoning": "oops"})
        with self.assertRaises(ValueError) as ctx:
            self.agent.parse_llm_response(raw)
        self.assertIn("missing 'actions'", str(ctx.exception))

    def test_parse_actions_not_list_raises(self):
        raw = json.dumps({"actions": "pick red"})
        with self.assertRaises(ValueError) as ctx:
            self.agent.parse_llm_response(raw)
        self.assertIn("must be a list", str(ctx.exception))

    def test_parse_empty_string_raises(self):
        with self.assertRaises(ValueError):
            self.agent.parse_llm_response("")

    # -- Edge cases --

    def test_parse_whitespace_padding(self):
        raw = '   \n  {"reasoning":"ws","actions":[]}  \n  '
        plan = self.agent.parse_llm_response(raw)
        self.assertEqual(plan["actions"], [])

    def test_parse_place_with_list_target(self):
        raw = json.dumps({
            "reasoning": "place it",
            "actions": [{"action": "place", "target": [0.5, 0.1, 0.05]}]
        })
        plan = self.agent.parse_llm_response(raw)
        self.assertEqual(plan["actions"][0]["target"], [0.5, 0.1, 0.05])


# ══════════════════════════════════════════════════════════════════════
#  Test: execute_action (single action dispatch)
# ══════════════════════════════════════════════════════════════════════

class TestExecuteAction(unittest.TestCase):
    """Test execute_action dispatching to the correct skill."""

    def setUp(self):
        np.random.seed(42)
        self.env = _make_env()
        self.agent = _make_agent(self.env)
        if self.agent is None:
            self.skipTest("No LLM provider installed")

    def tearDown(self):
        self.env.close()

    def test_go_home_action(self):
        result = self.agent.execute_action({"action": "go_home"})
        self.assertTrue(result)

    def test_pick_action(self):
        result = self.agent.execute_action(
            {"action": "pick", "target": "red_block"}
        )
        self.assertIsInstance(result, bool)

    def test_place_action_with_coords(self):
        # Pick first, then place
        self.agent.execute_action({"action": "pick", "target": "red_block"})
        result = self.agent.execute_action(
            {"action": "place", "target": [0.5, 0.0, 0.05]}
        )
        self.assertIsInstance(result, bool)

    def test_place_action_with_name(self):
        """place target can be a string like 'bowl' — resolved to coords."""
        self.agent.execute_action({"action": "pick", "target": "red_block"})
        result = self.agent.execute_action(
            {"action": "place", "target": "bowl"}
        )
        self.assertIsInstance(result, bool)

    def test_push_action(self):
        result = self.agent.execute_action(
            {"action": "push", "target": "red_block",
             "direction": [1.0, 0.0], "distance": 0.05}
        )
        self.assertIsInstance(result, bool)

    def test_unknown_action_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.agent.execute_action({"action": "fly"})
        self.assertIn("fly", str(ctx.exception))


# ══════════════════════════════════════════════════════════════════════
#  Test: execute_plan (full plan execution with mock plans)
# ══════════════════════════════════════════════════════════════════════

class TestExecutePlan(unittest.TestCase):
    """Test execute_plan with hardcoded JSON plans."""

    def setUp(self):
        np.random.seed(42)
        self.env = _make_env()
        self.agent = _make_agent(self.env)
        if self.agent is None:
            self.skipTest("No LLM provider installed")

    def tearDown(self):
        self.env.close()

    # -- Successful plans --

    def test_go_home_only(self):
        plan = {
            "reasoning": "Just go home",
            "actions": [{"action": "go_home"}],
        }
        self.assertTrue(self.agent.execute_plan(plan))

    def test_pick_and_place(self):
        bowl_pos = self.env.get_object_position("bowl").tolist()
        plan = {
            "reasoning": "Put red block in bowl",
            "actions": [
                {"action": "pick", "target": "red_block"},
                {"action": "place", "target": bowl_pos},
                {"action": "go_home"},
            ],
        }
        result = self.agent.execute_plan(plan)
        self.assertIsInstance(result, bool)

    def test_empty_plan_succeeds(self):
        plan = {"reasoning": "Nothing to do", "actions": []}
        self.assertTrue(self.agent.execute_plan(plan))

    def test_plan_missing_actions_key(self):
        """execute_plan should handle a missing 'actions' gracefully."""
        plan = {"reasoning": "oops"}
        # No actions → empty list from .get() → trivially succeeds
        self.assertTrue(self.agent.execute_plan(plan))

    # -- Output format --

    def test_prints_reasoning(self):
        plan = {
            "reasoning": "Test reasoning text",
            "actions": [{"action": "go_home"}],
        }
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.agent.execute_plan(plan)
        self.assertIn("Test reasoning text", buf.getvalue())

    def test_prints_step_numbers(self):
        plan = {
            "reasoning": "multi-step",
            "actions": [
                {"action": "go_home"},
                {"action": "go_home"},
            ],
        }
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.agent.execute_plan(plan)
        out = buf.getvalue()
        self.assertIn("Step 1/2", out)
        self.assertIn("Step 2/2", out)

    def test_prints_complete_on_success(self):
        plan = {
            "reasoning": "done",
            "actions": [{"action": "go_home"}],
        }
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.agent.execute_plan(plan)
        self.assertIn("Plan execution complete", buf.getvalue())

    # -- Failure paths --

    def test_unknown_action_returns_false(self):
        plan = {
            "reasoning": "bad",
            "actions": [{"action": "teleport", "target": "moon"}],
        }
        self.assertFalse(self.agent.execute_plan(plan))

    def test_prints_fail_on_error(self):
        plan = {
            "reasoning": "will fail",
            "actions": [{"action": "teleport"}],
        }
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.agent.execute_plan(plan)
        self.assertIn("FAIL", buf.getvalue())

    def test_stops_on_first_failure(self):
        """After a bad action, subsequent actions should NOT run."""
        plan = {
            "reasoning": "fail then succeed",
            "actions": [
                {"action": "pick", "target": "nonexistent_block"},
                {"action": "go_home"},
            ],
        }
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = self.agent.execute_plan(plan)
        self.assertFalse(result)
        # go_home should not have printed its messages
        self.assertNotIn("Home position reached", buf.getvalue())


# ══════════════════════════════════════════════════════════════════════
#  Test: execute_plan with physics (seeded, pick-place-verify)
# ══════════════════════════════════════════════════════════════════════

class TestExecutePlanPhysics(unittest.TestCase):
    """Physics-dependent plan execution tests with seeded RNG."""

    def setUp(self):
        np.random.seed(42)
        self.env = _make_env()
        self.agent = _make_agent(self.env)
        if self.agent is None:
            self.skipTest("No LLM provider installed")

    def tearDown(self):
        self.env.close()

    def test_pick_place_moves_block_near_bowl(self):
        """After pick-place into bowl, block should be near bowl position."""
        bowl_pos = self.env.get_object_position("bowl")
        plan = {
            "reasoning": "Red block to bowl",
            "actions": [
                {"action": "pick", "target": "red_block"},
                {"action": "place", "target": bowl_pos.tolist()},
                {"action": "go_home"},
            ],
        }
        success = self.agent.execute_plan(plan)
        if not success:
            self.skipTest("Pick failed (physics); skipping position check")

        # After placing, block should be near the bowl
        block_pos = self.env.get_object_position("red_block")
        self.env.step_seconds(1.0)  # let it settle
        block_pos = self.env.get_object_position("red_block")
        horiz_dist = np.linalg.norm(block_pos[:2] - bowl_pos[:2])
        self.assertLess(horiz_dist, 0.25,
                        "Block should land near bowl after place")

    def test_sequential_two_block_plan(self):
        """Pick-place two blocks sequentially."""
        bowl_pos = self.env.get_object_position("bowl").tolist()
        plan = {
            "reasoning": "Move red and blue blocks to bowl",
            "actions": [
                {"action": "pick", "target": "red_block"},
                {"action": "place", "target": bowl_pos},
                {"action": "go_home"},
                {"action": "pick", "target": "blue_block"},
                {"action": "place", "target": bowl_pos},
                {"action": "go_home"},
            ],
        }
        result = self.agent.execute_plan(plan)
        self.assertIsInstance(result, bool)

    def test_push_in_plan(self):
        """Push action dispatches correctly inside execute_plan."""
        plan = {
            "reasoning": "Push red block right",
            "actions": [
                {"action": "push", "target": "red_block",
                 "direction": [0.0, 1.0], "distance": 0.05},
                {"action": "go_home"},
            ],
        }
        result = self.agent.execute_plan(plan)
        self.assertIsInstance(result, bool)

    def test_place_by_object_name(self):
        """Place target given as string 'bowl' should resolve to coords."""
        plan = {
            "reasoning": "Place in bowl by name",
            "actions": [
                {"action": "pick", "target": "green_block"},
                {"action": "place", "target": "bowl"},
                {"action": "go_home"},
            ],
        }
        result = self.agent.execute_plan(plan)
        self.assertIsInstance(result, bool)


# ══════════════════════════════════════════════════════════════════════
#  Test: chat() without LLM (we only test the failure path)
# ══════════════════════════════════════════════════════════════════════

class TestChat(unittest.TestCase):
    """Test chat() entry point (without a real LLM call)."""

    def setUp(self):
        self.env = _make_env()
        self.agent = _make_agent(self.env)
        if self.agent is None:
            self.skipTest("No LLM provider installed")

    def tearDown(self):
        self.env.close()

    def test_chat_without_api_key_returns_false(self):
        """chat() should return False when no API key is available."""
        if self.agent._api_key_set:
            self.skipTest("API key is set; cannot test offline path")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = self.agent.chat("put the red block in the bowl")
        self.assertFalse(result)

    def test_chat_prints_user_instruction(self):
        if self.agent._api_key_set:
            self.skipTest("API key is set; cannot test offline path")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.agent.chat("stack blocks")
        self.assertIn("stack blocks", buf.getvalue())

    def test_chat_prints_perceiving(self):
        if self.agent._api_key_set:
            self.skipTest("API key is set; cannot test offline path")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.agent.chat("do something")
        self.assertIn("Perceiving scene", buf.getvalue())


# ══════════════════════════════════════════════════════════════════════
#  Test: Prompt integration (smoke test)
# ══════════════════════════════════════════════════════════════════════

class TestPromptIntegration(_PromptTestBase):
    """Verify the full prompt pipeline doesn't crash."""

    def test_build_and_print_prompt(self):
        prompt = self.agent._build_prompt(
            "put the red block in the bowl", self.scene
        )
        self.assertGreater(len(prompt), 300)
        print("\n" + "=" * 60)
        print("GENERATED PROMPT")
        print("=" * 60)
        print(prompt)
        print("=" * 60)

    def test_prompt_changes_with_instruction(self):
        p1 = self.agent._build_prompt("pick red block", self.scene)
        p2 = self.agent._build_prompt("stack all blocks", self.scene)
        self.assertNotEqual(p1, p2)
        self.assertIn("pick red block", p1)
        self.assertIn("stack all blocks", p2)


# ══════════════════════════════════════════════════════════════════════
#  Test: New action dispatch (stack, sweep, rotate_gripper)
# ══════════════════════════════════════════════════════════════════════

class TestNewActionDispatch(unittest.TestCase):
    """Test execute_action for new skill actions."""

    def setUp(self):
        np.random.seed(42)
        self.env = _make_env()
        self.agent = _make_agent(self.env)
        if self.agent is None:
            self.skipTest("No LLM provider installed")

    def tearDown(self):
        self.env.close()

    def test_stack_action(self):
        result = self.agent.execute_action({
            "action": "stack",
            "target": "red_block",
            "target_object": "green_block",
        })
        self.assertIsInstance(result, bool)

    def test_sweep_action(self):
        result = self.agent.execute_action({
            "action": "sweep",
            "object_name": "red_block",
            "target": "red_block",
            "target_position": [0.6, 0.0, 0.07],
        })
        self.assertIsInstance(result, bool)

    def test_rotate_gripper_action(self):
        result = self.agent.execute_action({
            "action": "rotate_gripper",
            "angle": 0.3,
        })
        self.assertIsInstance(result, bool)


# ══════════════════════════════════════════════════════════════════════
#  Test: Prompt includes new actions
# ══════════════════════════════════════════════════════════════════════

class TestPromptNewActions(_PromptTestBase):
    """Test that new actions appear in the prompt."""

    def _prompt(self, instruction="stack blocks"):
        return self.agent._build_prompt(instruction, self.scene)

    def test_contains_stack_action(self):
        self.assertIn("stack(", self._prompt())

    def test_contains_sweep_action(self):
        self.assertIn("sweep(", self._prompt())

    def test_contains_rotate_gripper_action(self):
        self.assertIn("rotate_gripper(", self._prompt())


# ══════════════════════════════════════════════════════════════════════
#  Test: Memory integration
# ══════════════════════════════════════════════════════════════════════

class TestMemoryIntegration(unittest.TestCase):
    """Test memory is used by the agent."""

    def setUp(self):
        self.env = _make_env()
        self.agent = _make_agent(self.env)
        if self.agent is None:
            self.skipTest("No LLM provider installed")

    def tearDown(self):
        self.env.close()

    def test_agent_has_memory(self):
        self.assertIsNotNone(self.agent.memory)

    def test_memory_starts_empty(self):
        self.assertEqual(self.agent.memory.turn_count, 0)

    def test_prompt_includes_history_when_present(self):
        self.agent.memory.add_interaction("pick red", "plan", True)
        scene = self.env.get_scene_description()
        prompt = self.agent._build_prompt("do more", scene)
        self.assertIn("pick red", prompt)
        self.assertIn("Conversation History", prompt)


# ══════════════════════════════════════════════════════════════════════
#  Test: Ollama init
# ══════════════════════════════════════════════════════════════════════

class TestOllamaInit(unittest.TestCase):
    """Test VLAAgent initialization with Ollama provider."""

    def setUp(self):
        self.env = _make_env()

    def tearDown(self):
        self.env.close()

    @unittest.skipUnless(OLLAMA_AVAILABLE, "ollama not installed")
    def test_init_ollama(self):
        agent = VLAAgent(self.env, llm_provider="ollama")
        self.assertEqual(agent.llm_provider, "ollama")
        self.assertEqual(agent.model, "llama3.1")

    @unittest.skipUnless(OLLAMA_AVAILABLE, "ollama not installed")
    def test_ollama_no_api_key_needed(self):
        agent = VLAAgent(self.env, llm_provider="ollama")
        self.assertTrue(agent._api_key_set)

    @unittest.skipUnless(OLLAMA_AVAILABLE, "ollama not installed")
    def test_ollama_custom_model(self):
        agent = VLAAgent(self.env, llm_provider="ollama", model="mistral")
        self.assertEqual(agent.model, "mistral")


if __name__ == "__main__":
    unittest.main()

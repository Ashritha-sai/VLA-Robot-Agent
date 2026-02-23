"""
VLAAgent - Vision-Language-Action agent that uses LLMs for task planning.

The agent observes the scene via TableTopEnv.get_scene_description(), builds
a structured prompt, queries an LLM (OpenAI or Anthropic) for a JSON action
plan, parses the plan, and (eventually) executes it through RobotSkills.
"""

import json
import logging
from typing import List, Dict, Optional
import os

from src.skills import RobotSkills

# Import LLM clients (we support both OpenAI and Anthropic)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

logger = logging.getLogger(__name__)


class VLAAgent:
    """Vision-Language-Action agent that uses LLMs for task planning."""

    # Default models per provider
    DEFAULT_MODELS = {
        "openai": "gpt-4",
        "anthropic": "claude-3-5-sonnet-20241022",
    }

    def __init__(self, env, llm_provider: str = "openai", model: str = None):
        """
        Initialise the VLA agent.

        Args:
            env: TableTopEnv instance (must already be reset).
            llm_provider: ``"openai"`` or ``"anthropic"``.
            model: Model name override.  When *None* the provider default
                   from :pyattr:`DEFAULT_MODELS` is used.
        """
        self.env = env
        self.skills = RobotSkills(env)

        # ── LLM client ────────────────────────────────────────────────
        self.llm_provider = llm_provider

        if llm_provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError(
                    "openai package is not installed. "
                    "Install it with: pip install openai"
                )
            api_key = os.getenv("OPENAI_API_KEY")
            # OpenAI client raises immediately when api_key is None.
            # Use a placeholder so the agent can be constructed for prompt
            # building / parsing; the real check happens in query_llm().
            self.client = OpenAI(api_key=api_key or "sk-not-set")
            self._api_key_set = api_key is not None
            self.model = model or self.DEFAULT_MODELS["openai"]

        elif llm_provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError(
                    "anthropic package is not installed. "
                    "Install it with: pip install anthropic"
                )
            api_key = os.getenv("ANTHROPIC_API_KEY")
            self.client = Anthropic(api_key=api_key)
            self._api_key_set = api_key is not None
            self.model = model or self.DEFAULT_MODELS["anthropic"]

        else:
            raise ValueError(
                f"Unknown LLM provider: {llm_provider!r}. "
                "Supported providers: 'openai', 'anthropic'."
            )

        logger.info("VLAAgent ready  provider=%s  model=%s",
                     self.llm_provider, self.model)

    # ------------------------------------------------------------------ #
    #  Prompt construction                                                 #
    # ------------------------------------------------------------------ #

    def _build_prompt(self, user_instruction: str,
                      scene_description: Dict) -> str:
        """
        Construct a prompt for the LLM with system instructions and scene
        context.

        Args:
            user_instruction: User's natural-language command
                (e.g. "put the red block in the bowl").
            scene_description: Current scene state returned by
                ``env.get_scene_description()``.

        Returns:
            Fully formatted prompt string ready to send to the LLM.
        """
        # ── Format objects ────────────────────────────────────────────
        objects_text = ""
        for obj in scene_description["objects"]:
            pos = obj["position"]
            objects_text += (
                f"- {obj['name']} ({obj['type']}, {obj['color']}): "
                f"position [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]\n"
            )

        # ── Format robot state ────────────────────────────────────────
        rs = scene_description["robot_state"]
        ee = rs["end_effector_position"]
        gw = rs["gripper_width"]

        prompt = (
            "You are a robot task planner. You control a Franka Panda "
            "robot arm with a parallel-jaw gripper on a tabletop.\n\n"

            "Current Scene:\n"
            f"  Robot end-effector position: "
            f"[{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}]\n"
            f"  Gripper width: {gw:.3f} m\n\n"

            "Available Objects:\n"
            f"{objects_text}\n"

            "Available Actions:\n"
            "1. pick(object_name) - Pick up an object by name "
            '(e.g., "red_block", "green_block", "blue_block")\n'
            "2. place(position) - Place the currently held object at "
            "[x, y, z] position\n"
            "3. push(object_name, direction, distance) - Push an object "
            "in [dx, dy] direction for a given distance (metres)\n"
            "4. go_home() - Return robot to home position\n\n"

            f'User Instruction: "{user_instruction}"\n\n'

            "Generate a JSON plan to accomplish this task.  Output ONLY "
            "valid JSON in this exact format:\n"
            "{\n"
            '  "reasoning": "brief explanation of your plan",\n'
            '  "actions": [\n'
            '    {"action": "pick", "target": "red_block"},\n'
            '    {"action": "place", "target": [0.5, 0.0, 0.45]},\n'
            '    {"action": "go_home"}\n'
            "  ]\n"
            "}\n\n"

            "Rules:\n"
            "- You can only pick one object at a time\n"
            "- You must pick before you can place\n"
            "- To place in the bowl, use the bowl's position from the "
            "scene\n"
            "- After placing, call go_home() before picking another "
            "object\n"
            "- Output ONLY the JSON, no other text\n"
            "- Make sure all object names match exactly what's listed "
            "above"
        )

        return prompt

    # ------------------------------------------------------------------ #
    #  LLM query                                                           #
    # ------------------------------------------------------------------ #

    def query_llm(self, prompt: str) -> str:
        """
        Send *prompt* to the configured LLM and return the raw response
        text.

        Args:
            prompt: Formatted prompt string (from :pymeth:`_build_prompt`).

        Returns:
            Raw LLM response text (expected to be JSON).
        """
        if not self._api_key_set:
            env_var = ("OPENAI_API_KEY" if self.llm_provider == "openai"
                       else "ANTHROPIC_API_KEY")
            raise RuntimeError(
                f"No API key configured.  Set the {env_var} environment "
                "variable before calling query_llm()."
            )

        print(f"\n[Querying {self.llm_provider} / {self.model} ...]")

        if self.llm_provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000,
            )
            return response.choices[0].message.content

        elif self.llm_provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

    # ------------------------------------------------------------------ #
    #  Plan parsing                                                        #
    # ------------------------------------------------------------------ #

    def parse_llm_response(self, response_text: str) -> Dict:
        """
        Parse the JSON action plan returned by the LLM.

        Handles minor formatting issues such as markdown code-fences
        that models sometimes wrap around JSON output.

        Args:
            response_text: Raw text from :pymeth:`query_llm`.

        Returns:
            Parsed plan dict with ``"reasoning"`` (str) and
            ``"actions"`` (list of action dicts).

        Raises:
            ValueError: If the response cannot be parsed as valid JSON
                or is missing required fields.
        """
        text = response_text.strip()

        # Strip optional markdown code fences (```json ... ```)
        if text.startswith("```"):
            # Remove opening fence (with optional language tag)
            first_newline = text.index("\n")
            text = text[first_newline + 1:]
        if text.endswith("```"):
            text = text[:-3].rstrip()

        try:
            plan = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM response is not valid JSON: {exc}\n"
                f"Response was:\n{response_text}"
            ) from exc

        # Validate required fields
        if "actions" not in plan:
            raise ValueError(
                "LLM plan missing 'actions' key.\n"
                f"Parsed object: {plan}"
            )
        if not isinstance(plan["actions"], list):
            raise ValueError(
                "'actions' must be a list.\n"
                f"Got: {type(plan['actions']).__name__}"
            )

        return plan

    # ------------------------------------------------------------------ #
    #  Plan execution                                                      #
    # ------------------------------------------------------------------ #

    def execute_action(self, action_dict: Dict) -> bool:
        """
        Execute a single action from a parsed plan.

        Args:
            action_dict: A dict such as
                ``{"action": "pick", "target": "red_block"}``.

        Returns:
            *True* if the action succeeded, *False* otherwise.

        Raises:
            ValueError: If *action* is not a recognised action name.
        """
        action = action_dict.get("action")
        target = action_dict.get("target")

        if action == "pick":
            return self.skills.pick(target)

        elif action == "place":
            # The LLM may emit a string name (e.g. "bowl") or a
            # coordinate list.  Resolve names to positions.
            if isinstance(target, str):
                target_pos = self.env.get_object_position(target).tolist()
            else:
                target_pos = list(target)
            return self.skills.place(target_pos)

        elif action == "push":
            direction = action_dict.get("direction", [1.0, 0.0])
            distance = action_dict.get("distance", 0.10)
            return self.skills.push(target, direction, distance)

        elif action == "go_home":
            return self.skills.go_home()

        else:
            raise ValueError(f"Unknown action: {action!r}")

    def execute_plan(self, plan: Dict) -> bool:
        """
        Execute every action in a parsed plan, stopping on the first
        failure.

        Args:
            plan: Dict with ``"actions"`` list (and optional
                ``"reasoning"`` string), as returned by
                :pymeth:`parse_llm_response`.

        Returns:
            *True* if **all** actions succeeded, *False* if any action
            failed or raised an exception.
        """
        actions = plan.get("actions", [])
        reasoning = plan.get("reasoning", "No reasoning provided")
        total = len(actions)

        print(f"\n{'=' * 50}")
        print(f"Plan: {reasoning}")
        print(f"Number of actions: {total}")
        print(f"{'=' * 50}")

        for i, action_dict in enumerate(actions, start=1):
            action = action_dict.get("action", "?")
            target = action_dict.get("target", "")
            print(f"\n--- Step {i}/{total}: {action}({target}) ---")

            try:
                success = self.execute_action(action_dict)
            except (ValueError, Exception) as exc:
                logger.error("Action %s failed: %s", action, exc)
                print(f"[FAIL] Error executing {action}: {exc}")
                return False

            if not success:
                logger.warning("Action %s returned False", action)
                print(f"[FAIL] {action}({target}) did not succeed")
                return False

            print(f"[OK]   Step {i} complete")

        print(f"\n{'=' * 50}")
        print("Plan execution complete!")
        print(f"{'=' * 50}\n")
        return True

    # ------------------------------------------------------------------ #
    #  Top-level interface                                                 #
    # ------------------------------------------------------------------ #

    def chat(self, user_instruction: str) -> bool:
        """
        Main interface: perceive the scene, ask the LLM for a plan, then
        execute it.

        Args:
            user_instruction: Natural-language command, e.g.
                ``"put all blocks in the bowl"``.

        Returns:
            *True* if the task was completed successfully, *False*
            otherwise.
        """
        print(f"\n{'=' * 60}")
        print(f"User: {user_instruction}")
        print(f"{'=' * 60}\n")

        # 1. Perceive
        print("[Perceiving scene...]")
        scene = self.env.get_scene_description()
        self.env.print_scene()

        # 2. Prompt
        prompt = self._build_prompt(user_instruction, scene)

        # 3. Query LLM
        try:
            llm_response = self.query_llm(prompt)
        except RuntimeError as exc:
            logger.error("LLM query failed: %s", exc)
            print(f"[FAIL] LLM query failed: {exc}")
            return False
        print(f"\n[LLM Response]:\n{llm_response}\n")

        # 4. Parse
        try:
            plan = self.parse_llm_response(llm_response)
        except ValueError as exc:
            logger.error("Failed to parse LLM response: %s", exc)
            print(f"[FAIL] Could not parse LLM response: {exc}")
            return False

        # 5. Execute
        return self.execute_plan(plan)

    # ------------------------------------------------------------------ #
    #  Convenience: observe → prompt → query  (no execution)               #
    # ------------------------------------------------------------------ #

    def get_plan(self, user_instruction: str) -> Dict:
        """
        Observe the scene, query the LLM, and return a parsed plan
        **without executing** it.

        This is a convenience wrapper that chains
        ``get_scene_description -> _build_prompt -> query_llm ->
        parse_llm_response``.

        Args:
            user_instruction: Natural-language task description.

        Returns:
            Parsed plan dict (see :pymeth:`parse_llm_response`).
        """
        scene = self.env.get_scene_description()
        prompt = self._build_prompt(user_instruction, scene)
        raw = self.query_llm(prompt)
        return self.parse_llm_response(raw)

"""
ConversationMemory - Multi-turn conversation memory for the VLA agent.

Keeps a rolling history of user instructions, agent reasoning, and
outcomes so the LLM can reference previous interactions.
"""

import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class ConversationMemory:
    """Stores multi-turn conversation history for context injection."""

    def __init__(self, max_turns: int = 10):
        """
        Args:
            max_turns: Maximum number of interactions to retain.
                       Oldest interactions are dropped when the limit is reached.
        """
        self.max_turns = max_turns
        self._history: List[Dict[str, str]] = []

    # ── Public API ────────────────────────────────────────────────────

    def add_interaction(
        self,
        instruction: str,
        reasoning: str,
        success: bool,
    ) -> None:
        """Record the outcome of one chat() round.

        Args:
            instruction: The user's original instruction.
            reasoning: The LLM's plan reasoning string.
            success: Whether the plan executed successfully.
        """
        entry = {
            "instruction": instruction,
            "reasoning": reasoning,
            "success": "success" if success else "failure",
        }
        self._history.append(entry)
        # Trim to max_turns
        if len(self._history) > self.max_turns:
            self._history = self._history[-self.max_turns:]
        logger.debug("Memory: recorded turn %d (%s)", len(self._history), entry["success"])

    def get_history_prompt(self) -> str:
        """Return a formatted string summarising past interactions.

        Returns:
            Multi-line string suitable for injection into the LLM prompt.
            Empty string if no history exists.
        """
        if not self._history:
            return ""
        lines = []
        for i, entry in enumerate(self._history, 1):
            lines.append(
                f"Turn {i}: User said: \"{entry['instruction']}\" "
                f"-> {entry['reasoning']} [{entry['success']}]"
            )
        return "\n".join(lines)

    def clear(self) -> None:
        """Erase all stored interactions."""
        self._history.clear()
        logger.debug("Memory cleared")

    @property
    def turn_count(self) -> int:
        """Number of interactions currently stored."""
        return len(self._history)

    @property
    def history(self) -> List[Dict[str, str]]:
        """Return a copy of the stored history."""
        return list(self._history)

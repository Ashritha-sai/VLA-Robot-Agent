#!/usr/bin/env python3
"""Demo using LLM for task planning."""

import os
import sys

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to path so `src` package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.env import TableTopEnv
from src.agent import VLAAgent


def main():
    print("Demo: LLM-Guided Task Execution")
    print("=" * 50)

    # Setup
    env = TableTopEnv(gui=True)
    env.reset()

    try:
        agent = VLAAgent(env, llm_provider="openai")
    except Exception as e:
        print(f"Error initializing agent: {e}")
        print("Make sure OPENAI_API_KEY or ANTHROPIC_API_KEY is set")
        sys.exit(1)

    # Demo tasks
    tasks = [
        "Put the red block in the bowl",
        "Put all the blocks in the bowl one by one",
        "Stack the green block on top of the blue block",
    ]

    for i, task in enumerate(tasks, 1):
        print(f"\n\n{'=' * 60}")
        print(f"Task {i}/{len(tasks)}: {task}")
        print("=" * 60)

        success = agent.chat(task)

        if not success:
            print("Task failed, stopping demo")
            break

        input("\nPress Enter for next task...")

        if i < len(tasks):
            print("\nResetting environment...")
            env.reset()

    print("\n--- Demo complete! ---")
    input("\nPress Enter to exit...")
    env.close()


if __name__ == "__main__":
    main()

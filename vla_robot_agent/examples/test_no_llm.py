#!/usr/bin/env python3
"""Test robot skills without LLM - executes a hardcoded action plan with
success checking, post-place verification, and retry logic."""

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
from src.skills import RobotSkills
import numpy as np
import json

BOWL_PROXIMITY = 0.12   # metres — block must be this close to bowl centre
MAX_ATTEMPTS = 3         # retries per block


def verify_in_bowl(env, block_name):
    """Return True if *block_name* is within BOWL_PROXIMITY of the bowl."""
    block_pos = env.get_object_position(block_name)
    bowl_pos = env.get_object_position("bowl")
    dist = np.linalg.norm(block_pos[:2] - bowl_pos[:2])
    return dist < BOWL_PROXIMITY


def pick_and_place(env, skills, block_name):
    """Pick *block_name* and place it in the bowl, retrying on failure."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        tag = f" (attempt {attempt}/{MAX_ATTEMPTS})" if attempt > 1 else ""
        print(f"\n--- pick({block_name}){tag} ---")

        picked = skills.pick(block_name)
        if not picked:
            print(f"   >> Pick FAILED for {block_name}.")
            if attempt < MAX_ATTEMPTS:
                print("   >> Returning home and retrying...")
                skills.go_home()
                continue
            print(f"   >> Giving up on {block_name}.")
            return False

        # Place in bowl
        bowl_pos = env.get_object_position("bowl")
        print(f"\n--- place({block_name} -> bowl) ---")
        placed = skills.place(bowl_pos.tolist())
        if not placed:
            print(f"   >> Place motion FAILED for {block_name}.")
            return False

        # Verify block actually landed near the bowl
        if verify_in_bowl(env, block_name):
            print(f"   >> Verified: {block_name} is in the bowl!")
            return True
        else:
            actual = env.get_object_position(block_name)
            print(f"   >> Verification FAILED: {block_name} slipped to "
                  f"[{actual[0]:.3f}, {actual[1]:.3f}, {actual[2]:.3f}].")
            if attempt < MAX_ATTEMPTS:
                print("   >> Returning home and retrying...")
                skills.go_home()
            else:
                print(f"   >> Giving up on {block_name} after {MAX_ATTEMPTS} attempts.")

    return False


def main():
    print("Test: Hardcoded Plan Execution (No LLM)")
    print("=" * 50)

    env = TableTopEnv(gui=True)
    env.reset()
    skills = RobotSkills(env)

    blocks = ["red_block", "green_block", "blue_block"]

    # Hardcoded plan (simulating LLM output)
    plan = {
        "reasoning": "Pick each block and place in bowl sequentially",
        "actions": (
            [{"action": "go_home", "target": None}]
            + [step
               for b in blocks
               for step in ({"action": "pick", "target": b},
                            {"action": "place", "target": "bowl"})]
            + [{"action": "go_home", "target": None}]
        ),
    }

    print(f"\nPlan: {plan['reasoning']}")
    print(f"Actions: {json.dumps(plan['actions'], indent=2)}")

    # --- Execute with verification ---
    print("\n\n--- Execution (with verification & retry) ---\n")

    skills.go_home()

    results = {}
    for block in blocks:
        ok = pick_and_place(env, skills, block)
        results[block] = ok
        # Always go home between blocks
        skills.go_home()

    # --- Summary ---
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    total_ok = 0
    for block, ok in results.items():
        status = "IN BOWL" if ok else "FAILED"
        if ok:
            total_ok += 1
        print(f"  {block}: {status}")
    print(f"\n  {total_ok}/{len(blocks)} blocks placed successfully.")

    print("\nFinal scene:")
    env.print_scene()

    print("\n--- Test complete! ---")
    input("\nPress Enter to exit...")
    env.close()


if __name__ == "__main__":
    main()

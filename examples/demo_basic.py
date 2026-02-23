#!/usr/bin/env python3
"""Basic demo showing pick and place operations with success verification."""

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
import time

# How close (metres) a block must be to the bowl to count as "in the bowl"
BOWL_PROXIMITY = 0.12
MAX_PICK_ATTEMPTS = 3


def verify_in_bowl(env, block_name):
    """Check whether *block_name* is close to the bowl after placement."""
    block_pos = env.get_object_position(block_name)
    bowl_pos = env.get_object_position("bowl")
    dist = np.linalg.norm(block_pos[:2] - bowl_pos[:2])
    return dist < BOWL_PROXIMITY


def pick_and_place_to_bowl(env, skills, block_name, step_base):
    """
    Pick *block_name* and place it in the bowl.

    Retries the pick up to MAX_PICK_ATTEMPTS times if the grasp fails or
    the block slips.  After placing, verifies the block actually landed
    near the bowl.

    Returns (success: bool, steps_used: int).
    """
    bowl_pos = env.get_object_position("bowl")
    step = step_base

    for attempt in range(1, MAX_PICK_ATTEMPTS + 1):
        # --- Pick ---
        step += 1
        label = f"  (attempt {attempt}/{MAX_PICK_ATTEMPTS})" if attempt > 1 else ""
        print(f"\n{step}. Picking {block_name}...{label}")
        picked = skills.pick(block_name)
        if not picked:
            print(f"   >> Pick failed for {block_name}, ", end="")
            if attempt < MAX_PICK_ATTEMPTS:
                print("returning home and retrying...")
                skills.go_home()
                time.sleep(0.5)
                continue
            else:
                print("giving up after max retries.")
                return False, step

        time.sleep(0.5)

        # --- Place ---
        step += 1
        print(f"\n{step}. Placing {block_name} in bowl...")
        placed = skills.place(bowl_pos.tolist())
        if not placed:
            print(f"   >> Place motion failed for {block_name}.")
            return False, step

        time.sleep(0.5)

        # --- Verify ---
        in_bowl = verify_in_bowl(env, block_name)
        if in_bowl:
            print(f"   >> Verified: {block_name} is in the bowl!")
            return True, step
        else:
            block_pos = env.get_object_position(block_name)
            print(f"   >> Verification FAILED: {block_name} is at "
                  f"[{block_pos[0]:.3f}, {block_pos[1]:.3f}, {block_pos[2]:.3f}], "
                  f"not in the bowl.")
            if attempt < MAX_PICK_ATTEMPTS:
                print("   >> Returning home and retrying...")
                skills.go_home()
                time.sleep(0.5)
            else:
                print(f"   >> Giving up on {block_name} after {MAX_PICK_ATTEMPTS} attempts.")

    return False, step


def main():
    print("Demo: Basic Pick and Place")
    print("=" * 50)

    # Setup environment
    env = TableTopEnv(gui=True)
    env.reset()
    skills = RobotSkills(env)

    # Show initial scene
    print("\nInitial Scene:")
    env.print_scene()

    results = {}
    step = 0

    # Step 1 — Home
    step += 1
    print(f"\n{step}. Going to home position...")
    skills.go_home()
    time.sleep(1)

    # Pick-and-place red block
    ok, step = pick_and_place_to_bowl(env, skills, "red_block", step)
    results["red_block"] = ok

    # Return home between blocks
    step += 1
    print(f"\n{step}. Returning home...")
    skills.go_home()
    time.sleep(0.5)

    # Pick-and-place green block
    ok, step = pick_and_place_to_bowl(env, skills, "green_block", step)
    results["green_block"] = ok

    # Final home
    step += 1
    print(f"\n{step}. Returning home...")
    skills.go_home()

    # --- Summary ---
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    for block, ok in results.items():
        status = "IN BOWL" if ok else "FAILED"
        print(f"  {block}: {status}")

    print("\nFinal scene:")
    env.print_scene()

    input("\nPress Enter to exit...")
    env.close()


if __name__ == "__main__":
    main()

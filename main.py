#!/usr/bin/env python3
"""
VLA Robot Agent - Interactive Demo
"""

import argparse
import os
import sys
from src.env import TableTopEnv
from src.agent import VLAAgent
from colorama import init, Fore, Style

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Initialize colorama for colored terminal output
init(autoreset=True)

# Trajectory recorder (lazy-loaded when commands are used)
_recorder = None


def print_banner():
    """Print welcome banner"""
    banner = f"""
{Fore.CYAN}{'='*70}
   Vision-Language-Action Robot Agent
   Embodied AI Demo with Franka Panda
{'='*70}{Style.RESET_ALL}
"""
    print(banner)

def print_help():
    """Print available commands"""
    help_text = f"""
{Fore.YELLOW}Available Commands:{Style.RESET_ALL}
  Type any natural language instruction (e.g., "Put the red block in the bowl")
  'reset'                   - Reset environment with new random object positions
  'scene'                   - Display current scene description
  'objects'                 - List all tracked objects
  'add <type> <color> [x y z]' - Add object (e.g., 'add block yellow 0.6 0.1 0.07')
  'remove <name>'           - Remove an object by name
  'record start'            - Start recording trajectory
  'record stop'             - Stop recording trajectory
  'record save <path>'      - Save recorded trajectory to file
  'record play <path>'      - Play back a saved trajectory
  'help'                    - Show this help message
  'quit' or 'exit'          - Exit the program

{Fore.YELLOW}Example Instructions:{Style.RESET_ALL}
  "Put all the blocks in the bowl"
  "Stack the red block on the green block"
  "Move the blue block to the left side of the table"
  "Put the green block in the bowl, then go home"
"""
    print(help_text)


def _handle_add(env, parts):
    """Handle the 'add <type> <color> [x y z]' command."""
    if len(parts) < 3:
        print(f"{Fore.RED}Usage: add <type> <color> [x y z]{Style.RESET_ALL}")
        return
    obj_type = parts[1]
    color = parts[2]
    position = None
    if len(parts) >= 6:
        try:
            position = [float(parts[3]), float(parts[4]), float(parts[5])]
        except ValueError:
            print(f"{Fore.RED}Invalid coordinates. Usage: add <type> <color> [x y z]{Style.RESET_ALL}")
            return
    # Auto-generate name: color_type or color_type_2, etc.
    base_name = f"{color}_{obj_type}"
    name = base_name
    counter = 2
    while name in env.objects:
        name = f"{base_name}_{counter}"
        counter += 1
    try:
        env.add_object(name, obj_type, color, position)
        print(f"{Fore.GREEN}Added '{name}' ({obj_type}, {color}){Style.RESET_ALL}")
    except ValueError as e:
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")


def _handle_remove(env, parts):
    """Handle the 'remove <name>' command."""
    if len(parts) < 2:
        print(f"{Fore.RED}Usage: remove <name>{Style.RESET_ALL}")
        return
    name = parts[1]
    try:
        env.remove_object(name)
        print(f"{Fore.GREEN}Removed '{name}'{Style.RESET_ALL}")
    except ValueError as e:
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")


def _handle_record(env, skills, parts):
    """Handle the 'record <start|stop|save|play>' commands."""
    global _recorder
    if len(parts) < 2:
        print(f"{Fore.RED}Usage: record <start|stop|save <path>|play <path>>{Style.RESET_ALL}")
        return

    from src.skills.trajectory import TrajectoryRecorder

    sub = parts[1].lower()
    if sub == "start":
        _recorder = TrajectoryRecorder(env)
        _recorder.start()
        print(f"{Fore.GREEN}Recording started.{Style.RESET_ALL}")
    elif sub == "stop":
        if _recorder is None:
            print(f"{Fore.RED}No recording in progress.{Style.RESET_ALL}")
            return
        _recorder.stop()
        print(f"{Fore.GREEN}Recording stopped ({_recorder.frame_count} frames).{Style.RESET_ALL}")
    elif sub == "save":
        if len(parts) < 3:
            print(f"{Fore.RED}Usage: record save <path>{Style.RESET_ALL}")
            return
        if _recorder is None:
            print(f"{Fore.RED}No recording to save.{Style.RESET_ALL}")
            return
        _recorder.save(parts[2])
        print(f"{Fore.GREEN}Trajectory saved to {parts[2]}.{Style.RESET_ALL}")
    elif sub == "play":
        if len(parts) < 3:
            print(f"{Fore.RED}Usage: record play <path>{Style.RESET_ALL}")
            return
        player = TrajectoryRecorder(env)
        player.load(parts[2])
        print(f"{Fore.GREEN}Playing trajectory from {parts[2]}...{Style.RESET_ALL}")
        player.replay(skills)
    else:
        print(f"{Fore.RED}Unknown record command: {sub}{Style.RESET_ALL}")


def main():
    parser = argparse.ArgumentParser(description='VLA Robot Agent Demo')
    parser.add_argument('--gui', action='store_true', default=True,
                        help='Launch with PyBullet GUI (default: True)')
    parser.add_argument('--llm', type=str, default='openai',
                        choices=['openai', 'anthropic', 'ollama'],
                        help='LLM provider to use')
    parser.add_argument('--model', type=str, default=None,
                        help='Specific model name (optional)')
    parser.add_argument('--vision', action='store_true', default=False,
                        help='Enable vision module for scene perception')

    args = parser.parse_args()

    print_banner()

    # Initialize environment
    print(f"{Fore.GREEN}Initializing PyBullet environment...{Style.RESET_ALL}")
    env = TableTopEnv(gui=args.gui)
    env.reset()

    # Initialize agent
    print(f"{Fore.GREEN}Initializing {args.llm} agent...{Style.RESET_ALL}")
    try:
        agent = VLAAgent(
            env, llm_provider=args.llm, model=args.model,
            use_vision=args.vision,
        )
    except Exception as e:
        print(f"{Fore.RED}Failed to initialize agent: {e}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Make sure you've set your API key:{Style.RESET_ALL}")
        print(f"  export OPENAI_API_KEY='your-key-here'")
        print(f"  or export ANTHROPIC_API_KEY='your-key-here'")
        sys.exit(1)

    print(f"{Fore.GREEN}Ready!{Style.RESET_ALL}\n")
    print_help()

    # Main interaction loop
    while True:
        try:
            # Get user input
            user_input = input(f"\n{Fore.CYAN}You:{Style.RESET_ALL} ").strip()

            if not user_input:
                continue

            lower = user_input.lower()
            parts = user_input.split()

            # Handle commands
            if lower in ['quit', 'exit', 'q']:
                print(f"{Fore.YELLOW}Shutting down...{Style.RESET_ALL}")
                break

            elif lower == 'reset':
                print(f"{Fore.GREEN}Resetting environment...{Style.RESET_ALL}")
                env.reset()
                env.print_scene()

            elif lower == 'scene':
                env.print_scene()

            elif lower == 'objects':
                objs = env.list_objects()
                if not objs:
                    print("No objects in scene.")
                else:
                    for o in objs:
                        pos = env.get_object_position(o["name"])
                        print(f"  {o['name']:20s}  type={o['type']:10s}  color={o['color']:8s}  pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

            elif parts[0].lower() == 'add':
                _handle_add(env, parts)

            elif parts[0].lower() == 'remove':
                _handle_remove(env, parts)

            elif parts[0].lower() == 'record':
                _handle_record(env, agent.skills, parts)

            elif lower == 'help':
                print_help()

            else:
                # Execute user instruction
                agent.chat(user_input)

        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Interrupted. Type 'quit' to exit.{Style.RESET_ALL}")

        except Exception as e:
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()

    # Cleanup
    env.close()
    print(f"{Fore.GREEN}Goodbye!{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
